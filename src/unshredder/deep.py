"""Learned seam compatibility (after Paixão et al., "Self-supervised deep reconstruction
of mixed strip-shredded text documents", Pattern Recognition 2020).

A SqueezeNet classifies 32x32 binary patches straddling a seam (16 columns from each
strip) as valid/invalid. Training samples come for free from simulated shredding of
intact documents. At inference the network slides down the whole seam, logits are
averaged, and compatibility is the best positive probability over small vertical shifts.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from skimage.filters import threshold_sauvola

from .config import Config, ReconstructConfig, TrainConfig
from .generate import load_documents

PATCH = 32
HALF = PATCH // 2


def device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def binarize(img: np.ndarray) -> np.ndarray:
    """Sauvola binarisation: 1.0 = paper, 0.0 = ink."""
    if img.std() < 1e-3:
        return np.ones_like(img, dtype=np.float32)
    return (img > threshold_sauvola(img, window_size=25)).astype(np.float32)


def build_model(pretrained: bool) -> nn.Module:
    from torchvision.models import SqueezeNet1_1_Weights, squeezenet1_1

    model = squeezenet1_1(weights=SqueezeNet1_1_Weights.DEFAULT if pretrained else None)
    conv = nn.Conv2d(512, 2, kernel_size=1)
    nn.init.normal_(conv.weight, 0, 0.01)
    nn.init.zeros_(conv.bias)
    model.classifier[1] = conv
    # torchvision applies ReLU to the logits; with a freshly initialised head both logits
    # clamp to 0 and training stalls at loss ln(2). Raw logits are averaged instead.
    model.classifier[2] = nn.Identity()
    return model


def _forward(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """x: (B, 1, H, 32) in [0, 1] -> (B, 2) logits, averaged over the whole seam."""
    return model((x - 0.5).expand(-1, 3, -1, -1) / 0.5)


# ---------------------------------------------------------------- training data


def _damage(strip: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Replace the 2 outer columns on each side with random noise (shredder damage)."""
    s = strip.copy()
    for cols in (slice(0, 2), slice(-2, None)):
        s[:, cols] = rng.random(s[:, cols].shape) < 0.5
    return s


def _patches(left: np.ndarray, right: np.ndarray, rng, limit: int, min_ink: float) -> list[np.ndarray]:
    h = min(left.shape[0], right.shape[0])
    seam = np.hstack([left[:h, -HALF:], right[:h, :HALF]])
    out = []
    for y in rng.permutation(np.arange(0, h - PATCH, 2)):
        p = seam[y : y + PATCH]
        if (p < 0.5).mean() >= min_ink:
            out.append(p)
            if len(out) >= limit:
                break
    return out


def make_samples(doc: np.ndarray, cfg: TrainConfig, rng: np.random.Generator) -> tuple[list, list]:
    """Positive/negative seam patches from one intact (binarised) document."""
    w = doc.shape[1]
    cuts = np.linspace(0, w, cfg.strips_per_doc + 1).round().astype(int)
    strips = [_damage(doc[:, a:b], rng) for a, b in zip(cuts[:-1], cuts[1:]) if b - a >= HALF]
    n = len(strips)
    if n < 3:
        return [], []
    pos, neg = [], []
    for i in rng.permutation(n - 1):
        pos += _patches(strips[i], strips[i + 1], rng, 50, cfg.min_ink)
        if len(pos) >= cfg.samples_per_doc:
            break
    pos = pos[: cfg.samples_per_doc]
    tries = 0
    while len(neg) < len(pos) and tries < 50 * n:
        tries += 1
        i, j = rng.integers(0, n, 2)
        if i == j or j == i + 1:  # anything but the true order (includes swapped neighbours)
            continue
        neg += _patches(strips[i], strips[j], rng, 10, cfg.min_ink)
    return pos, neg[: len(pos)]


def build_dataset(full: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns patches (N,32,32) uint8, labels (N,), doc ids (N,) for a by-document split.

    Training pages use the `generate` settings but a disjoint seed, so they never
    coincide with the pages later shredded for evaluation."""
    cfg = full.train
    rng = np.random.default_rng(full.seed + 10_000)
    gen_cfg = copy.deepcopy(full.generate)
    gen_cfg.input_dir = cfg.docs_dir
    gen_cfg.num_docs = cfg.num_docs
    X, y, d = [], [], []
    for k, doc in enumerate(load_documents(gen_cfg, rng)):
        arr = binarize(np.asarray(doc, dtype=np.float32) / 255.0)
        pos, neg = make_samples(arr, cfg, rng)
        X += pos + neg
        y += [1] * len(pos) + [0] * len(neg)
        d += [k] * (len(pos) + len(neg))
    X = (np.stack(X) * 255).astype(np.uint8)
    return X, np.array(y, dtype=np.int64), np.array(d)


def train(full: Config) -> Path:
    cfg = full.train
    torch.manual_seed(full.seed)
    dev = device(cfg.device)
    t0 = time.time()
    X, y, docs = build_dataset(full)
    rng = np.random.default_rng(full.seed)
    ids = np.unique(docs)
    val_docs = set(rng.choice(ids, max(1, int(len(ids) * cfg.val_frac)), replace=False).tolist())
    is_val = np.isin(docs, list(val_docs))
    print(f"{len(X)} patches ({y.mean():.0%} positive) from {len(ids)} docs in {time.time() - t0:.0f}s; "
          f"{is_val.sum()} for validation; device={dev}")

    Xt = torch.from_numpy(X).unsqueeze(1)
    yt = torch.from_numpy(y)
    tr_idx, va_idx = np.flatnonzero(~is_val), np.flatnonzero(is_val)

    model = build_model(cfg.pretrained).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()
    out = Path(cfg.model_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(cfg.epochs):
        model.train()
        perm = rng.permutation(tr_idx)
        total = 0.0
        for s in range(0, len(perm), cfg.batch_size):
            b = perm[s : s + cfg.batch_size]
            xb = Xt[b].to(dev).float() / 255
            loss = loss_fn(_forward(model, xb), yt[b].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        acc = _accuracy(model, Xt, yt, va_idx, dev, cfg.batch_size)
        print(f"epoch {epoch + 1}/{cfg.epochs}  loss={total / len(perm):.4f}  val_acc={acc:.4f}  "
              f"[{time.time() - t0:.0f}s]", flush=True)
        if acc > best:
            best = acc
            torch.save({"state_dict": model.state_dict(), "val_acc": acc}, out)
    print(f"saved best model (val_acc={best:.4f}) to {out}")
    return out


@torch.no_grad()
def _accuracy(model, Xt, yt, idx, dev, bs) -> float:
    model.eval()
    correct = 0
    for s in range(0, len(idx), bs * 4):
        b = idx[s : s + bs * 4]
        pred = _forward(model, Xt[b].to(dev).float() / 255).argmax(1).cpu()
        correct += (pred == yt[b]).sum().item()
    return correct / max(1, len(idx))


# ---------------------------------------------------------------- inference

_MODELS: dict[str, nn.Module] = {}


def load_model(path: str, dev: torch.device) -> nn.Module:
    key = f"{path}@{dev}"
    if key not in _MODELS:
        if not Path(path).exists():
            raise FileNotFoundError(f"no trained model at {path}; run `unshredder train` first")
        model = build_model(pretrained=False)
        model.load_state_dict(torch.load(path, map_location="cpu")["state_dict"])
        _MODELS[key] = model.to(dev).eval()
    return _MODELS[key]


@torch.no_grad()
def compatibility(nodes: list[np.ndarray], cfg: ReconstructConfig) -> np.ndarray:
    """C[a, b] = P(node b directly right of node a), maximised over vertical shifts."""
    dev = device(cfg.device)
    model = load_model(cfg.model_path, dev)
    bins = [binarize(im) for im in nodes]
    n = len(bins)
    h = min(b.shape[0] for b in bins)
    if cfg.max_rows and h > cfg.max_rows:  # paper: central 3000 rows only
        top = (h - cfg.max_rows) // 2
        h_used = cfg.max_rows
    else:
        top, h_used = 0, h

    def edge(b, side):
        e = b[:, -HALF:] if side == "r" else b[:, :HALF]
        if e.shape[1] < HALF:  # pad strips narrower than 16 px with paper
            pad = np.ones((e.shape[0], HALF - e.shape[1]), dtype=np.float32)
            e = np.hstack([pad, e] if side == "r" else [e, pad])
        return e

    # Pad rows by max_shift so every shift has a full-height window.
    S = cfg.max_shift
    R = torch.ones(n, h_used, HALF)
    L = torch.ones(n, h_used + 2 * S, HALF)
    for k, b in enumerate(bins):
        R[k] = torch.from_numpy(edge(b, "r")[top : top + h_used])
        Lk = edge(b, "l")
        lo, hi = max(0, top - S), min(b.shape[0], top + h_used + S)
        L[k, S - (top - lo) : S - (top - lo) + (hi - lo)] = torch.from_numpy(Lk[lo:hi])
    R, L = R.to(dev), L.to(dev)

    shifts = range(-S, S + 1, cfg.shift_step)
    C = np.zeros((n, n), dtype=np.float32)
    bs = max(1, cfg.inference_batch // len(shifts))
    use_amp = dev.type == "cuda"
    for a in range(n):
        best = torch.zeros(n, device=dev)
        for s0 in range(0, n, bs):
            bidx = torch.arange(s0, min(n, s0 + bs), device=dev)
            windows = torch.stack([L[bidx, S + s : S + s + h_used] for s in shifts], 1)  # (B, nS, h, 16)
            left = R[a].expand(len(bidx), len(shifts), -1, -1)
            x = torch.cat([left, windows], dim=3).reshape(-1, 1, h_used, PATCH)
            with torch.autocast(dev.type, enabled=use_amp):
                p = torch.softmax(_forward(model, x).float(), 1)[:, 1]
            best[bidx] = p.reshape(len(bidx), len(shifts)).max(1).values
        C[a] = best.cpu().numpy()
    return C
