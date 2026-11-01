"""Baseline: Paixão et al., "Self-supervised deep reconstruction of mixed strip-shredded
text documents" (Pattern Recognition 2020), github.com/thiagopx/deeprec-pr20.

Their released SqueezeNet weights (traindata/<model>/model/<epoch>.npy) ported to PyTorch,
with their inference reproduced from docrec/compatibility/proposed.py:
  * strips with masked-out areas filled white; blank strips dropped (5x5 gaussian blur,
    no pixel < 127)
  * Sauvola binarisation (skimage defaults) of the whole strip
  * seam image: 16 px from each strip's border (following the mask border row by row),
    central 3000 rows, right shred shifted vertically by -10..10 rows (empty rows white)
  * compatibility = max over shifts of softmax P(compatible)
  * ordering: minimum-cost Hamiltonian path on max(C) - C.
Deviation: the path is solved with LKH (elkai) instead of Concorde.

Usage:
  python pr20.py --weights <model>/9.npy --data ../../datasets/D2 --mode single --out results/cdip_D2_single.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_sauvola

import safe_npy
from unshredder import solve
from unshredder.datasets import list_docs
from unshredder.evaluate import evaluate

HALF = 16
ROWS = 3000
VSHIFT = 10


# ---------------------------------------------------------------- model


class Fire(nn.Module):
    def __init__(self, name: str, cin: int, s: int, e: int):
        super().__init__()
        self.name = name
        self.squeeze = nn.Conv2d(cin, s, 1)
        self.e1 = nn.Conv2d(s, e, 1)
        self.e3 = nn.Conv2d(s, e, 3, padding=1)

    def forward(self, x):
        x = torch.relu(self.squeeze(x))
        return torch.cat([torch.relu(self.e1(x)), torch.relu(self.e3(x))], 1)


class SqueezeNetPR20(nn.Module):
    """Their TF graph: valid conv1 / pools (floor), ReLU on conv10, spatial mean = logits."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=2)
        self.fires = nn.ModuleDict({
            "fire2": Fire("fire2", 64, 16, 64), "fire3": Fire("fire3", 128, 16, 64),
            "fire4": Fire("fire4", 128, 32, 128), "fire5": Fire("fire5", 256, 32, 128),
            "fire6": Fire("fire6", 256, 48, 192), "fire7": Fire("fire7", 384, 48, 192),
            "fire8": Fire("fire8", 384, 64, 256), "fire9": Fire("fire9", 512, 64, 256),
        })
        self.conv10 = nn.Conv2d(512, 2, 1)
        self.pool = nn.MaxPool2d(3, 2)

    def forward(self, x):
        f = self.fires
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(f["fire3"](f["fire2"](x)))
        x = self.pool(f["fire5"](f["fire4"](x)))
        x = f["fire9"](f["fire8"](f["fire7"](f["fire6"](x))))
        return torch.relu(self.conv10(x)).mean(dim=(2, 3))

    def load_pr20(self, path: str) -> None:
        w = safe_npy.load(path)

        def put(conv: nn.Conv2d, layer: str):  # TF kernels are (kh, kw, in, out)
            k, b = w[layer]["kernel"], w[layer]["bias"]
            conv.weight.data = torch.from_numpy(np.ascontiguousarray(k.transpose(3, 2, 0, 1))).float()
            conv.bias.data = torch.from_numpy(np.asarray(b)).float()

        put(self.conv1, "conv1")
        put(self.conv10, "conv10")
        for name, fire in self.fires.items():
            put(fire.squeeze, f"{name}_squeeze1x1")
            put(fire.e1, f"{name}_expand1x1")
            put(fire.e3, f"{name}_expand3x3")


# ---------------------------------------------------------------- strips


def load_strips(doc_dir: Path) -> list[dict]:
    """Strips of one document in true order: gray image (masked-out = white) and mask."""
    out = []
    for p in sorted((doc_dir / "strips").iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            continue
        gray = np.asarray(Image.open(p).convert("L"), dtype=np.float32)
        mp = doc_dir / "masks" / f"{p.stem}.npy"
        mask = np.load(mp) > 0 if mp.exists() else np.ones_like(gray, dtype=bool)
        gray = np.where(mask, gray, 255.0)
        out.append({"name": f"{doc_dir.name}/{p.name}", "gray": gray, "mask": mask})
    return out


def is_blank(gray: np.ndarray) -> bool:
    # cv2.GaussianBlur((5,5), 0) uses sigma = 0.3 * ((5 - 1) * 0.5 - 1) + 0.8 = 1.1
    return not (gaussian_filter(gray, 1.1, truncate=2 / 1.1) < 127).any()


def seam_features(strip: dict) -> tuple[np.ndarray, np.ndarray]:
    """(left, right) border images, ROWS x 16 each, binary (1 = paper), as in proposed.py."""
    gray, mask = strip["gray"], strip["mask"]
    binary = (gray > threshold_sauvola(gray, r=127.5)).astype(np.float32)  # their uint8 images: r = half the dtype range
    h, w = gray.shape
    has = mask.any(axis=1)
    first = np.where(has, mask.argmax(axis=1), 0)
    last = np.where(has, w - 1 - mask[:, ::-1].argmax(axis=1), w - 1)
    off = (h - ROWS) // 2
    left = np.ones((ROWS, HALF), np.float32)
    right = np.ones((ROWS, HALF), np.float32)
    for y in range(ROWS):
        yy = y + off
        if not 0 <= yy < h:
            continue
        x = first[yy]
        n = min(HALF, w - x)
        left[y, :n] = binary[yy, x : x + n]
        x = last[yy]
        n = min(HALF, x + 1)
        right[y, :n] = binary[yy, x - n + 1 : x + 1]
    return left, right


@torch.no_grad()
def compatibilities(model: nn.Module, feats: list[tuple[np.ndarray, np.ndarray]], dev, batch: int = 256) -> np.ndarray:
    """C[i, j] = max over vertical shifts of P(j directly right of i)."""
    n = len(feats)
    Lf = torch.from_numpy(np.stack([f[0] for f in feats])).to(dev)  # (n, ROWS, 16)
    Rf = torch.from_numpy(np.stack([f[1] for f in feats])).to(dev)
    # all shifted versions of every left feature: (n, 21, ROWS, 16), empty rows white
    shifted = torch.ones(n, 2 * VSHIFT + 1, ROWS, HALF, device=dev)
    for k, s in enumerate(range(-VSHIFT, VSHIFT + 1)):
        if s < 0:
            shifted[:, k, : s] = Lf[:, -s:]  # slide up
        elif s > 0:
            shifted[:, k, s:] = Lf[:, :-s]  # slide down
        else:
            shifted[:, k] = Lf
    C = np.zeros((n, n), np.float32)
    per = max(1, batch // (2 * VSHIFT + 1))
    for i in range(n):
        for j0 in range(0, n, per):
            js = torch.arange(j0, min(n, j0 + per), device=dev)
            right = Rf[i].expand(len(js), 2 * VSHIFT + 1, ROWS, HALF)
            x = torch.cat([right, shifted[js]], dim=3).reshape(-1, 1, ROWS, 2 * HALF).expand(-1, 3, -1, -1)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                p = torch.softmax(model(x).float(), 1)[:, 1]
            C[i, js.cpu().numpy()] = p.view(len(js), -1).max(1).values.cpu().numpy()
    np.fill_diagonal(C, 0)
    return C


def reconstruct(model, strips: list[dict], dev) -> tuple[list[int], list[int]]:
    """Returns (path over non-blank strips, blank strip indices)."""
    content = [k for k, s in enumerate(strips) if not is_blank(s["gray"])]
    blanks = [k for k in range(len(strips)) if k not in content]
    feats = [seam_features(strips[k]) for k in content]
    C = compatibilities(model, feats, dev)
    cost = C.max() - C
    np.fill_diagonal(cost, np.inf)
    path = solve.tsp_path(cost, list(range(len(content))))
    return [content[k] for k in path], blanks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True, help="dataset folder, e.g. datasets/D2")
    ap.add_argument("--mode", choices=["single", "mixed"], default="single")
    ap.add_argument("--docs", help='document slice, e.g. "20:"')
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SqueezeNetPR20()
    model.load_pr20(a.weights)
    model = model.to(dev).eval()

    docs = list_docs(a.data)
    if a.docs:
        lo, _, hi = a.docs.partition(":")
        docs = docs[int(lo) if lo else None : int(hi) if hi else None]
    groups = [[d] for d in docs] if a.mode == "single" else [docs]
    rng = np.random.default_rng(a.seed)
    results = []
    for group in groups:
        strips, truth = [], {}
        for d, doc_dir in enumerate(group):
            for i, s in enumerate(load_strips(doc_dir)):
                truth[s["name"]] = {"doc": d, "index": i, "flipped": False}
                strips.append(s)
        order = rng.permutation(len(strips))  # shuffle, as their pipeline does
        strips = [strips[k] for k in order]
        t0 = time.perf_counter()
        path, blanks = reconstruct(model, strips, dev)
        elapsed = time.perf_counter() - t0
        # One chain (their output is a single path); blanks appended, flagged, ignored by accuracy.
        chain = [{"name": strips[k]["name"], "flipped": False} for k in path]
        chain += [{"name": strips[k]["name"], "flipped": False, "blank": True} for k in blanks]
        m = evaluate([chain], truth) | {"seconds": round(elapsed, 1),
                                        "group": group[0].name if a.mode == "single" else "mixed"}
        results.append(m)
        print(json.dumps(m), flush=True)

    keys = ["accuracy", "neighbor_accuracy"]
    summary = {k: round(float(np.mean([r[k] for r in results])), 4) for k in keys}
    summary |= {"weights": a.weights, "data": a.data, "mode": a.mode, "docs": a.docs or ":", "groups": len(results)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"summary": summary, "groups": results}, indent=1))
    print("SUMMARY", json.dumps(summary))


if __name__ == "__main__":
    main()
