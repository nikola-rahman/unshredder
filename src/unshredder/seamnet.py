"""SeamNet: learned seam compatibility as a contrastive bi-encoder over edge bands.

Each strip edge (a `band`-px-wide band at the cut, the left band mirrored so the cut is
always on the right) is encoded by a fine-tuned ConvNeXt into one embedding per row
(stride 4). The score for "b directly right of a" sums row-wise agreement down the whole
seam, maximised over small vertical shifts:

    S(a, b) = max_s  sum_y  <r_a(y), l_b(y + s)>

so evidence accumulates over every cut character while blank rows contribute ~0. All
pairs cost a few matrix products, so thousands of strips score in seconds.

Training is self-supervised (InfoNCE): pages are shredded and degraded on the GPU
(gpu_shred.py) and each edge must pick its true neighbour among all edges on all GPUs --
same-page strips are the hard negatives -- or a learned per-row "no neighbour" logit for
page borders. That logit doubles as the document-splitting threshold at reconstruction.
"""

from __future__ import annotations

import hashlib
import math
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.checkpoint import checkpoint

from .config import Config
from .costs import normalize_strip
from .datasets import list_docs, load_doc
from .generate import synth_document
from .tracking import Tracker

STRIDE = 4
COMPILE_CHUNK = 64  # bands per compiled-trunk call during training
_MEAN, _STD = 0.449, 0.226  # ImageNet statistics, grayscale


def device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------- model


class SeamNet(nn.Module):
    def __init__(self, band: int = 48, dim: int = 128, pretrained: bool = True, row_context: str = "conv"):
        super().__init__()
        from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

        cn = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        self.band, self.dim, self.row_context = band, dim, row_context
        self.stem, self.stage1 = cn.features[0], cn.features[1]
        # Stage-2 downsampling halves only the width: rows stay at stride 4 because seam
        # detail is vertical, while band width is collapsed to one vector per row anyway.
        # The (1,2) kernel starts from the pretrained (2,2) one summed over its rows.
        norm, conv = cn.features[2]
        down = nn.Conv2d(96, 192, kernel_size=(1, 2), stride=(1, 2))
        with torch.no_grad():
            down.weight.copy_(conv.weight.sum(dim=2, keepdim=True))
            down.bias.copy_(conv.bias)
        self.down = nn.Sequential(norm, down)
        self.stage2 = cn.features[3]
        w = band // (2 * STRIDE)
        self.collapse = nn.Conv2d(192, 256, kernel_size=(1, w))  # band width -> per row
        if row_context == "transformer":
            self.rows = RowTransformer(256)
        else:
            self.rows = nn.Sequential(  # vertical context along the seam
                nn.GELU(),
                nn.Conv1d(256, 256, 15, padding=7, groups=256),
                nn.Conv1d(256, 256, 1),
                nn.GELU(),
                nn.Conv1d(256, 256, 15, padding=7, groups=256),
                nn.Conv1d(256, 256, 1),
                nn.GELU(),
            )
        self.head_r = nn.Conv1d(256, dim, 1)
        self.head_l = nn.Conv1d(256, dim, 1)
        self.log_scale = nn.Parameter(torch.tensor(math.log(1 / math.sqrt(dim))))
        self.null = nn.Parameter(torch.tensor(0.0))  # per-row "no neighbour" logit
        self.compiled_trunk = None  # set by train(); eval keeps eager for variable heights
        self.checkpoint = False  # set by train(): recompute trunk activations in backward

    def encode(self, bands: torch.Tensor, side: str) -> torch.Tensor:
        """bands (N, 1, H, band), cut on the right, [0,1] -> (N, dim, H/4)."""
        head = self.head_r if side == "r" else self.head_l
        if not (self.training and self.compiled_trunk):
            return head(self.trunk(bands))
        # The compiled trunk only ever sees one shape: fixed chunks, the last one padded
        # with blank paper. Batches vary in strip count, and every new shape would
        # otherwise trigger a recompile or slow dynamic-shape kernels.
        n, chunk = bands.shape[0], COMPILE_CHUNK
        pad = (-n) % chunk
        if pad:
            bands = torch.cat([bands, bands.new_ones(pad, *bands.shape[1:])])
        if self.checkpoint:
            out = torch.cat([checkpoint(self.compiled_trunk, c, use_reentrant=False) for c in bands.split(chunk)])
        else:
            out = torch.cat([self.compiled_trunk(c) for c in bands.split(chunk)])
        return head(out[:n])

    def trunk(self, bands: torch.Tensor) -> torch.Tensor:
        x = ((bands - _MEAN) / _STD).expand(-1, 3, -1, -1).contiguous(memory_format=torch.channels_last)
        f = self.stage2(self.down(self.stage1(self.stem(x))))
        return self.rows(self.collapse(f).squeeze(-1))


class RowTransformer(nn.Module):
    """Pre-norm transformer over the rows of a seam: lets evidence from distant lines
    inform each row. Local windows keep it linear in seam height."""

    def __init__(self, width: int, depth: int = 2, heads: int = 8, window: int = 64):
        super().__init__()
        self.window = window
        self.pos = nn.Conv1d(width, width, 9, padding=4, groups=width)  # conv positional encoding
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)

    def forward(self, f: torch.Tensor) -> torch.Tensor:  # (N, C, H) -> (N, C, H)
        n, c, h = f.shape
        x = f + self.pos(f)
        w = self.window
        pad = (-h) % w
        x = F.pad(x, (0, pad)).transpose(1, 2)  # (N, H', C)
        # Attention within windows of `w` rows (linear in seam height); the conv trunk's
        # receptive field already spans window borders.
        x = x.reshape(n * (h + pad) // w, w, c)
        x = self.blocks(x).reshape(n, h + pad, c).transpose(1, 2)
        return x[:, :, :h]


def pair_scores(R: torch.Tensor, L: torch.Tensor, max_shift: int, scale: torch.Tensor) -> torch.Tensor:
    """R (n, d, h), L (m, d, h) -> (n, m): max over row shifts of the summed dot product."""
    n, d, h = R.shape
    best = None
    Rf = R.reshape(n, d * h)
    for s in range(-max_shift, max_shift + 1):
        Ls = torch.roll(L, s, dims=2)
        if s > 0:
            Ls[:, :, :s] = 0
        elif s < 0:
            Ls[:, :, s:] = 0
        sc = Rf @ Ls.reshape(L.shape[0], d * h).T
        best = sc if best is None else torch.maximum(best, sc)
    return best * scale


def edge_bands(strip: np.ndarray, band: int) -> tuple[np.ndarray, np.ndarray]:
    """Right-edge band and mirrored left-edge band (cut on the right), padded with paper."""
    if strip.shape[1] < band:
        pad = np.ones((strip.shape[0], band - strip.shape[1]), dtype=np.float32)
        r = np.hstack([pad, strip])
        l_ = np.hstack([pad, strip[:, ::-1]])
    else:
        r = strip[:, -band:]
        l_ = strip[:, :band][:, ::-1]
    return np.ascontiguousarray(r), np.ascontiguousarray(l_)


# ---------------------------------------------------------------- training data


def _draw_content(window: np.ndarray, rng) -> np.ndarray:
    """Cheap page content the renderers don't produce: stamps, signatures, highlighter."""
    from PIL import ImageDraw

    img = Image.fromarray(window)
    draw = ImageDraw.Draw(img)
    h, w = window.shape
    if rng.random() < 0.15:  # rubber stamp: outlined ellipse/box in a mid tone
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        rx, ry = rng.integers(80, 260), rng.integers(50, 160)
        tone, width = int(rng.integers(60, 170)), int(rng.integers(3, 8))
        box = [cx - rx, cy - ry, cx + rx, cy + ry]
        (draw.ellipse if rng.random() < 0.5 else draw.rectangle)(box, outline=tone, width=width)
    if rng.random() < 0.2:  # signature / handwriting: random smooth pen strokes
        x, y = float(rng.integers(0, w)), float(rng.integers(0, h))
        pts = []
        for _ in range(int(rng.integers(20, 60))):
            x += rng.normal(12, 10)
            y += rng.normal(0, 9)
            pts.append((x, y))
        draw.line(pts, fill=int(rng.integers(0, 90)), width=int(rng.integers(2, 6)), joint="curve")
    if rng.random() < 0.15:  # highlighter over a line of text
        y = int(rng.integers(0, h))
        band = np.asarray(img, dtype=np.float32)
        band[y : y + int(rng.integers(30, 70)), int(rng.integers(0, w // 2)) : int(rng.integers(w // 2, w))] *= rng.uniform(0.8, 0.93)
        return band.astype(np.uint8)
    return np.asarray(img)


class ShardPages:
    """Pages rendered on the fly from PDFs in local tar shards (one worker's view).

    Each worker owns the shards with index % n_workers == worker_id, reads two at a time in random
    order, keeps a small buffer of PDFs to mix them, and renders a random inked page at 300 dpi.
    Shards are picked up as sync_shards.py finishes them (it renames *.part to *.tar)."""

    DPI = 300
    MAX_PIXELS = 40_000_000  # skip anything larger than ~A3 at 300 dpi
    MIN_INK = 0.005

    def __init__(self, shard_dir: str, worker: int, n_workers: int, rng, buffer: int = 64):
        self.dir, self.worker, self.n, self.rng, self.size = Path(shard_dir), worker, n_workers, rng, buffer
        self.done: set[str] = set()
        self.readers: list = []
        self.buffer: list[bytes] = []

    def _mine(self) -> list[Path]:
        return [p for p in sorted(self.dir.glob("*.tar")) if int(hashlib.md5(p.name.encode()).hexdigest(), 16) % self.n == self.worker]

    def _open_next(self) -> bool:
        todo = [p for p in self._mine() if p.name not in self.done]
        if not todo:
            if not self.done:
                return False
            self.done.clear()  # new epoch over this worker's shards
            todo = self._mine()
        path = todo[int(self.rng.integers(len(todo)))]
        self.done.add(path.name)
        tar = tarfile.open(path)  # noqa: SIM115 - stays open while its members are consumed; closed in _next_pdf
        self.readers.append((tar, (m for m in tar if m.isfile() and m.name.endswith(".pdf"))))
        return True

    def _next_pdf(self) -> bytes | None:
        while len(self.readers) < 2 and self._open_next():
            pass
        while self.readers:
            k = int(self.rng.integers(len(self.readers)))
            tar, members = self.readers[k]
            m = next(members, None)
            if m is None:
                tar.close()
                self.readers.pop(k)
                self._open_next()
                continue
            return tar.extractfile(m).read()
        return None

    def page(self) -> np.ndarray | None:
        """A uint8 grayscale page, or None while no shard is available yet."""
        import pymupdf

        pymupdf.TOOLS.mupdf_display_errors(False)
        zoom = pymupdf.Matrix(self.DPI / 72, self.DPI / 72)
        for _ in range(20):  # skip broken, encrypted, blank or oversized PDFs
            while len(self.buffer) < self.size:
                pdf = self._next_pdf()
                if pdf is None:
                    break
                self.buffer.append(pdf)
            if not self.buffer:
                return None
            data = self.buffer.pop(int(self.rng.integers(len(self.buffer))))
            try:
                with pymupdf.open(stream=data, filetype="pdf") as doc:
                    if doc.needs_pass or doc.page_count == 0:
                        continue
                    pg = doc[int(self.rng.integers(min(doc.page_count, 50)))]
                    if pg.rect.width * pg.rect.height * (self.DPI / 72) ** 2 > self.MAX_PIXELS:
                        continue
                    pix = pg.get_pixmap(matrix=zoom, colorspace=pymupdf.csGRAY, alpha=False)
                    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
                    if (img < 128).mean() >= self.MIN_INK:
                        return img
            except Exception:  # noqa: BLE001, S112 - hostile or malformed crawl PDFs are skipped
                continue
        return None


class PageSource(torch.utils.data.IterableDataset):
    """Endless stream of training material for the GPU shredder:
    ("page", uint8 window, n_strips) from synthetic or real intact pages, or
    ("strips", R, L) edge bands from real shredded documents (already in true order)."""

    def __init__(self, cfg: Config, seed: int, rank: int = 0, world: int = 1):
        self.cfg, self.seed, self.rank, self.world = cfg, seed, rank, world
        sc = cfg.seamnet
        self.page_paths = []
        for d in sc.page_dirs:
            self.page_paths += sorted(p for p in Path(d).rglob("*") if p.suffix.lower() in {".tif", ".tiff", ".png", ".jpg"})
        self.shred_docs = [doc for d in sc.shred_dirs for doc in list_docs(d)]
        self.has_shards = bool(sc.shard_dir) and Path(sc.shard_dir).is_dir()
        w = np.array([sc.synth_weight, sc.page_weight if self.page_paths else 0,
                      sc.shred_weight if self.shred_docs else 0, sc.shard_weight if self.has_shards else 0], dtype=float)
        self.weights = w / w.sum()
        # Window tall enough for the GPU shredder's scale/stretch/shift range.
        self.window_rows = int(sc.crop_rows * 1.4) + 2 * sc.max_shift + 16

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed + (info.id if info else 0) * 7919)
        self.shards = None
        if self.has_shards:  # disjoint shards per data-loader worker across all ranks
            n_workers = info.num_workers if info else 1
            self.shards = ShardPages(self.cfg.seamnet.shard_dir, self.rank * n_workers + (info.id if info else 0),
                                     self.world * n_workers, rng)
        while True:
            yield self.sample(rng)

    def _window(self, ink_rows: np.ndarray, rows: int, rng) -> int:
        """Window start: best of a few random candidates by amount of ink, so steps are
        spent on text rather than on blank margins."""
        hi = max(1, len(ink_rows) - rows)
        cands = rng.integers(0, hi, 4)
        csum = np.concatenate([[0.0], np.cumsum(ink_rows)])
        ink = [csum[min(len(ink_rows), c + rows)] - csum[c] for c in cands]
        return int(cands[int(np.argmax(ink))])

    def sample(self, rng):
        sc = self.cfg.seamnet
        kind = rng.choice(4, p=self.weights)
        if kind == 2:  # real shredded document
            strips = [img for _, img in load_doc(self.shred_docs[rng.integers(len(self.shred_docs))])]
            h = max(max(s.shape[0] for s in strips), sc.crop_rows)
            strips = [np.pad(s, ((0, h - s.shape[0]), (0, 0)), constant_values=1.0) for s in strips]
            y0 = self._window(np.mean([1 - s.mean(axis=1) for s in strips], axis=0), sc.crop_rows, rng)
            bands = [edge_bands(normalize_strip(s[y0 : y0 + sc.crop_rows]), sc.band) for s in strips]
            return ("strips", np.stack([b[0] for b in bands]), np.stack([b[1] for b in bands]))
        if kind == 3:
            page = self.shards.page()
            if page is None:  # shards not synced yet: fall back to the other sources
                return self.sample_without_shards(rng)
        elif kind == 1:
            page = np.asarray(Image.open(self.page_paths[rng.integers(len(self.page_paths))]).convert("L"))
        else:
            page = np.asarray(synth_document(rng, self.cfg.generate))
        rows, width = self.window_rows, sc.page_width
        if page.shape[0] < rows:
            page = np.pad(page, ((0, rows - page.shape[0]), (0, 0)), constant_values=255)
        y0 = self._window(1 - page.mean(axis=1) / 255, rows, rng)
        window = page[y0 : y0 + rows]
        if window.shape[1] >= width:  # fixed width so pages batch on the GPU
            x0 = int(rng.integers(0, window.shape[1] - width + 1))
            window = window[:, x0 : x0 + width]
        else:
            window = np.pad(window, ((0, 0), (0, width - window.shape[1])), constant_values=255)
        if sc.augment > 0:
            window = _draw_content(np.ascontiguousarray(window), rng)
        n = int(rng.integers(sc.strips_per_page[0], sc.strips_per_page[1] + 1))
        return ("page", np.ascontiguousarray(window), n)


    def sample_without_shards(self, rng):
        w = self.weights.copy()
        w[3] = 0
        if w.sum() == 0:
            time.sleep(1)  # only shards configured and none ready yet
            return self.sample(rng)
        saved, self.weights = self.weights, w / w.sum()
        try:
            return self.sample(rng)
        finally:
            self.weights = saved


def collate(items):
    pages = [it for it in items if it[0] == "page"]
    shreds = [it for it in items if it[0] == "strips"]
    out = {"pages": None, "n": [p[2] for p in pages], "R": [], "L": []}
    if pages:
        out["pages"] = torch.from_numpy(np.stack([p[1] for p in pages])).unsqueeze(1)
    for _, R, L in shreds:
        out["R"].append(torch.from_numpy(R))
        out["L"].append(torch.from_numpy(L))
    return out


def build_batch(batch: dict, sc, gen: torch.Generator, dev) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CPU batch -> GPU shredded & augmented edge bands with neighbour targets."""
    from . import gpu_shred

    Rs, Ls, ts = [], [], []
    base = 0
    if batch["pages"] is not None:
        pages = batch["pages"].to(dev, non_blocking=True).float() / 255
        R, L, t = gpu_shred.make_batch(pages, batch["n"], sc.crop_rows, sc.band, sc.max_shift, gen, sc.augment)
        Rs.append(R), Ls.append(L), ts.append(t)
        base += R.shape[0]
    for R, L in zip(batch["R"], batch["L"]):  # real shreds: already cut and scanned, lighter scan noise
        R, L = R.to(dev).unsqueeze(1), L.to(dev).unsqueeze(1)
        R, L = gpu_shred.scan_augment(R, L, gen, sc.augment * 0.5)
        k = R.shape[0]
        t = torch.arange(base + 1, base + k + 1, device=dev)
        t[-1] = -1
        Rs.append(gpu_shred.normalize_bands(R)), Ls.append(gpu_shred.normalize_bands(L)), ts.append(t)
        base += k
    return torch.cat(Rs), torch.cat(Ls), torch.cat(ts)


# ---------------------------------------------------------------- loss (single or multi GPU)


def _gather_pair(a: torch.Tensor, b: torch.Tensor, world: int):
    """Differentiable all-gather of two same-length, variable-size batches in ONE collective (the per-collective
    latency dominates on PCIe-only multi-GPU boxes). Returns the padded global a and b, this rank's offset and a
    validity mask."""
    import torch.distributed as dist
    import torch.distributed.nn.functional as dnn

    n = torch.tensor([a.shape[0]], device=a.device)
    sizes = [torch.zeros_like(n) for _ in range(world)]
    dist.all_gather(sizes, n)
    sizes = [int(v) for v in sizes]
    m = max(sizes)
    pad = (0, 0) * (a.dim() - 1) + (0, m - a.shape[0])
    parts = dnn.all_gather(torch.stack([F.pad(a, pad), F.pad(b, pad)]))
    valid = torch.cat([torch.arange(m, device=a.device) < k for k in sizes])
    return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts]), dist.get_rank() * m, valid


_BALANCE_LOG: list = []  # filled only when SEAMNET_BALANCE is set (see train)


def contrastive_loss(model: SeamNet, R: torch.Tensor, L: torch.Tensor, target: torch.Tensor, max_shift: int,
                     world: int = 1):
    """InfoNCE over all edges on all GPUs, plus a learned "no neighbour" option."""
    er_h, el_h = model.encode(R, "r"), model.encode(L, "l")  # bf16 under autocast
    er, el = er_h.float(), el_h.float()
    n, h = er.shape[0], er.shape[2]
    dev = er.device
    if world > 1:  # exchange in the encoder's own precision (bf16): same values, half the bytes on the GPU links
        if _BALANCE_LOG:  # diagnostics: per-rank edge count, own compute time, time spent waiting for the others
            torch.cuda.synchronize()
            t_ready = time.time()
        er_g, el_g, offset, valid = _gather_pair(er_h, el_h, world)
        if _BALANCE_LOG:
            torch.cuda.synchronize()
            _BALANCE_LOG.append((n, t_ready, time.time()))
        er_g, el_g = er_g.float(), el_g.float()
    else:
        er_g, el_g, offset, valid = er, el, 0, torch.ones(n, dtype=torch.bool, device=dev)
    N = er_g.shape[0]
    scale = model.log_scale.exp()
    null = (model.null * h).expand(n, 1)
    own = torch.arange(n, device=dev) + offset
    bad = ~valid[None, :] | (torch.arange(N, device=dev)[None, :] == own[:, None])

    # local right edges vs every left edge
    S_r = pair_scores(er, el_g, max_shift, scale).masked_fill(bad, float("-inf"))
    t_row = torch.where(target >= 0, target + offset, torch.full_like(target, N)).to(dev)
    loss_r = F.cross_entropy(torch.cat([S_r, null], 1), t_row)

    # local left edges vs every right edge
    S_l = pair_scores(er_g, el, max_shift, scale).T.masked_fill(bad, float("-inf"))
    t_col = torch.full((n,), N, dtype=torch.long, device=dev)
    has = target >= 0
    t_col[target[has]] = torch.nonzero(has).squeeze(1) + offset
    loss_l = F.cross_entropy(torch.cat([S_l, null], 1), t_col)

    with torch.no_grad():
        acc = (torch.cat([S_r, null], 1).argmax(1) == t_row).float().mean()
    return (loss_r + loss_l) / 2, acc  # a tensor: reading it (.item()) would make the CPU wait for the GPU


# ---------------------------------------------------------------- training loop


def _setup_distributed():
    import os

    import torch.distributed as dist

    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world == 1:
        return 1, 0, None
    local = int(os.environ["LOCAL_RANK"])
    if not torch.cuda.is_available():  # CPU fallback, for testing the multi-process path
        dist.init_process_group("gloo")
        return world, dist.get_rank(), torch.device("cpu")
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    return world, dist.get_rank(), torch.device("cuda", local)


def _allreduce_grads(model: nn.Module, world: int) -> None:
    import torch.distributed as dist
    from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    flat = _flatten_dense_tensors(grads)
    dist.all_reduce(flat)
    flat /= world
    for g, r in zip(grads, _unflatten_dense_tensors(flat, grads)):
        g.copy_(r)


def train(cfg: Config) -> Path:
    """Single GPU: `unshredder train-seamnet ...`.
    Multi GPU:  `torchrun --nproc_per_node 8 -m unshredder train-seamnet ...`."""
    sc = cfg.seamnet
    world, rank, ddp_dev = _setup_distributed()
    main = rank == 0
    torch.manual_seed(cfg.seed + rank)
    dev = ddp_dev or device(sc.device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = SeamNet(sc.band, sc.dim, sc.pretrained, sc.row_context)
    if sc.init_from:
        model.load_state_dict(torch.load(sc.init_from, map_location="cpu")["state_dict"])
        if main:
            print(f"initialised from {sc.init_from}")
    model = model.to(dev).to(memory_format=torch.channels_last)
    if world > 1:  # identical start on every GPU
        import torch.distributed as dist

        for prm in model.parameters():
            dist.broadcast(prm.data, 0)
    if sc.compile and dev.type == "cuda":
        model.compiled_trunk = torch.compile(model.trunk)  # fixed training shapes: fuses LN/GELU/dwconv
    model.checkpoint = sc.checkpoint
    opt = torch.optim.AdamW(model.parameters(), lr=sc.lr, weight_decay=sc.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / sc.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / sc.steps)))
    )
    loader = torch.utils.data.DataLoader(
        PageSource(cfg, cfg.seed + 10_000 + 1000 * rank, rank, world), batch_size=sc.pages_per_batch, num_workers=sc.workers,
        collate_fn=collate, prefetch_factor=4 if sc.workers else None, persistent_workers=bool(sc.workers),
        pin_memory=dev.type == "cuda",
    )
    gen = torch.Generator(device=dev)
    gen.manual_seed(cfg.seed * 1000 + rank)
    out = Path(sc.model_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    shift_rows = math.ceil(sc.max_shift / STRIDE)
    amp = dev.type == "cuda"
    tracker = Tracker(cfg.wandb_project if main else None, cfg.run_name or out.stem, cfg.to_dict(), "train")
    t0, run_loss, run_acc, best = time.time(), 0.0, 0.0, -1.0
    if main:
        print(f"training on {world}x {dev.type}; sources synth/pages/shreds = "
              f"{loader.dataset.weights.round(2).tolist()}; augment={sc.augment}", flush=True)
    import os

    prof_at = int(os.environ.get("SEAMNET_PROFILE", "0"))  # profile 10 steps from this one on rank 0, then stop
    bal_at = int(os.environ.get("SEAMNET_BALANCE", "0"))  # log per-rank edges / gather wait for 10 steps from here
    prof = None
    data_waits, t_fetch = [], time.time()
    for step, batch in enumerate(loader, start=1):
        if bal_at and bal_at <= step < bal_at + 10:
            data_waits.append(round(1000 * (time.time() - t_fetch)))
        if prof_at and main and step == prof_at:
            prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                      torch.profiler.ProfilerActivity.CUDA])
            prof.__enter__()
        if prof is not None and step == prof_at + 10:
            torch.cuda.synchronize()
            prof.__exit__(None, None, None)
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25), flush=True)
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15), flush=True)
            prof = None
        model.train()
        if bal_at and step == bal_at:
            _BALANCE_LOG.append("on")
        if bal_at and step == bal_at + 10:
            rows = [r for r in _BALANCE_LOG if r != "on"]
            waits = [t2 - t1 for _, t1, t2 in rows]
            print(f"BALANCE rank {rank}: edges {[r[0] for r in rows]}  gather wait ms "
                  f"{[round(1000 * w) for w in waits]}  data wait ms {data_waits}", flush=True)
            _BALANCE_LOG.clear()
        with torch.no_grad():
            R, L, target = build_batch(batch, sc, gen, dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
            loss, acc = contrastive_loss(model, R, L, target, shift_rows, world)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if world > 1:
            _allreduce_grads(model, world)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run_loss += loss.detach()  # running totals stay on the GPU; read once per log step
        run_acc += acc
        if step % sc.log_every == 0 and main:
            run_loss, run_acc = float(run_loss), float(run_acc)
            print(f"step {step}/{sc.steps}  loss={run_loss / sc.log_every:.4f}  batch_top1={run_acc / sc.log_every:.3f}  "
                  f"edges/step={R.shape[0] * world}  lr={sched.get_last_lr()[0]:.2e}  [{time.time() - t0:.0f}s]",
                  flush=True)
            tracker.log({"train/loss": run_loss / sc.log_every, "train/batch_top1": run_acc / sc.log_every,
                         "train/lr": sched.get_last_lr()[0], "train/elapsed_s": time.time() - t0}, step=step)
        if step % sc.log_every == 0:
            run_loss = run_acc = 0.0
        if step % sc.eval_every == 0 or step == sc.steps:
            if main:
                meta = {"band": sc.band, "dim": sc.dim, "row_context": sc.row_context, "step": step}
                torch.save({"state_dict": model.state_dict(), **meta}, out.with_suffix(".last.pt"))
                if sc.val_dirs:
                    val = validate(model, sc.val_dirs, sc.val_docs, sc.band, shift_rows, dev)
                    per_doc = val.pop("per_page", {})
                    print(f"  val: {val}", flush=True)
                    tracker.log({f"val/{k}": v for k, v in val.items()}, step=step)
                    tracker.log_distribution("val/recovery_distribution", per_doc, step=step)
                    if val["top1"] > best:
                        best = val["top1"]
                        torch.save({"state_dict": model.state_dict(), **meta, "val": val}, out)
                else:
                    torch.save({"state_dict": model.state_dict(), **meta}, out)
            if world > 1:
                import torch.distributed as dist

                dist.barrier()
        if step >= sc.steps:
            break
        t_fetch = time.time()
    if main:
        tracker.summary({"best_val_top1": best})
        tracker.finish()
        print(f"done; model at {out} (best val top1={best:.4f})")
    if world > 1:
        import torch.distributed as dist

        dist.destroy_process_group()
    return out


@torch.no_grad()
def _as_float(s: np.ndarray) -> np.ndarray:
    """Strips may be kept as 8-bit (4x less memory); the model wants [0, 1] floats."""
    return s.astype(np.float32) / 255.0 if s.dtype == np.uint8 else s


def embed(model: SeamNet, strips: list[np.ndarray], dev, batch: int = 32,
          keep_on: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Edge embeddings (n, dim, rows). keep_on: store them there in bf16 (e.g. the CPU, so GPU
    memory stays flat however large the pile); default: fp32 on `dev`."""
    model.eval()
    h = max(s.shape[0] for s in strips)
    h = int(math.ceil(h / STRIDE) * STRIDE)
    R, L = [], []
    amp = dev.type == "cuda"
    for i in range(0, len(strips), batch):
        rs, ls = zip(*(edge_bands(np.pad(_as_float(s), ((0, h - s.shape[0]), (0, 0)), constant_values=1.0), model.band)
                       for s in strips[i : i + batch]))
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
            r = model.encode(torch.from_numpy(np.stack(rs)).unsqueeze(1).to(dev), "r").float()
            l_ = model.encode(torch.from_numpy(np.stack(ls)).unsqueeze(1).to(dev), "l").float()
        if keep_on is not None:
            r, l_ = r.to(keep_on, torch.bfloat16), l_.to(keep_on, torch.bfloat16)
        R.append(r)
        L.append(l_)
    return torch.cat(R), torch.cat(L)


@torch.no_grad()
def validate(model: SeamNet, dirs: list[str], limit: int, band: int, shift_rows: int, dev) -> dict:
    """Pairwise top-1 on real shreds: per document, and with all docs pooled together."""
    strips, owner, index = [], [], []
    for d in dirs:
        for k, doc in enumerate(list_docs(d)[:limit]):
            for i, (_, img) in enumerate(load_doc(doc)):
                strips.append(normalize_strip(img))
                owner.append(f"{Path(d.split('[')[0]).name}/{doc.name}")  # e.g. "D1/D007"
                index.append(i)
    R, L = embed(model, strips, dev)
    S = pair_scores(R, L, shift_rows, model.log_scale.exp()).cpu().numpy()
    np.fill_diagonal(S, -np.inf)
    owner, index = np.array(owner), np.array(index)
    hits_doc = hits_all = total = 0
    for a in range(len(strips)):
        b = np.flatnonzero((owner == owner[a]) & (index == index[a] + 1))
        if not len(b):
            continue
        total += 1
        same = owner == owner[a]
        hits_doc += int(np.argmax(np.where(same, S[a], -np.inf)) == b[0])
        hits_all += int(np.argmax(S[a]) == b[0])
    out = {"top1": round(hits_doc / max(1, total), 4), "top1_pooled": round(hits_all / max(1, total), 4), "pairs": total}
    # 95% binomial half-width; seams within a document are correlated, so read it as a floor.
    p_ = hits_all / max(1, total)
    out["top1_pooled_ci95"] = round(1.96 * math.sqrt(p_ * (1 - p_) / max(1, total)), 4)

    # End-user metric: assemble all validation strips as one mixed pile with the
    # confidence-ordered linker, then measure how much of each page comes out intact.
    # Blank strips (no ink) are left out, as in the published metric.
    from .costs import _softmax_with_null
    from .evaluate import page_recovery
    from .solve import link_by_confidence

    null = float(model.null.item() * R.shape[2])
    prob = np.minimum(_softmax_with_null(S, null, 1), _softmax_with_null(S, null, 0))
    inked = [k for k, im in enumerate(strips) if (im < 0.5).mean() >= 0.001]
    chains = link_by_confidence(prob, 0.5, strips=inked)
    truth, rank = {}, {}
    for k in sorted(inked, key=lambda k: (owner[k], index[k])):  # re-index pages without blanks
        rank[owner[k]] = rank.get(owner[k], -1) + 1
        truth[str(k)] = {"doc": owner[k], "index": rank[owner[k]], "flipped": False}
    recon = [[{"name": str(k), "flipped": False} for k, _ in ch] for ch in chains]
    out |= page_recovery(recon, truth, per_page=True)

    # Reviewer effort: decisions per document to get every validation document 100% right
    # (simulated oracle reviewer, at the model's best calibration; see review.py).
    from .review import END, decisions_to_100

    local = {k: i for i, k in enumerate(inked)}
    by_doc: dict = {}
    for k in inked:
        by_doc.setdefault(owner[k], []).append(k)
    truth_right = [END] * len(inked)
    for members in by_doc.values():
        members.sort(key=lambda k: index[k])
        for x, y in zip(members, members[1:]):
            truth_right[local[x]] = local[y]
    sub = S[np.ix_(inked, inked)]
    r = decisions_to_100(sub, null, truth_right, [owner[k] for k in inked], pages=len(by_doc))
    out["decisions_per_doc_to_100"] = r["decisions_per_doc_to_100"]
    out["decisions_per_doc_ci95_lo"], out["decisions_per_doc_ci95_hi"] = r["decisions_per_doc_ci95"]
    out["review_temperature"] = r["temperature"]
    return out


# ---------------------------------------------------------------- inference

_MODELS: dict[str, SeamNet] = {}


def load_model(path: str, dev) -> SeamNet:
    key = f"{path}@{dev}"
    if key not in _MODELS:
        if not Path(path).exists():
            raise FileNotFoundError(f"no SeamNet model at {path}; run `unshredder train-seamnet` first")
        ckpt = torch.load(path, map_location="cpu")
        model = SeamNet(ckpt["band"], ckpt["dim"], pretrained=False, row_context=ckpt.get("row_context", "conv"))
        model.load_state_dict(ckpt["state_dict"])
        _MODELS[key] = model.to(dev).eval()
    return _MODELS[key]


@dataclass
class SparseScores:
    """Top-k candidates per edge in both directions, without the dense n x n matrix.

    right_idx[a] / right_val[a]: the k best strips b to go right of a (score S[a, b]), best first.
    left_idx[b] / left_val[b]:   the k best strips a to go left of b (score S[a, b]), best first.
    row_lse[a] = logsumexp_b S[a, b] and col_lse[b] = logsumexp_a S[a, b] over ALL strips
    (self excluded), so softmax confidences are exact even though only k scores are kept:
    P(b right of a) = exp(S[a, b] - logaddexp(row_lse[a], null)). Memory is O(n k)."""

    right_idx: np.ndarray
    right_val: np.ndarray
    left_idx: np.ndarray
    left_val: np.ndarray
    row_lse: np.ndarray
    col_lse: np.ndarray
    null: float

    @property
    def n(self) -> int:
        return len(self.row_lse)

    def save(self, path) -> None:
        np.savez(path, sparse=1, right_idx=self.right_idx, right_val=self.right_val, left_idx=self.left_idx,
                 left_val=self.left_val, row_lse=self.row_lse, col_lse=self.col_lse, null=self.null)

    @classmethod
    def load(cls, z) -> "SparseScores":
        return cls(z["right_idx"], z["right_val"], z["left_idx"], z["left_val"], z["row_lse"], z["col_lse"], float(z["null"]))

    @classmethod
    def from_dense(cls, S: np.ndarray, null: float, k: int) -> "SparseScores":
        S = np.array(S, np.float32)
        np.fill_diagonal(S, -np.inf)
        k = max(1, min(k, S.shape[0] - 1))

        def top(M):
            idx = np.argpartition(-M, k - 1, axis=1)[:, :k]
            order = np.argsort(-np.take_along_axis(M, idx, 1), axis=1, kind="stable")
            idx = np.take_along_axis(idx, order, 1)
            return idx, np.take_along_axis(M, idx, 1)

        def lse(M, axis):
            m = M.max(axis)
            return m + np.log(np.exp(M - np.expand_dims(m, axis)).sum(axis))

        ri, rv = top(S)
        li, lv = top(S.T)
        return cls(ri, rv, li, lv, lse(S, 1), lse(S, 0), float(null))


def _block_scores(Rb: torch.Tensor, Lb: torch.Tensor, max_shift: int, scale: float) -> torch.Tensor:
    """Like pair_scores for one block, in fp32 (TF32 matmuls) from compact (bf16) storage."""
    return pair_scores(Rb.float(), Lb.float(), max_shift, scale)


def _merge_top(vals, idx, new_vals, new_idx, k):
    v = torch.cat([vals, new_vals], 1)
    i = torch.cat([idx, new_idx], 1)
    top = v.topk(k, dim=1)
    return top.values, torch.gather(i, 1, top.indices)


def sparse_pair_scores(R: torch.Tensor, L: torch.Tensor, max_shift: int, scale: float, k: int,
                       devices: list | None = None, block: int = 1024) -> tuple:
    """Blockwise all-pairs scoring that keeps only top-k per row and per column plus exact
    log-sum-exps. R, L (n, d, h) may live on the CPU in bf16; blocks are moved to the GPU(s).
    Column blocks are spread over `devices` (one partial result per device, merged at the end)."""
    n = R.shape[0]
    k = max(1, min(k, n - 1))
    devices = devices or [R.device]
    neg = -torch.inf
    parts = []
    col_blocks = list(range(0, n, block))
    for dev_i, dev in enumerate(devices):
        rv = torch.full((n, k), neg, device=dev)
        ri = torch.zeros((n, k), dtype=torch.long, device=dev)
        rl = torch.full((n,), neg, device=dev)
        col_tops = {}
        for c0 in col_blocks[dev_i::len(devices)]:
            c1 = min(n, c0 + block)
            Lb = L[c0:c1].to(dev, non_blocking=True)
            cv = torch.full((c1 - c0, k), neg, device=dev)
            ci = torch.zeros((c1 - c0, k), dtype=torch.long, device=dev)
            cl = torch.full((c1 - c0,), neg, device=dev)
            for r0 in range(0, n, block):
                r1 = min(n, r0 + block)
                B = _block_scores(R[r0:r1].to(dev, non_blocking=True), Lb, max_shift, scale)  # (rows, cols)
                rr = torch.arange(r0, r1, device=dev)[:, None]
                cc = torch.arange(c0, c1, device=dev)[None, :]
                B = B.masked_fill(rr == cc, neg)  # a strip is never its own neighbour
                kk = min(k, B.shape[1])
                t = B.topk(kk, dim=1)
                rv[r0:r1], ri[r0:r1] = _merge_top(rv[r0:r1], ri[r0:r1], t.values, t.indices + c0, k)
                rl[r0:r1] = torch.logaddexp(rl[r0:r1], torch.logsumexp(B, 1))
                kk = min(k, B.shape[0])
                t = B.T.topk(kk, dim=1)
                cv, ci = _merge_top(cv, ci, t.values, t.indices + r0, k)
                cl = torch.logaddexp(cl, torch.logsumexp(B, 0))
            col_tops[c0] = (cv.cpu(), ci.cpu(), cl.cpu())
        parts.append((rv.cpu(), ri.cpu(), rl.cpu(), col_tops))
    # merge the per-device row results; column results are disjoint per device
    rv, ri, rl = parts[0][0], parts[0][1], parts[0][2]
    for prv, pri, prl, _ in parts[1:]:
        rv, ri = _merge_top(rv, ri, prv, pri, k)
        rl = torch.logaddexp(rl, prl)
    cols = {c0: t for part in parts for c0, t in part[3].items()}
    cv = torch.cat([cols[c0][0] for c0 in col_blocks])
    ci = torch.cat([cols[c0][1] for c0 in col_blocks])
    cl = torch.cat([cols[c0][2] for c0 in col_blocks])
    return ri.numpy(), rv.numpy(), ci.numpy(), cv.numpy(), rl.numpy(), cl.numpy()


@torch.no_grad()
def sparse_scores(nodes: list[np.ndarray], model_path: str, max_shift: int, k: int = 50,
                  dev_name: str = "auto", devices: list | None = None) -> SparseScores:
    """Like `scores`, but never builds the n x n matrix: top-k candidates per edge in both
    directions + exact normalisers. Edge embeddings are kept in bf16 on the CPU and streamed
    to the GPU(s) in blocks, so GPU memory does not grow with the pile."""
    dev = device(dev_name)
    model = load_model(model_path, dev)
    R, L = embed(model, nodes, dev, keep_on=torch.device("cpu"))
    if devices is None:
        devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())] or [dev]
    torch.backends.cuda.matmul.allow_tf32 = True
    out = sparse_pair_scores(R, L, math.ceil(max_shift / STRIDE), float(model.log_scale.exp()), k, devices)
    return SparseScores(*out, null=float(model.null.item() * R.shape[2]))


@torch.no_grad()
def scores(nodes: list[np.ndarray], model_path: str, max_shift: int, dev_name: str = "auto") -> tuple[np.ndarray, np.ndarray]:
    """Returns S (n, n) and the null score per pair (same for all pairs, scaled by height)."""
    dev = device(dev_name)
    model = load_model(model_path, dev)
    R, L = embed(model, nodes, dev)
    S = pair_scores(R, L, math.ceil(max_shift / STRIDE), model.log_scale.exp()).cpu().numpy().astype(np.float32)
    null = float(model.null.item() * R.shape[2])
    return S, np.broadcast_to(np.float32(null), S.shape)  # same for every pair: a view, no memory
