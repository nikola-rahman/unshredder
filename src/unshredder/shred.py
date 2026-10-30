"""Cut documents into vertical strips, degrade them, and shuffle them into one pile."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from .config import ShredConfig


@dataclass
class Strip:
    name: str
    image: np.ndarray  # float32 HxW in [0, 1], as scanned (possibly flipped)
    doc: int
    index: int  # position within its document, left to right
    flipped: bool


def _cut_positions(width: int, cfg: ShredConfig, rng: np.random.Generator) -> list[int]:
    if cfg.strip_width > 0:
        n = max(1, round(width / cfg.strip_width))
    else:
        n = cfg.num_strips
    cuts = np.linspace(0, width, n + 1)
    if cfg.width_jitter and n > 1:
        cuts[1:-1] += rng.uniform(-cfg.width_jitter, cfg.width_jitter, n - 1)
    cuts = np.round(cuts).astype(int)
    min_w = 2 + 2 * cfg.kerf
    for i in range(1, len(cuts)):  # keep cuts strictly increasing with usable widths
        cuts[i] = max(cuts[i], cuts[i - 1] + min_w)
    cuts[-1] = width
    return [int(c) for c in cuts]


def shred(docs: list[Image.Image], cfg: ShredConfig, rng: np.random.Generator) -> list[Strip]:
    strips: list[Strip] = []
    for d, doc in enumerate(docs):
        arr = np.asarray(doc.convert("L"), dtype=np.float32) / 255.0
        J = cfg.vertical_jitter
        if J:  # pad with paper so each strip can be cropped at its own vertical offset
            arr = np.pad(arr, ((J, J), (0, 0)), constant_values=1.0)
        cuts = _cut_positions(arr.shape[1], cfg, rng)
        for i, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
            lo = a + (cfg.kerf if i > 0 else 0)
            hi = b - (cfg.kerf if b < arr.shape[1] else 0)
            dy = int(rng.integers(-J, J + 1)) if J else 0
            s = arr[J + dy : arr.shape[0] - J + dy, lo:hi].copy()
            if cfg.blur > 0:
                s = gaussian_filter(s, cfg.blur)
            if cfg.brightness_jitter > 0:
                gain = 1.0 + rng.uniform(-cfg.brightness_jitter, cfg.brightness_jitter)
                s = s * gain
            if cfg.edge_damage > 0:
                k = min(cfg.edge_damage, s.shape[1] // 2)
                for cols in (slice(0, k), slice(-k, None)):
                    s[:, cols] = np.where(rng.random(s[:, cols].shape) < 0.5, s[:, cols], 1.0)
            if cfg.noise > 0:
                s = s + rng.normal(0, cfg.noise, s.shape).astype(np.float32)
            flipped = bool(rng.random() < cfg.flip_prob)
            if flipped:
                s = s[::-1, ::-1]
            strips.append(Strip("", np.clip(s, 0, 1).astype(np.float32), d, i, flipped))

    order = rng.permutation(len(strips))
    strips = [strips[k] for k in order]
    for k, s in enumerate(strips):
        s.name = f"strip_{k:04d}.png"
    return strips


def save_strips(strips: list[Strip], strips_dir: Path, truth_path: Path) -> None:
    """Strips go in their own folder; ground truth is written separately so the
    reconstruction step only ever sees the images."""
    strips_dir.mkdir(parents=True, exist_ok=True)
    for s in strips:
        Image.fromarray((s.image * 255).round().astype(np.uint8)).save(strips_dir / s.name)
    truth = {s.name: {"doc": s.doc, "index": s.index, "flipped": s.flipped} for s in strips}
    truth_path.write_text(json.dumps(truth, indent=1))


def load_strips(strips_dir: Path) -> tuple[list[str], list[np.ndarray]]:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    paths = sorted(p for p in Path(strips_dir).iterdir() if p.suffix.lower() in exts)
    if not paths:
        raise FileNotFoundError(f"no strip images in {strips_dir}")
    images = [np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0 for p in paths]
    return [p.name for p in paths], images


def pile_preview(images: list[np.ndarray], gap: int = 3) -> Image.Image:
    """All strips side by side in pile order, for eyeballing the input."""
    h = max(im.shape[0] for im in images)
    w = sum(im.shape[1] for im in images) + gap * (len(images) - 1)
    out = np.full((h, w), 0.6, dtype=np.float32)
    x = 0
    for im in images:
        out[: im.shape[0], x : x + im.shape[1]] = im
        x += im.shape[1] + gap
    return Image.fromarray((out * 255).astype(np.uint8))
