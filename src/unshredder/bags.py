"""Simulated bags: many real/synthetic pages shredded together into one pile.

A bag is the unit of the real job: thousands of strips from hundreds of pages, some
strips lost, all shuffled. The output folder has the same layout as `unshredder shred`
(strips/ + ground_truth.json), with "doc" = source page, so the standard reconstruction
and evaluation run on it unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from . import gpu_shred


def make_bag(page_dirs: list[str], n_pages: int, out_dir: str, seed: int = 0, strips_per_page: tuple[int, int] = (25, 35),
             loss: float = 0.03, strength: float = 1.0, device: str = "auto") -> Path:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    paths = sorted(p for d in page_dirs for p in Path(d).rglob("*") if p.suffix.lower() in exts)
    if len(paths) < n_pages:
        raise ValueError(f"only {len(paths)} pages available in {page_dirs}")
    rng = np.random.default_rng(seed)
    chosen = [paths[i] for i in rng.choice(len(paths), n_pages, replace=False)]
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else
                       ("cpu" if device == "auto" else device))
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)

    out = Path(out_dir)
    (out / "strips").mkdir(parents=True, exist_ok=True)
    pile = []  # (image, page, index)
    for k, path in enumerate(chosen):
        page = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255
        t = torch.from_numpy(page).to(dev)[None, None]
        n = int(rng.integers(strips_per_page[0], strips_per_page[1] + 1))
        for i, strip in enumerate(gpu_shred.cut_strips_full(t, n, gen, strength)):
            if rng.random() < loss:  # lost in the bag / destroyed
                continue
            pile.append((strip.cpu().numpy(), k, i))
        if (k + 1) % 50 == 0:
            print(f"{k + 1}/{n_pages} pages shredded", flush=True)

    order = rng.permutation(len(pile))
    truth, sources = {}, {}
    for j, idx in enumerate(order):
        img, page, index = pile[idx]
        name = f"strip_{j:06d}.png"
        Image.fromarray((img * 255).round().astype(np.uint8)).save(out / "strips" / name)
        truth[name] = {"doc": int(page), "index": int(index), "flipped": False}
    sources = {int(k): str(p) for k, p in enumerate(chosen)}
    (out / "ground_truth.json").write_text(json.dumps(truth))
    (out / "bag.json").write_text(json.dumps({"pages": n_pages, "strips": len(pile), "seed": seed, "loss": loss,
                                              "strength": strength, "sources": sources}, indent=1))
    print(f"bag: {n_pages} pages -> {len(pile)} strips in {out}")
    return out
