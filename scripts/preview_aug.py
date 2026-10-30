"""Save a preview of GPU-shredded training seams: each column pair is strip i's right band
next to strip i+1's left band, i.e. how the true seam looks to the model.
Usage: python scripts/preview_aug.py out.png [strength]"""

import sys
import time

import numpy as np
import torch
from PIL import Image

from unshredder import config, gpu_shred
from unshredder.seamnet import PageSource, collate, device

strength = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
cfg = config.load(None, ["seamnet.crop_rows=512", "seamnet.strips_per_page=[30, 30]"])
src = PageSource(cfg, 0)
rng = np.random.default_rng(1)
batch = collate([src.sample(rng) for _ in range(2)])
dev = device()
gen = torch.Generator(device=dev)
gen.manual_seed(0)
pages = batch["pages"].to(dev).float() / 255
t = time.perf_counter()
R, L, target = gpu_shred.make_batch(pages, batch["n"], 512, 48, 12, gen, strength)
print(f"{R.shape[0]} strips in {time.perf_counter() - t:.2f}s on {dev}")
R, L = R.cpu().numpy()[:, 0], L.cpu().numpy()[:, 0]
cols = []
for i in range(8, 20):
    j = int(target[i])
    if j < 0:
        continue
    seam = np.hstack([R[i], L[j][:, ::-1]])  # unmirror the left band
    cols += [seam, np.full((seam.shape[0], 6), 0.5, np.float32)]
Image.fromarray((np.hstack(cols) * 255).astype(np.uint8)).save(sys.argv[1])
