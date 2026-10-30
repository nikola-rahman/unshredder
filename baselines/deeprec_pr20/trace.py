"""Trace mean/abs activations per layer of the ported net, for 0/1 and 0/255 inputs."""

import sys

import numpy as np
import torch

import safe_npy
from pr20 import SqueezeNetPR20

w = safe_npy.load(sys.argv[1])
for k in ["conv1", "fire2_squeeze1x1", "fire9_expand3x3", "conv10"]:
    v = w[k]
    print(k, {p: (a.shape, float(np.abs(a).mean()), float(a.mean())) for p, a in v.items()})
m = SqueezeNetPR20()
m.load_pr20(sys.argv[1])
m.eval()
acts = {}
for name, mod in [("conv1", m.conv1)] + [(n, f) for n, f in m.fires.items()] + [("conv10", m.conv10)]:
    mod.register_forward_hook(lambda _m, _i, o, name=name: acts.__setitem__(name, o))
rng = np.random.default_rng(0)
x01 = torch.from_numpy((rng.random((4, 32, 32)) > 0.2).astype(np.float32))[:, None].expand(-1, 3, -1, -1)
for scale in (1.0, 255.0):
    acts.clear()
    with torch.no_grad():
        out = m(x01 * scale)
    print(f"input x{scale}: logits {out.numpy().round(3).tolist()}")
    for k, v in acts.items():
        print(f"   {k:7s} shape={tuple(v.shape)} mean={v.mean():.4f} frac>0={(v > 0).float().mean():.3f}")
