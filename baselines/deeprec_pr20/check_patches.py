"""Sanity check of the weight port on the task it was trained on: 32x32 binary patches
(16 px from each side of a cut) from an intact page, positives vs negatives."""

import sys

import numpy as np
import torch
from PIL import Image
from skimage.filters import threshold_sauvola

from pr20 import SqueezeNetPR20

weights, page_path = sys.argv[1], sys.argv[2]
im = Image.open(page_path)
if im.mode in ("LA", "RGBA", "PA"):  # composite over white paper using the alpha channel
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    im = Image.alpha_composite(bg, im.convert("RGBA"))
gray = np.asarray(im.convert("L"), dtype=np.float32)
binary = gray > threshold_sauvola(gray)
w = binary.shape[1]
cuts = np.linspace(0, w, 31).astype(int)
strips = [binary[:, a:b] for a, b in zip(cuts[:-1], cuts[1:])]
rng = np.random.default_rng(0)


def patches(i, j, k=200):
    img = np.hstack([strips[i][:, -16:], strips[j][:, :16]])
    out = []
    for y in rng.permutation(np.arange(0, img.shape[0] - 32, 2)):
        p = img[y : y + 32]
        if (~p).mean() >= 0.2:  # their neutral filter
            out.append(p)
        if len(out) >= k:
            break
    return out


pos = [p for i in range(29) for p in patches(i, i + 1, 20)]
neg = [p for _ in range(29) for p in patches(*rng.choice(30, 2, replace=False), 20)]
x = torch.from_numpy(np.stack(pos + neg).astype(np.float32))[:, None].expand(-1, 3, -1, -1)
y = np.array([1] * len(pos) + [0] * len(neg))
m = SqueezeNetPR20()
m.load_pr20(weights)
m.eval()
with torch.no_grad():
    logits = m(x)
p = torch.softmax(logits, 1)[:, 1].numpy()
print(f"{len(pos)} pos / {len(neg)} neg; acc(class1=pos)={((p > 0.5) == y).mean():.3f}  "
      f"mean P1 pos={p[y == 1].mean():.3f} neg={p[y == 0].mean():.3f}; logits sample={logits[:3].numpy().round(3).tolist()}")

xs = x[:, 0].numpy()
print(f"input: per-patch ink frac mean={1 - xs.mean():.3f} min={1 - xs.mean((1, 2)).max():.3f} "
      f"max={1 - xs.mean((1, 2)).min():.3f}; distinct patches={len({p.tobytes() for p in xs})}/{len(xs)}")
print(f"logit std across patches: {logits.std(0).numpy().round(4).tolist()}")
feats = {}
m.fires["fire9"].register_forward_hook(lambda _m, _i, o: feats.__setitem__("f9", o))
m.conv1.register_forward_hook(lambda _m, _i, o: feats.__setitem__("c1", o))
with torch.no_grad():
    m(x)
print(f"conv1 std across patches={feats['c1'].std(0).mean():.4f}  fire9 std across patches={feats['f9'].std(0).mean():.4f} "
      f"fire9 frac>0={(feats['f9'] > 0).float().mean():.3f}")
