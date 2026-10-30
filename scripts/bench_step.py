"""Time one SeamNet training step (forward + backward + optimizer) on random input,
excluding data loading. Usage: python scripts/bench_step.py [--no-compile] [--strips N]"""

import argparse
import time

import torch

from unshredder.seamnet import SeamNet, contrastive_loss

ap = argparse.ArgumentParser()
ap.add_argument("--strips", type=int, default=120)
ap.add_argument("--rows", type=int, default=1024)
ap.add_argument("--band", type=int, default=48)
ap.add_argument("--no-compile", action="store_true")
ap.add_argument("--iters", type=int, default=20)
a = ap.parse_args()

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
dev = torch.device("cuda")
model = SeamNet(a.band, 128, pretrained=False).to(dev).to(memory_format=torch.channels_last)
if not a.no_compile:
    model.compiled_trunk = torch.compile(model.trunk)
opt = torch.optim.AdamW(model.parameters(), 1e-4)
R = torch.rand(a.strips, 1, a.rows, a.band, device=dev)
L = torch.rand(a.strips, 1, a.rows, a.band, device=dev)
target = torch.arange(1, a.strips + 1, device=dev)
target[-1] = -1

def step():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, _ = contrastive_loss(model, R, L, target, 3)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

for _ in range(3):  # warmup / compile
    step()
torch.cuda.synchronize()
t = time.perf_counter()
for _ in range(a.iters):
    step()
torch.cuda.synchronize()
dt = (time.perf_counter() - t) / a.iters
print(f"compile={not a.no_compile} strips={a.strips} rows={a.rows} band={a.band}: {dt * 1000:.0f} ms/step, "
      f"{a.strips / dt:.0f} strips/s, peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
