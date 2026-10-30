"""Training-loss summary per window of steps from a training log: mean, spread, standard
error of the mean, mean batch top-1 and the learning rate at the window's end.

  python scripts/loss_summary.py logs/run3.pt.log [window]
"""

import re
import statistics as st
import sys

path = sys.argv[1]
window = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
rows = []
for line in open(path):
    m = re.match(r"step (\d+)/\d+\s+loss=([\d.]+)\s+batch_top1=([\d.]+).*lr=([\d.e+-]+)", line)
    if m:
        rows.append((int(m[1]), float(m[2]), float(m[3]), float(m[4])))
print(f"{path}: {len(rows)} points, last step {rows[-1][0]}")
print(f"{'steps':>13} {'loss mean':>10} {'loss sd':>8} {'se of mean':>10} {'top1 mean':>10} {'lr':>9}")
for w in range(0, rows[-1][0], window):
    win = [r for r in rows if w < r[0] <= w + window]
    if len(win) < 3:
        continue
    losses = [r[1] for r in win]
    print(f"{w:>6}-{w + window:<6} {st.mean(losses):10.4f} {st.stdev(losses):8.4f} "
          f"{st.stdev(losses) / len(losses) ** 0.5:10.4f} {st.mean(r[2] for r in win):10.3f} {win[-1][3]:9.2e}")
