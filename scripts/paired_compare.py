"""Paired per-document comparison of two checkpoints on the validation documents (pooled
pile, confidence linker, blank strips excluded): page recovery per document for each,
how many documents improved / got worse, and a paired bootstrap 95% interval for the
mean difference.

  python scripts/paired_compare.py <checkpoint A> <checkpoint B>
"""

import subprocess
import sys

import numpy as np


def per_doc(ckpt: str) -> dict[str, float]:
    out = subprocess.run([sys.executable, "scripts/per_doc_recovery.py", ckpt, "--per-doc-json"],
                         capture_output=True, text=True, check=True).stdout
    import json

    return json.loads(out.strip().splitlines()[-1])


a_path, b_path = sys.argv[1], sys.argv[2]
A, B = per_doc(a_path), per_doc(b_path)
docs = sorted(A)
d = np.array([B[k] - A[k] for k in docs])
rng = np.random.default_rng(0)
boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)])
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"A = {a_path}: mean page recovery {np.mean([A[k] for k in docs]):.3f}")
print(f"B = {b_path}: mean page recovery {np.mean([B[k] for k in docs]):.3f}")
print(f"B - A per document: mean {d.mean():+.3f}, paired 95% CI [{lo:+.3f}, {hi:+.3f}]")
print(f"documents better with B: {(d > 0).sum()}, worse: {(d < 0).sum()}, unchanged: {(d == 0).sum()} of {len(d)}")
