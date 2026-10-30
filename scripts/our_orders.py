"""Our model's reconstruction order for each document (single-document mode), in the
same JSON-lines format as baselines/ranca2013/run_ranca.py, for score_orders.py.

  python scripts/our_orders.py <checkpoint> <out.jsonl> <doc_dir> [doc_dir ...]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from unshredder import config
from unshredder.costs import compute
from unshredder.datasets import load_doc
from unshredder.reconstruct import reconstruct

ckpt, out, docs = sys.argv[1], sys.argv[2], sys.argv[3:]
cfg = config.load("configs/seamnet.yaml", [f"reconstruct.model_path={ckpt}", "reconstruct.solver=tsp"])
rng = np.random.default_rng(0)
with open(out, "w") as fh:
    for doc in docs:
        strips = [img for _, img in load_doc(Path(doc))]
        perm = rng.permutation(len(strips))
        t0 = time.perf_counter()
        shuffled = [strips[k] for k in perm]
        chains, _, _ = reconstruct(shuffled, cfg.reconstruct, num_docs=1)
        order = [int(perm[s]) for ch in chains for s, _ in ch]
        c = compute(shuffled, cfg.reconstruct).cost  # lower = better, shuffled-index space
        n = len(strips)
        cost = [[None] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j and np.isfinite(c[i, j]):
                    cost[int(perm[i])][int(perm[j])] = float(c[i, j])
        fh.write(json.dumps({"doc": doc, "order": order, "n": n, "cost": cost,
                             "seconds": round(time.perf_counter() - t0, 1)}) + "\n")
        print(f"{Path(doc).name}: {len(strips)} strips", flush=True)
