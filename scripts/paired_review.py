"""Paired comparison of two review-sim results (review.json with per-document decisions):
mean decisions per document to 100% for each, and a paired bootstrap 95% CI of B - A.

  python scripts/paired_review.py <A/review.json> <B/review.json>
"""

import json
import sys

import numpy as np

A, B = (json.load(open(p)) for p in sys.argv[1:3])
docs = sorted(set(A["per_doc"]) & set(B["per_doc"]))
a = np.array([A["per_doc"][d] for d in docs], float)
b = np.array([B["per_doc"][d] for d in docs], float)
d = b - a
rng = np.random.default_rng(0)
boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)])
lo, hi = np.percentile(boot, [2.5, 97.5])
for name, r in (("A", A), ("B", B)):
    s = r["summary"]
    print(f"{name}: {s['decisions_per_doc_to_100']} decisions/doc to 100% (95% CI {s['decisions_per_doc_ci95']}), "
          f"temperature {s['temperature']}, rejections {s['rejections']}, auto-joins {s['auto_links']}")
print(f"B - A: {d.mean():+.2f} decisions/doc, paired 95% CI [{lo:+.2f}, {hi:+.2f}]; "
      f"B cheaper on {(d < 0).sum()}, dearer on {(d > 0).sum()}, equal on {(d == 0).sum()} of {len(d)} documents")
