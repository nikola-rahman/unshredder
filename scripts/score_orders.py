"""Score reconstructed strip orders (one JSON line per document: {"doc", "order"}) from any
method with the same metrics: Paixão accuracy, page recovery, seam recovery, pages
complete. Blank strips (no ink) are excluded, as in the published metric. Prints the mean
and a 95% interval across documents.

  python scripts/score_orders.py orders.jsonl [more.jsonl ...]
"""

import json
import sys
from pathlib import Path

import numpy as np

from unshredder.costs import normalize_strip
from unshredder.datasets import load_doc
from unshredder.evaluate import evaluate, page_recovery
from unshredder.solve import tsp_path


CONTAINER_DATA = "/data/datasets"  # where docker_run.sh mounts the datasets
HOST_DATA = "datasets"


def score_doc(rec: dict) -> dict:
    doc = rec["doc"]
    if doc.startswith(CONTAINER_DATA) and not Path(doc).exists():
        doc = HOST_DATA + doc[len(CONTAINER_DATA):]
    strips = [img for _, img in load_doc(Path(doc))]
    inked = [k for k, im in enumerate(strips) if (normalize_strip(im) < 0.5).mean() >= 0.001]
    rank = {k: r for r, k in enumerate(inked)}  # true position among inked strips
    if not inked:  # nothing to reconstruct on a page without ink
        return {"accuracy": 1.0, "page_recovery": 1.0, "seam_recovery": 1.0, "pages_complete": 1.0}
    order = [k for k in rec["order"] if k in rank]
    truth = {str(k): {"doc": 0, "index": rank[k], "flipped": False} for k in inked}
    chain = [{"name": str(k), "flipped": False} for k in order]
    out = {"accuracy": evaluate([chain], truth)["accuracy"]} | page_recovery([chain], truth)
    if "cost" in rec:  # components: the scoring function alone, and our solver on that cost
        C = np.array([[np.inf if v is None else v for v in row] for row in rec["cost"]], dtype=float)
        sub = C[np.ix_(inked, inked)]
        hits = [int(np.argmin(sub[r])) == r + 1 for r in range(len(inked) - 1)]
        out["top1"] = float(np.mean(hits)) if hits else 1.0
        tsp_order = [inked[k] for k in tsp_path(sub, list(range(len(inked))))]
        tsp_chain = [{"name": str(k), "flipped": False} for k in tsp_order]
        out |= {f"tsp_{k}": v for k, v in page_recovery([tsp_chain], truth).items()}
        out["tsp_accuracy"] = evaluate([tsp_chain], truth)["accuracy"]
    return out


def summarize(path: str) -> dict:
    recs = [json.loads(line) for line in open(path)]
    per = [score_doc(r) for r in recs]
    out = {"file": path, "documents": len(per)}
    keys = ["accuracy", "page_recovery", "seam_recovery", "pages_complete", "top1",
            "tsp_accuracy", "tsp_page_recovery", "tsp_seam_recovery", "tsp_pages_complete"]
    for k in keys:
        if not all(k in p for p in per):
            continue
        v = np.array([p[k] for p in per], dtype=float)
        out[k] = round(float(v.mean()), 4)
        out[k + "_ci95"] = round(float(1.96 * v.std(ddof=1) / np.sqrt(len(v))), 4) if len(v) > 1 else None
    return out


if __name__ == "__main__":
    for f in sys.argv[1:]:
        print(json.dumps(summarize(f)))
