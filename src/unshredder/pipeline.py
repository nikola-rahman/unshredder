"""End-to-end stages: make strips -> reconstruct from disk -> evaluate; plus parameter sweeps."""

from __future__ import annotations

import copy
import csv
import itertools
import json
import time
from pathlib import Path

import numpy as np

from . import config as config_mod
from .config import Config
from .datasets import list_docs, load_doc, load_pile
from .evaluate import evaluate, link_metrics, page_recovery_inked
from .generate import load_documents
from .reconstruct import reconstruct, save_reconstruction
from .shred import load_strips, pile_preview, save_strips, shred
from .tracking import Tracker


def make_strips(cfg: Config) -> Path:
    """Generate (or load) documents, shred them and write the shuffled pile to disk."""
    out = Path(cfg.output_dir)
    rng = np.random.default_rng(cfg.seed)
    docs = load_documents(cfg.generate, rng)
    (out / "originals").mkdir(parents=True, exist_ok=True)
    for k, d in enumerate(docs):
        d.save(out / "originals" / f"doc_{k:03d}.png")
    strips = shred(docs, cfg.shred, rng)
    save_strips(strips, out / "strips", out / "ground_truth.json")
    pile_preview([s.image for s in strips]).save(out / "pile_preview.png")
    config_mod.save(cfg, out / "config.yaml")
    return out


def run(cfg: Config, verbose: bool = True) -> dict:
    out = make_strips(cfg)
    truth = json.loads((out / "ground_truth.json").read_text())

    # Reconstruction only sees the strip images on disk, never the ground truth...
    names, images = load_strips(out / "strips")
    oracle = len({t["doc"] for t in truth.values()}) if cfg.reconstruct.num_docs == "oracle" else None
    t0 = time.perf_counter()
    chains, blanks, conf = reconstruct(images, cfg.reconstruct, num_docs=oracle)
    elapsed = time.perf_counter() - t0
    result = save_reconstruction(chains, blanks, names, images, out / "reconstructed", conf)

    # ...which is only used afterwards, for scoring.
    metrics = evaluate(result, truth) | {"reconstruct_seconds": round(elapsed, 3)}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    if verbose:
        print(f"[{out}] " + "  ".join(f"{k}={v}" for k, v in metrics.items()))
    return metrics


def sweep(base: Config, grid: dict[str, list], repeats: int = 1) -> list[dict]:
    """Run every combination in `grid` (dotted key -> list of values), `repeats` seeds each."""
    keys = list(grid)
    root = Path(base.output_dir)
    rows = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        for r in range(repeats):
            cfg = copy.deepcopy(base)
            for k, v in zip(keys, combo):
                config_mod.set_value(cfg, k, v)
            cfg.seed = base.seed + r
            tag = "_".join(f"{k.split('.')[-1]}={v}" for k, v in zip(keys, combo)) + f"_seed={cfg.seed}"
            cfg.output_dir = str(root / tag.replace(" ", ""))
            rows.append(dict(zip(keys, combo)) | {"seed": cfg.seed} | run(cfg, verbose=False))
            print(_fmt_row(rows[-1]))

    root.mkdir(parents=True, exist_ok=True)
    with open(root / "sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    _print_summary(rows, keys)
    print(f"\nwrote {root / 'sweep.csv'}")
    return rows


def _fmt_row(row: dict) -> str:
    return "  ".join(f"{k}={v}" for k, v in row.items())


def _print_summary(rows: list[dict], keys: list[str]) -> None:
    """Mean metrics per grid point (averaged over seeds)."""
    metrics = ["neighbor_accuracy", "perfect_doc_rate", "purity", "docs_found", "reconstruct_seconds"]
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(str(row[k]) for k in keys), []).append(row)
    header = [k.split(".")[-1] for k in keys] + metrics
    table = [
        list(g) + [f"{np.mean([r[m] for r in rs]):.3f}" for m in metrics] for g, rs in groups.items()
    ]
    widths = [max(len(str(c)) for c in col) for col in zip(header, *table)]
    print()
    for line in [header, *table]:
        print("  ".join(str(c).rjust(w) for c, w in zip(line, widths)))


def benchmark(cfg: Config, data: str, mode: str = "mixed", docs_range: str | None = None) -> dict:
    """Reconstruct a real shredded dataset. mode="single": each document on its own (the
    paper's k=1); mode="mixed": all documents shuffled into one pile. docs_range selects
    documents by position, Python-slice style: "0:20" (validation), "20:" (test)."""
    rng = np.random.default_rng(cfg.seed)
    docs = list_docs(data)
    if docs_range:
        lo, _, hi = docs_range.partition(":")
        docs = docs[int(lo) if lo else None : int(hi) if hi else None]
    groups = [[d] for d in docs] if mode == "single" else [docs]
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    per_group = []
    for group in groups:
        names, images, truth = load_pile(group, rng)
        oracle = len(group) if cfg.reconstruct.num_docs == "oracle" else None
        t0 = time.perf_counter()
        chains, blanks, conf = reconstruct(images, cfg.reconstruct, num_docs=oracle)
        elapsed = time.perf_counter() - t0
        tag = group[0].name if mode == "single" else "mixed"
        result = save_reconstruction(chains, blanks, names, images, out / tag, conf)
        m = evaluate(result, truth) | {"reconstruct_seconds": round(elapsed, 2), "group": tag}
        per_group.append(m)
        print("  ".join(f"{k}={v}" for k, v in m.items()), flush=True)

    keys = ["accuracy", "neighbor_accuracy", "perfect_doc_rate", "purity", "reconstruct_seconds"]
    summary = {k: round(float(np.mean([m[k] for m in per_group])), 4) for k in keys}
    summary |= {"dataset": data, "mode": mode, "docs_range": docs_range or ":", "documents": len(docs),
                "groups": len(per_group)}
    (out / "benchmark.json").write_text(json.dumps({"summary": summary, "groups": per_group}, indent=1))
    tracker = Tracker(cfg.wandb_project, cfg.run_name or f"bench-{Path(data).name}-{mode}", cfg.to_dict(), "benchmark",
                      group="benchmarks")
    tracker.summary(summary)
    tracker.finish()
    config_mod.save(cfg, out / "config.yaml")
    print("SUMMARY " + "  ".join(f"{k}={v}" for k, v in summary.items()))
    return summary


def eval_bag(cfg: Config, bag_dir: str) -> dict:
    """Reconstruct a simulated bag (see bags.py) and score it with pile-scale metrics."""
    bag = Path(bag_dir)
    truth = json.loads((bag / "ground_truth.json").read_text())
    names, images = load_strips(bag / "strips")
    t0 = time.perf_counter()
    chains, blanks, conf = reconstruct(images, cfg.reconstruct)
    elapsed = time.perf_counter() - t0
    out = Path(cfg.output_dir)
    result = save_reconstruction(chains, blanks, names, images, out, conf, render_images=False)
    metrics = (evaluate(result, truth) | link_metrics(result, truth) | page_recovery_inked(result, truth)
               | {"reconstruct_seconds": round(elapsed, 1), "bag": str(bag)})
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    tracker = Tracker(cfg.wandb_project, cfg.run_name or f"bag-{bag.name}", cfg.to_dict(), "bag-eval", group="bags")
    tracker.summary(metrics)
    tracker.finish()
    print("BAG " + "  ".join(f"{k}={v}" for k, v in metrics.items()), flush=True)
    return metrics


def review_sim(cfg: Config, source: str, k: int = 5, auto_threshold: float = 0.99, error_rate: float = 0.0) -> dict:
    """Simulated reviewer on a pile with known answers: the validation documents pooled
    (source="val") or a simulated bag folder. Blank strips are left out (unplaceable by
    content). Scores are cached next to the output so reviewer settings can be swept."""
    from . import seamnet
    from .costs import normalize_strip
    from .review import END, decisions_to_100

    if source == "val":
        strips, page = [], []
        for d in cfg.seamnet.val_dirs:
            for doc in list_docs(d):
                for _, img in load_doc(doc):
                    strips.append(img)
                    page.append(f"{d}/{doc.name}")
        index = [0] * len(strips)
        seen: dict[str, int] = {}
        for i, p in enumerate(page):
            index[i] = seen.get(p, 0)
            seen[p] = index[i] + 1
    else:
        truth = json.loads((Path(source) / "ground_truth.json").read_text())
        names, strips = load_strips(Path(source) / "strips")
        page = [truth[n]["doc"] for n in names]
        index = [truth[n]["index"] for n in names]

    inked = [i for i, im in enumerate(strips) if (normalize_strip(im) < 0.5).mean() >= 0.001]
    strips = [strips[i] for i in inked]
    page = [page[i] for i in inked]
    index = [index[i] for i in inked]
    by_page: dict = {}
    for i in range(len(strips)):
        by_page.setdefault(page[i], []).append(i)
    truth_right = [END] * len(strips)
    for members in by_page.values():
        members.sort(key=lambda i: index[i])
        for a, b in zip(members, members[1:]):
            truth_right[a] = b

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "scores.npz"
    if cache.exists():
        z = np.load(cache)
        S, null = z["S"], float(z["null"])
    else:
        S, nullm = seamnet.scores(strips, cfg.reconstruct.model_path, cfg.reconstruct.max_shift, cfg.reconstruct.device)
        null = float(nullm.flat[0])
        np.savez(cache, S=S, null=null)

    t0 = time.perf_counter()
    best = decisions_to_100(S, null, truth_right, page, pages=len(by_page), k=k, auto_threshold=auto_threshold,
                            error_rate=error_rate)
    per_doc = best.pop("per_doc")
    summary = best | {"strips": len(strips), "k": k, "auto_threshold": auto_threshold,
                      "error_rate": error_rate, "seconds": round(time.perf_counter() - t0, 1)}
    (out / "review.json").write_text(json.dumps({"summary": summary, "per_doc": per_doc}, indent=1))
    print("REVIEW " + "  ".join(f"{key}={v}" for key, v in summary.items()), flush=True)
    return summary
