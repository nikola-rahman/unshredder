"""Per-document recovery for a checkpoint on the fixed validation documents, assembled as
one mixed pile (confidence-ordered linker, blank strips excluded).

  python scripts/per_doc_recovery.py <checkpoint> [--wandb-run "<run name>"] [config overrides...]

With --wandb-run, the histogram and per-document table are attached to that existing run.
"""

import sys

import numpy as np
import torch

from unshredder import config
from unshredder.costs import _softmax_with_null, normalize_strip
from unshredder.datasets import list_docs, load_doc
from unshredder.seamnet import STRIDE, device, embed, load_model, pair_scores
from unshredder.solve import link_by_confidence

ckpt = sys.argv[1]
args = sys.argv[2:]
as_json = "--per-doc-json" in args
args = [x for x in args if x != "--per-doc-json"]
wandb_run = None
if "--wandb-run" in args:
    i = args.index("--wandb-run")
    wandb_run = args[i + 1]
    args = args[:i] + args[i + 2:]
cfg = config.load("configs/seamnet.yaml", args)
dev = device("auto")
model = load_model(ckpt, dev)
strips, owner, index = [], [], []
for d in cfg.seamnet.val_dirs:
    for doc in list_docs(d):
        for i, (_, img) in enumerate(load_doc(doc)):
            strips.append(normalize_strip(img))
            owner.append(f"{d.split('/')[-1].split('[')[0]}/{doc.name}")
            index.append(i)
with torch.no_grad():
    R, L = embed(model, strips, dev)
    S = pair_scores(R, L, int(np.ceil(cfg.seamnet.max_shift / STRIDE)), model.log_scale.exp()).cpu().numpy()
np.fill_diagonal(S, -np.inf)
null = float(model.null.item() * R.shape[2])
prob = np.minimum(_softmax_with_null(S, null, 1), _softmax_with_null(S, null, 0))
inked = [k for k, im in enumerate(strips) if (im < 0.5).mean() >= 0.001]
chains = link_by_confidence(prob, 0.5, strips=inked)

# per document: rank of each inked strip within its document (blank strips removed)
rank, docs = {}, {}
for k in sorted(inked, key=lambda k: (owner[k], index[k])):
    docs.setdefault(owner[k], []).append(k)
    rank[k] = len(docs[owner[k]]) - 1
best = {d: 0 for d in docs}
for ch in chains:
    run, prev = 0, None
    for k, _ in ch:
        if prev is not None and owner[k] == owner[prev] and rank[k] == rank[prev] + 1:
            run += 1
        else:
            run = 1
        best[owner[k]] = max(best[owner[k]], run)
        prev = k
frac = {d: best[d] / len(ks) for d, ks in docs.items()}
vals = np.array(sorted(frac.values(), reverse=True))
if as_json:  # machine-readable per-document fractions only (used by paired_compare.py)
    import json

    print(json.dumps(frac))
    sys.exit(0)
print(f"{len(frac)} validation documents, {len(inked)} inked strips, pile reconstructed as {len(chains)} chains")
print(f"mean page recovery {vals.mean():.3f}; median {np.median(vals):.3f}")
for t in (1.0, 0.8, 0.6, 0.4):
    print(f"  documents >= {int(t * 100)}% recovered intact: {(vals >= t).sum()}/{len(vals)}")
print("best:", ", ".join(f"{d} {v:.0%}" for d, v in sorted(frac.items(), key=lambda x: -x[1])[:5]))
print("worst:", ", ".join(f"{d} {v:.0%}" for d, v in sorted(frac.items(), key=lambda x: x[1])[:5]))

if wandb_run:
    import wandb

    api = wandb.Api()
    match = [r for r in api.runs(f"{api.default_entity}/{cfg.wandb_project}") if r.name == wandb_run]
    if not match:
        sys.exit(f"no wandb run named {wandb_run!r}")
    run = wandb.init(project=cfg.wandb_project, id=match[0].id, resume="allow",
                     settings=wandb.Settings(console="off", save_code=False, disable_git=True))
    from unshredder.tracking import recovery_chart

    run.log({"val/recovery_distribution": wandb.Image(recovery_chart(list(frac.values()), wandb_run))})
    run.summary.update({"final/page_recovery_mean": float(vals.mean())})
    run.finish()
    print(f"attached to wandb run {wandb_run!r}")
