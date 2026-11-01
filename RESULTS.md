# Results log

Metric notes: `accuracy` = Paixão et al. reconstruction accuracy (fraction of output
neighbour links that are correct, blank strips excluded). `top1` / `top1_pooled` =
per-seam ranking on held-out real shreds (within a document / among all validation
documents). Bag metrics (recall at 99% precision, pages fully recovered, B-cubed) are
the ones that matter for the many-bags use case.

## Published reference (Paixão et al., CVPR 2020, Table 1)

| | D1 single | D2 single | D1 mixed (1,370 shreds) | D2 mixed (505 shreds) |
|---|---|---|---|---|
| Deep baseline (Paixão-b) | 96.78 | 94.78 | 97.08 | 95.24 |
| CVPR 2020 model | 93.14 | 95.39 | 94.81 | 97.22 |

## SeamNet, D1 held out (2026-10-30)

Validation = D1 docs 1-20, test = D1 docs 21-60 (evaluated once).

| Model | val single | val mixed (20 docs) | **test single** | **test mixed (40 docs)** |
|---|---|---|---|---|
| v1 (A6000, 2.5k steps) | 97.78 | 97.00 | | |
| A: augment 0 | 97.84 | 96.79 | **97.17** | **93.93** |
| B: augment 0.5 | 97.74 | 95.50 | | |
| C: augment 1.0 | 98.13 | 95.29 | | |

Ablations (single GPU, 3k steps from v1, best val top1_pooled on D1 docs 1-20):
augment 0: 86.2 | 0.5: 85.6 | 1.0: 83.8 | 1.0 + lr 1e-4: 84.6 | 0.5 + lr 1e-4: 85.6 |
0.5 + 3x real shreds: 85.2 | from scratch: 83.2 | row transformer from scratch: 84.0.

Findings:
- Strong augmentation lowers D1 accuracy (D1 is the cleanest set; needs checking on D2/D3).
- Learning rate and real-shred weight barely matter; per-seam top1 plateaus ~86-90%.
- The global solver lifts per-seam ~86% to ~97% end-to-end.
- Accuracy drops as the pile grows (96.8% for 20 docs -> 93.9% for 40 docs).

## Held-out folds (in progress, 2026-10-30)

Augmentation is necessary: without it, training from scratch fails to generalise to real
shreds (D1 held out: val top1_pooled ~0.48 vs ~0.8 for the row-transformer run), and on
D3 (messy real shreds) top1_pooled fell 0.78 -> 0.60 while augment 0.5 rose to ~0.84.
The earlier "augment 0 is best" result was confounded by initialising from v1, which had
already been trained with augmentation.

## PDFA real-document corpus (2026-10-30)

59 of 60 sampled shards (every 30th of pixparse/pdfa-eng-wds) -> 116,484 pages at 300 dpi
from 70,800 PDFs (2 pages per document). Leakage check (16x16 dHash vs D1/D2/D3 originals)
quarantined 658 pages; all 200 sampled matches were against individual D1 strip images
(false positives), none against an original page. Excluded anyway; checker now uses full
originals only.

## Run 1 vs run 2 (fixed splits; validation = 45 real documents pooled, 1,085 seams)

| | val top1_pooled (±1.8) | val page recovery (±~8) |
|---|---|---|
| run 1 · conv · aug 0.5 | 0.88-0.89 | 0.76 (best checkpoint) |
| run 2 · + 116k PDFA real pages | 0.89-0.90 | 0.75-0.79 |

No difference detectable on this validation set (all within noise).

## Ranca & Murray 2013 (UnShredder) vs ours — 45 real test documents, single-document

Their unmodified code (probabilistic "prediction" cost + Kruskal search) in a Python 2.7
container, probabilistic model fitted on two D1 training-split pages. Adapter fixes: strips
cropped to the paper edge (their pipeline assumes straight edges), PIL right-angle rotations
made expanding (their code otherwise crops tall strips to an empty box), and their
likelihood score exported as a cost. Their "blackGaus" cost cannot run on any NumPy built
for Python 2.7 (np.logaddexp on Decimal).

| | ours (run 1) | Ranca & Murray |
|---|---|---|
| seam scoring: true neighbour ranked #1 | 98.0 ± 2.2 | 22.6 ± 5.2 |
| their / our score + same TSP solver: page recovery | 96.0 ± 4.0 | 13.5 ± 1.4 |
| end-to-end paper accuracy | 98.5 ± 1.9 | 23.2 ± 5.9 |
| end-to-end page recovery | 96.0 ± 4.0 | 13.0 ± 2.1 |
| pages 100% complete | 91 ± 8 | 0 |

## Simulated reviewer: run 2 vs run 3 (45 validation documents pooled, 1,063 inked strips, top-5 suggestions)

| model | auto-commit at | decisions/doc | auto-joins correct | docs 100% |
|---|---|---|---|---|
| run 2 | 0.9999 | 5.1 | 99.65% | 95.6% |
| run 2 | 0.99999 | 6.0 | 99.76% | 97.8% |
| run 3 final | 0.9999 | 2.4 | 92.45% | 51.1% |
| run 3 final | 0.9999999 | 4.8 | 96.22% | 66.7% |

Run 3 (run 2 + 20k steps) ranks seams as well as run 2 (val top1_pooled ~0.90) but is
badly overconfident: wrong auto-joins are never shown to the reviewer, so documents stay
broken. Use run 2 for the demo. Next: post-hoc temperature calibration on separate data
for every run; consider capping the learned score scale / label smoothing in training.

## Reviewer decisions per document to 100% (validation, 45 docs pooled, best calibration) — 2026-10-30

Oracle reviewer, top-5 suggestions, auto-join ≥ 0.9999, page-check rejections until every page is 100% correct; temperature chosen per model from 1/16…32 (both chose 4.0). CPU time ≈ 7–8 s per evaluation on cached scores after vectorising the engine (was 2 min 45 s; top1_pooled: < 1 ms). Likelihood-fitted temperature scaling (1 simulation, ~2 s) was tried: it lands far more cautious (t = 0.43 / 0.16, 9.3 vs 15.4 decisions/doc) — same ordering, but not the deploy operating point, so the grid stays.

| model | decisions/doc | 95% CI | rejections | auto-joins |
|---|---|---|---|---|
| run 2 | 3.42 | [1.91, 5.64] | 40 | 1005 |
| run 3 final | 5.29 | [3.29, 7.93] | 142 | 1037 |

Paired (run 3 − run 2): +1.87 decisions/doc, 95% CI [+0.80, +3.29]; run 3 cheaper on 8, dearer on 24, equal on 13 docs. Run 2 is better; run 3's extra training made it more confidently wrong.

## Comparison with the state of the art: mixed test piles (2026-11-01)

Each dataset's test split (never used for training or checkpoint selection) pooled into one shuffled pile:
D1 = D047–D061 (15 docs, 345 strips), D2 = D016–D020 (5 docs, 122 strips), D3 = D076–D100 (25 docs, 559 strips).
All methods scored on the same strips with Paixão et al.'s accuracy (fraction of correct adjacent pairs; a
document's last strip followed by another's first counts as correct) and their blank rule. 95% CIs by
bootstrap over documents; differences are paired by document.

| pile | run 2 (ours) | PR 2020 (Paixão, Pattern Recognition) | CVPR 2020 (Paixão) |
|---|---|---|---|
| D1 | **100.0%** [100.0, 100.0] | 99.1% [97.4, 100.0] | 90.4% [78.4, 98.0] |
| D2 | 82.6% [65.1, 94.9] | 88.4% [79.7, 95.0] | 90.9% [84.0, 97.6] |
| D3 | **97.8%** [96.8, 98.8] | 91.2% [87.6, 94.5] | n/a (its only model was trained on D3's source documents) |

Paired differences (run 2 − baseline): D1 vs PR 2020 +0.9 [0.0, +2.6], vs CVPR 2020 +9.6 [+2.0, +21.6];
D2 vs PR 2020 −5.8 [−27.2, +8.3], vs CVPR 2020 −8.3 [−30.4, +5.8]; D3 vs PR 2020 **+6.6 [+3.4, +10.1]**.
Run 2 is clearly better on D3 and vs CVPR 2020 on D1, tied with PR 2020 on D1, and not distinguishable on D2
(5 documents; the gap is one page, D018, 13/26 joins vs 25–26 for both baselines).

Caveat: run 2 trained on *other* documents of D1–D3 (same shredders and scanners); Paixão's models never saw
real shreds from these datasets. Run 2 saw 30 D1, 10 D2 and 50 D3 training documents, so its clearest win (D3)
is also where this home advantage is largest; the two cannot be separated without leave-one-dataset-out
retraining.

Baselines were run from the authors' released code and weights. PR 2020 uses their CDIP model on D1/D2 and their
ISRI model on D3, as in their paper. Checks: CVPR 2020 (original TF code, CPU) reproduces its published full-D2 pile
result (97.02% vs 97.22%). PR 2020 ran in its original TF 1.11 code on D1/D2 (99.13%, 88.43%); our PyTorch port
(`baselines/deeprec_pr20`, fixed Sauvola range r=127.5) reproduces both exactly and was used for D3 (~400 pairs/s
on one A6000, against ~25 pairs/s for the original code). Speed: PR 2020 evaluates its network on every pair of strips
(×21 vertical shifts), ours and CVPR 2020 embed each edge once.
Result files: S3 `unshredder/cmp/2026-11-01/`.

## Serbian Cyrillic PDF collection (FinePDFs srp_Cyrl → Common Crawl), 2026-11-01

One Hetzner cx23 (2 vCPU, Helsinki), signed S3 reads of `s3://commoncrawl` with a read-only key (not billed),
ramp-up throttle capped at 200 req/s, ~61 records/s overall, ~4.5 h including one crash and resume.

| | count |
|---|---|
| FinePDFs srp_Cyrl records | 944,855 |
| **PDFs stored (unique, complete)** | **739,910 (253.3 GB, 943 shards)** |
| cut off by Common Crawl (WARC-Truncated) | 167,772 (17.8%; the largest files, see TODO second pass) |
| not a PDF / incomplete | 11,376 / 4,435 |
| exact duplicates skipped | 22,062 (incl. 743 refetched after the crash) |
| evaluation documents excluded (sha256 list) | 43 hits on the 36 evaluation PDFs |

Index: `s3://unshredder/unshredder/collect/finepdfs/srp_Cyrl/index.sqlite`, one row per PDF with its
shard key and byte offset (one ranged GET per PDF). Shard 230 was overwritten after a resume (state.json
lagged an uploaded shard); `repair_shard.py` found 743 mismatching rows and refetched all 743 into shard 942.
Final check: 320/320 sampled PDFs (300 random + 20 from shards 230/942) match their SHA-256.
