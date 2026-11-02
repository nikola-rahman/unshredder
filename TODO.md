# Open questions / to revisit

## Low-information strips and "same document?" across a gap (raised 2026-10-30, parked)
Strips with almost no ink (margins, blank space between paragraphs) cannot be placed exactly
from their edges — by the model or by a person. Their exact order rarely matters for reading,
but whether the pieces on either side belong to the *same page/document* does.

Current handling: reviewer presses S ("not sure — ask me later"); if still unsure at the end the
pieces stay separate (a missing join is preferable to a wrong one for evidence).

Ideas, not decided:
- Line guides in the review: horizontal lines at THIS STRIP's text baselines across the whole
  view, to check that the line grid continues across a blank gap.
- A third answer "same page, exact position unsure": joins pieces into one page but marks the
  gap as uncertain (visible marker; recorded as uncertain in exports, not as a confirmed join).
- Model: page-identity evidence beyond the seam (line grid, margins, header/footer bands, font,
  toner, paper tone, cut width/slant) to score "same page" between pieces.

## Scaling to hundreds of thousands of strips (planned 2026-10-30)
Measured on the 36-document job (7,644 inked strips, A6000): embedding 19.4 s (linear, ~2.5 ms/strip),
all-pairs scoring 6.2 s (quadratic), 13 GB GPU. Extrapolated to 100k strips at once: ~4 min
embedding, ~18 min pairs, ~85 GB of edge embeddings (fp32) -> does not fit.

Workflow (user): batch by batch of strips — a batch is any set of strips, NOT a bag; strips of one page can arrive in
different batches. Unplaced strips/open ends stay in a pool indefinitely; more batches are added later.

1. Sparse scoring (in progress): blockwise, bf16, keep top-K (~50) candidates per edge in both
   directions + exact running log-sum-exp per row/column so confidences are unchanged; multi-GPU
   by blocks. Review engine works from the candidate lists, never a dense n x n matrix.
2. Persistent pool: each strip's edges embedded once and stored; a new batch is embedded and scored
   only against the pool's OPEN edges (a joined piece exposes only its 2 outer edges; finished
   pages leave the pool). Cost per batch ~ new strips x open edges; pieces can grow across batches.
3. If needed: coarse pre-filter (compact edge descriptor + ANN retrieval of ~200) before the full
   model; validate recall@K of the true neighbour first.

## Upside-down strips (raised 2026-10-30)
The app assumes every strip is upright; the shred simulator never flips strips. Real scans will contain
strips rotated 180 deg (left/right edges swapped, text upside down) -> unplaceable or wrong joins.
Plan: (1) orientation classifier per strip (trained on randomly rotated strips; text has a clear "up":
baselines, descenders/ascenders incl. Cyrillic ц щ д); (2) keep both orientations for uncertain
(near-blank) strips — reconstruct.py/link_by_confidence already support n_orient; (3) "rotate 180"
button in the review; (4) random flips in the simulator so metrics cover it.
Related: face-down strips (back of page; double-sided documents), mixed shredders (different widths).

## Backlog

### Drop no-information strips from training and evaluation (raised 2026-11-01)
Completely white strips (margins, blank pages, blank space) carry no seam evidence, yet they are
still sampled as training pairs, which teaches noise. Some evaluation metrics already skip them
(Paixão accuracy, page_recovery_inked, "inked" counts in the app), but not all do.
To do: one shared ink test (e.g. dark-pixel fraction in the 48 px edge band, threshold set on
real scans so scanner noise/paper texture doesn't count as ink), applied per seam not just per
strip (a strip can have text on one edge only); exclude blank-blank seams from contrastive
training pairs (keep blank strips as negatives/null-edge examples only if that helps); exclude
them from every eval metric and the reviewer simulation; report how many were dropped.

### Cross-cut shredding (raised 2026-11-01)
Cross-cut / confetti shredders cut strips again horizontally into small rectangular particles
(DIN 66399 P-3..P-5; micro-cut P-6/P-7 is far smaller). Each piece has 4 edges, so matching becomes
2-D (left/right AND top/bottom), and pieces are tiny (often a few characters).
To do: shredder simulator for cross-cut (particle sizes per DIN level, cut angle/jitter, torn
corners); SeamNet variant scoring horizontal seams too (shared encoder, rotated bands); 2-D
assembly (grid placement instead of chains) and a review UI for 2-D pieces; find real cross-cut
datasets for evaluation; measure per DIN level what is still recoverable.

### Graph neural network for assembly (raised 2026-11-01)
The pile is naturally a graph: strips are nodes, top-k seam candidates are edges (already produced by
sparse scoring). A GNN can pass messages along candidate edges and score joins jointly instead of pair
by pair (one strip can have only one right neighbour; line grid / margins / font must stay consistent
along a chain). Related: DiffAssemble (GNN + diffusion, 2024), ReassembleNet (ICCV 2025). Start as a
re-scorer on top of SeamNet's top-k graph; measure on reviewer decisions per document.

### Reinforcement learning / search for assembly (raised 2026-11-01)
Treat assembly as sequential decisions (which join to commit next, or which question to ask the
reviewer next) and learn a policy, e.g. SD2RL (AAAI 2023), deep MCTS (Alphazzle, 2023). Reward:
correct pages for fewest reviewer decisions. Likely only practical inside a piece / small candidate
set, not over the whole pile.

### Second pass: PDFs that Common Crawl cut off (raised 2026-11-01)
~17% of srp_Cyrl FinePDFs records are WARC-Truncated: the crawler stopped at its size limit, so the
lost ones are the largest PDFs (often multi-page reports / official documents). This biases the
collected set toward short documents.
To do: (1) record skipped records in a `skipped` table of the index (id, url, reason, WARC-Truncated
length) so a second pass needs no re-scan and the size limit / lost sizes can be measured;
(2) second pass fetching the truncated ones from their original URLs (FinePDFs `url`), respecting
robots.txt, rate-limited per host, stored separately and flagged as live-fetched with the date;
(3) a small test on a sample first to see how many URLs are still online.

### Two validation sets: academic and practical (raised 2026-11-02)
Validate every run on two sets, logged side by side:
1. Academic: D1/D2/D3 validation splits (as now) — comparison with published work on public benchmarks.
2. Practical: Serbian documents.
   a. Simulated (buildable now): held-out Serbian PDFs — the 36 evaluation PDFs (already excluded from
      training via configs/finepdfs_eval_exclude.sha256) plus a few shards reserved and never synced for
      training — rendered and cut by the GPU shredder with a fixed seed, so the set is identical every run.
      Needs a validation mode for "rendered pages + simulator with fixed seed".
   b. Real (needs physical work): 20-30 Serbian pages through a strip shredder like those used in the
      ministries, scanned. The only end-to-end test of the actual use case; small but most valuable.
Also noted: run 4 (no D1-D3 shreds) stays flat at ~0.37 pooled top-1 on the D1-D3 validation splits while
its training accuracy rises — the sim-to-real gap; run 2 reached 0.90 by step 1,500 with D1-D3 real shreds.
