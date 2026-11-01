# Related work — strip-shredded document reconstruction (checked 2026-11-01)

Web search over 2021–2026 (arXiv, IEEE, Elsevier, Springer, ACL Anthology, Brazilian SBC
proceedings). Google Scholar "cited by" was not accessible, and Chinese-language venues were
not searched, so this may be incomplete.

## Strip-cut methods with results on the public real-shred benchmarks (D1 S-MARQUES, D2 S-ISRI-OCR, D3 S-CDIP)

| work | method | notes |
|---|---|---|
| Paixão et al., Pattern Recognition 107, 2020 ("deeprec-pr20", arXiv 2007.00779) | self-supervised SqueezeNet valid/invalid pair classifier + TSP (Concorde) | > 90% on all three datasets, mixed; **released weights**; ported in `baselines/deeprec_pr20/` |
| Paixão et al., CVPR 2020 (arXiv 2003.10063) | asymmetric metric learning (edge embeddings + dot product) | 97.22% D2 mixed, ~22× faster than PR20 at similar accuracy; **code + pretrained model** at github.com/thiagopx/deeprec-cvpr20 |
| Paixão PhD thesis (UFES 2022), CTD 2023, SIBGRAPI-WiP 2023 | summary of the two above | no new method |

No later method reporting on these benchmarks was found. Their line of work remains the
reference on public real shreds.

## Human in the loop — the reference for our reviewer metric

- **Paixão et al., "A human-in-the-loop recommendation-based framework for reconstruction of
  mechanically shredded documents", Pattern Recognition Letters 164, 2022**
  (doi:10.1016/j.patrec.2022.10.011); code at github.com/thiagopx/deeprec-prl22.
  - A recommender picks pairs of adjacent shreds for a person to verify; four strategies were compared.
  - It reports > 40% fewer errors, and it proposes a measure of how much human effort improves the result.
  - **Read in full before claiming anything about reviewer effort**; ideally adopt their protocol as a second effort metric.

## Newer work, not directly comparable

- **Wu, Chiang & Liu, IJRA 15(1), 2026** (Taiwan Ministry of Justice Investigation Bureau).
  - Method: PR20-style SqueezeNet v1.1 self-supervised on synthetic shreds, with an asymmetric TSP solved by Concorde.
  - Physically shredded documents: 86.5% Simplified Chinese, 94.8% Western scripts.
  - Uses its own test documents and no public-benchmark comparison was found. It confirms non-Latin scripts are harder (relevant for Cyrillic).
- **ShredBench, ACL 2026 Findings (arXiv 2604.23813).**
  - Multimodal LLMs reorder 8–16 rendered fragments of English, Chinese, code and tables.
  - Scores collapse as fragmentation grows (Gemini 3 Pro is the strongest), and OCR models fail.
  - These are synthetic renders with few pieces, so it is no competitor at pile scale. It could serve as a semantic-reasoning aid later.
- **Generic 2D puzzle / fragment reassembly.**
  - Examples: Rika et al. arXiv 2501.19325, 2025 (deep compatibility + genetic algorithm, square pieces); ReassembleNet, ICCV 2025 (frescoes); PuzzleFusion.
  - None targets strip-shredded text or reports on the benchmarks above.
- **Cross-cut / hand-torn.**
  - Examples: SIViP 2025 (doi:10.1007/s11760-025-04807-z; Siamese/GNN + DQN, 94.6% on its own data); ELM-based compatibility, Soft Computing 2022.
  - Relevant to the cross-cut backlog item in `TODO.md`.

## Consequence for our comparison

1. **Seam scoring and reconstruction:** run PR20 with its released weights and our leave-one-dataset-out checkpoints on the same D1/D2/D3 test documents, through the same solver and metric code (neighbour accuracy, single and mixed). Run CVPR 2020 the same way from its released model (github.com/thiagopx/deeprec-cvpr20).
2. **Human effort:** reviewer decisions per document to 100%, plus the effort measure from Paixão et al. PRL 2022, for our model, PR20 and CVPR 2020 scores, using their released PRL22 code to reproduce their protocol.

## Single document or mixed pile? (verified in the released code where available)

| work | setting |
|---|---|
| Ranca & Murray 2013 | single document only |
| Paixão PR 2020 (deeprec-pr20) | mixed: all test documents of a collection pooled; single also reported |
| Paixão CVPR 2020 (deeprec-cvpr20) | both: `experiment1` = single, `experiment2` = multi (mixed) |
| Paixão PRL 2022 (deeprec-prl22) | mixed: `experiments/multi.py` pools every document of a test set (`filter_blanks=True`); the human-in-the-loop run (`user_guided.py`) starts from those mixed reconstructions |
| Wu et al. 2026 | not stated in the abstract |
| ShredBench 2026 | one document per instance, 8–16 Voronoi pieces (not strips, not piles) |
| ours | both; the app and reviewer metric use mixed piles |

PRL 2022 measures human effort as a budget: the reviewer checks a fixed share of the n−1 adjacent
pairs of the current solution (in one or more rounds), and the metric is accuracy after that budget.
Ours is the inverse: the number of decisions until every page is 100% correct.

## Other approaches since 2020 (not evaluated on strip-shredded text)

- Learning to solve puzzles end-to-end instead of score + solver: deep RL (SD2RL, AAAI 2023), deep
  MCTS (Alphazzle, 2023), GNN + diffusion (DiffAssemble, 2024; ReassembleNet, ICCV 2025), deep
  compatibility + genetic algorithm (Rika et al., 2025). Square pieces or frescoes, small piece counts.
- Multimodal LLMs reading the fragments and writing out the text (ShredBench, 2026). Its nonsense-text
  control shows the output leans on language priors (ROUGE-L 0.73 -> 0.33), i.e. it can invent text:
  unsuitable as evidence, possibly useful as a hint for low-ink strips.
- Production systems: Fraunhofer IPK ePuzzler for the hand-torn Stasi files (~91k pages by ~2018; no
  newer status found).
