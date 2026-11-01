# unshredder

Reconstructs strip-shredded documents from a mixed pile of scanned strips. A learned seam
model (SeamNet) scores which strip continues which, confident joins are made automatically,
and a review app asks a person only about the uncertain ones.

On 45 held-out real shredded documents pooled into one pile, the model needs about
**3.4 reviewer decisions per document** to get every page 100% right (see `RESULTS.md`).

Automatic reconstruction of mixed piles, compared with the state of the art (Paixão et al. 2020,
run from their released code) on the same held-out test documents. Neighbour accuracy with 95% CIs:

| test pile | ours | PR 2020 | CVPR 2020 |
|---|---|---|---|
| D1, 15 docs | **100.0%** [100.0, 100.0] | 99.1% [97.4, 100.0] | 90.4% [78.4, 98.0] |
| D2, 5 docs | 82.6% [65.1, 94.9] | 88.4% [79.7, 95.0] | 90.9% [84.0, 97.6] |
| D3, 25 docs | **97.8%** [96.8, 98.8] | 91.2% [87.6, 94.5] | n/a |

Ours is better on D3 (+6.6 points, CI +3.4 to +10.1) and against CVPR 2020 on D1, tied with PR 2020 on D1,
and not distinguishable on D2 (5 documents). Our model trained on other documents of the same datasets;
theirs did not. Details in `RESULTS.md`.

Currently strip-cut only, upright strips. Cross-cut and rotated strips are in `TODO.md`.

## Setup

```bash
uv sync
uvx --from huggingface_hub hf download LatentLobster/unshredder-seamnet run2.pt --local-dir models   # model weights
```

A GPU is recommended for scoring; the app UI runs anywhere.

## Review app

The app has two parts: an inference API next to the GPU, and a local UI that forwards to it.

```bash
# on the GPU machine
unshredder serve --model models/run2.pt --jobs data/app_jobs

# on your machine (tunnel the API to port 8001 first, e.g. ssh -L 8001:127.0.0.1:8000 gpu-host)
unshredder app            # open http://localhost:8080
```

Upload PDFs (**Scan** → **Shred** to simulate shredding) or scanned strip images, then
**Unshred** and **Review**:

- **Review**: one question at a time — which strip continues this one? Keys `1–9` choose,
  `E` = page edge, `S` = ask later, `X` = cut a wrong join, `U` = undo. Strips can be dragged
  vertically to check alignment.
- **Pieces**: work on one piece until it is finished; view and cut whole pieces.
- **Quick confirm**: a grid of proposed joins to accept or reject in bulk (`Y` / `N` / `S`).

Jobs, answers and results are kept on disk under `--jobs`, so a review can be resumed.

## Command line

```bash
unshredder reconstruct strips/ -c configs/seamnet.yaml -s reconstruct.model_path=models/run2.pt
unshredder run -c configs/small.yaml                  # generate → shred → reconstruct → evaluate
unshredder benchmark datasets/D1 --mode mixed         # real shredded datasets (Paixão et al.)
unshredder review-sim val -c configs/seamnet.yaml     # simulated reviewer metric
unshredder train-seamnet -c configs/seamnet.yaml      # training
```

`unshredder <command> -h` lists the options; any config value can be overridden with
`-s key=value`.

## Layout

| path | |
|---|---|
| `src/unshredder/seamnet.py` | seam model, training, dense and sparse (top-k, multi-GPU) scoring |
| `src/unshredder/review.py` | review engine and simulated reviewer |
| `src/unshredder/server.py`, `static/` | review app API and UI |
| `src/unshredder/shred.py`, `bags.py` | shredding simulator, mixed-pile bags |
| `configs/` | run configurations |
| `RESULTS.md` | experiment log |
| `TODO.md` | open problems and backlog |

## Tests

```bash
uv run pytest
```

## License

MIT, see `LICENSE`. Model weights: [LatentLobster/unshredder-seamnet](https://huggingface.co/LatentLobster/unshredder-seamnet).
