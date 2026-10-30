"""Reassemble a pile of strip images into documents.

1. Strips with no ink at all (pure margin) are set aside: nothing in their pixels says
   where they go.
2. Score every ordered pair of strips (costs.py: classic pixel edges or learned cnn).
3. Order the content strips: one global minimum-cost path through all of them
   (tsp, as in Paixão et al. 2020), or greedy Kruskal-style linking.
4. Cut the path into documents at its least plausible links.
5. Attach the blank strips to chain ends as margins.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import solve
from .config import ReconstructConfig
from .costs import SPLIT_DEFAULTS, compute


def reconstruct(
    images: list[np.ndarray], cfg: ReconstructConfig, num_docs: int | None = None
) -> tuple[list[solve.Chain], set[int], dict[tuple[int, int], float]]:
    """Returns chains of (strip_index, flipped) in reading order, largest first, the set
    of blank strips (placed by guesswork), and the model's confidence for each link
    (strip a -> strip b) in the chains, when the compatibility model provides one.

    `num_docs` overrides cfg.num_docs (used for the "oracle" setting)."""
    if num_docs is None and cfg.num_docs not in ("auto", None):
        if cfg.num_docs == "oracle":
            raise ValueError("num_docs='oracle' needs the true count; use it via `run`")
        num_docs = int(cfg.num_docs)
    threshold = cfg.split_threshold if cfg.split_threshold is not None else SPLIT_DEFAULTS[cfg.compat]
    c = compute(images, cfg)
    content = [s for s in range(len(images)) if not c.blank_strip[s]]

    if cfg.solver == "tsp":
        if cfg.allow_flips:
            raise ValueError("solver=tsp does not handle flipped strips; use solver=greedy")
        path = solve.tsp_path(c.cost, content)
        chains = [[(s, 0) for s in doc] for doc in solve.split_path(path, c.split_score, num_docs, threshold)]
    elif cfg.solver == "greedy" and c.link_prob is not None:
        chains = solve.link_by_confidence(c.link_prob, cfg.link_threshold, c.n_orient, content)
        chains = [solve.canonical_orientation(ch) for ch in chains]
    elif cfg.solver == "greedy":
        linker = solve.Linker(content, c.n_orient)
        plausible = np.isfinite(c.cost) & (c.split_score <= threshold)
        linker.link_in_order(c.cost, plausible, np.inf, num_docs or 1)
        if num_docs:  # forced count: fall back to any remaining link
            linker.link_in_order(c.cost, np.isfinite(c.cost), np.inf, num_docs)
        chains = list(linker.chains.values())
        if cfg.refine:
            chains = [solve.refine(ch, c.cost, c.n_orient) for ch in chains]
        chains = [solve.canonical_orientation(ch) for ch in chains]
    else:
        raise ValueError(f"unknown solver {cfg.solver!r}")

    blanks = [s for s in range(len(images)) if c.blank_strip[s]]
    chains = attach_blank_strips(chains, blanks, images)
    conf = {}
    if c.link_prob is not None:
        for ch in chains:
            for (a, oa), (b, ob) in zip(ch, ch[1:]):
                conf[(a, b)] = float(c.link_prob[a * c.n_orient + oa, b * c.n_orient + ob])
    return sorted(chains, key=len, reverse=True), set(blanks), conf


def attach_blank_strips(chains: list[solve.Chain], blanks: list[int], images: list[np.ndarray]) -> list[solve.Chain]:
    """Distribute ink-free strips over chain ends, alternating left and right margins."""
    chains = [list(ch) for ch in sorted(chains, key=len, reverse=True)]
    turn = 0
    for s in blanks:
        fits = [k for k, ch in enumerate(chains) if images[ch[0][0]].shape[0] == images[s].shape[0]]
        if not fits:
            chains.append([(s, 0)])
            continue
        k = fits[turn % len(fits)]
        if (turn // len(fits)) % 2 == 0:
            chains[k].insert(0, (s, 0))
        else:
            chains[k].append((s, 0))
        turn += 1
    return chains


def render(chain: solve.Chain, images: list[np.ndarray]) -> Image.Image:
    parts = [images[s][::-1, ::-1] if o else images[s] for s, o in chain]
    h = max(p.shape[0] for p in parts)
    parts = [np.pad(p, ((0, h - p.shape[0]), (0, 0)), constant_values=1.0) for p in parts]
    return Image.fromarray((np.hstack(parts) * 255).round().astype(np.uint8))


def save_reconstruction(
    chains: list[solve.Chain], blanks: set[int], names: list[str], images: list[np.ndarray], out_dir: Path,
    conf: dict[tuple[int, int], float] | None = None, render_images: bool = True,
) -> list[list[dict]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("doc_*.png"):
        old.unlink()
    conf = conf or {}
    result = []
    for ch in chains:
        entries = []
        for k, (s, o) in enumerate(ch):
            e = {"name": names[s], "flipped": bool(o), "blank": s in blanks}
            if k + 1 < len(ch) and (s, ch[k + 1][0]) in conf:
                e["conf"] = round(conf[(s, ch[k + 1][0])], 6)  # confidence of the link to the next strip
            entries.append(e)
        result.append(entries)
    if render_images:
        for k, chain in enumerate(chains):
            render(chain, images).save(out_dir / f"doc_{k:03d}.png")
    (out_dir / "reconstruction.json").write_text(json.dumps(result, indent=1))
    return result
