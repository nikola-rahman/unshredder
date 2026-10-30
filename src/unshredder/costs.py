"""Pairwise compatibility between strips.

Strips are indexed as oriented nodes: node = strip * n_orient + orient, where orient 1
means the strip rotated 180 degrees. Matrices are [a, b] = "node b placed immediately
to the right of node a". Lower cost is better; inf means impossible.

Two compatibility models:
  classic  hand-crafted pixel-edge dissimilarity (+ row-profile cue on blank seams)
  seamnet  contrastive edge bi-encoder (seamnet.py), evidence summed down the seam
  cnn      Paixão et al. 2020 patch classifier (deep.py), kept as a baseline
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .config import ReconstructConfig

# Default "auto" split thresholds on split_score, per compatibility model.
SPLIT_DEFAULTS = {"classic": 0.73, "cnn": 0.5, "seamnet": 0.0}


@dataclass
class Costs:
    cost: np.ndarray  # solver cost per oriented pair
    split_score: np.ndarray  # how implausible a link is as a within-document seam
    blank_strip: np.ndarray  # per strip: no ink anywhere (pure margin)
    n_orient: int
    link_prob: np.ndarray | None = None  # model's P(b directly right of a), learned models only


def normalize_strip(img: np.ndarray) -> np.ndarray:
    """Rescale so the paper background is ~1.0 (undoes per-strip exposure drift)."""
    bg = float(np.percentile(img, 95))
    return np.clip(img / bg, 0, 1) if bg > 0.05 else img


def oriented(images: list[np.ndarray], allow_flips: bool) -> list[np.ndarray]:
    out = []
    for im in images:
        out.append(im)
        if allow_flips:
            out.append(im[::-1, ::-1])
    return out


def _edge_cost(nodes: list[np.ndarray], cfg: ReconstructConfig) -> np.ndarray:
    """Edge dissimilarity for nodes that all share the same height."""
    r1 = np.stack([im[:, -1] for im in nodes])
    r2 = np.stack([im[:, -2] if im.shape[1] > 1 else im[:, -1] for im in nodes])
    l1 = np.stack([im[:, 0] for im in nodes])
    l2 = np.stack([im[:, 1] if im.shape[1] > 1 else im[:, 0] for im in nodes])
    n = len(nodes)
    out = np.empty((n, n), dtype=np.float64)
    chunk = max(1, int(2e7 // (n * r1.shape[1] + 1)))
    for s in range(0, n, chunk):
        a = slice(s, s + chunk)
        if cfg.metric == "prediction":
            # Extrapolate each side one pixel across the cut and compare with the other side.
            pred_r = np.clip(2 * r1[a] - r2[a], 0, 1)[:, None, :]
            pred_l = np.clip(2 * l1 - l2, 0, 1)[None, :, :]
            d = (np.abs(pred_r - l1[None]) ** cfg.p + np.abs(pred_l - r1[a][:, None, :]) ** cfg.p) / 2
        elif cfg.metric == "l1":
            d = np.abs(r1[a][:, None, :] - l1[None]) ** cfg.p
        else:
            raise ValueError(f"unknown metric {cfg.metric!r}")
        # Normalise by the ink present at the seam (a soft Jaccard distance). Without this,
        # a text edge looks closer to blank paper than to its slightly misaligned true
        # neighbour, because blank paper only "disagrees" on the few ink pixels.
        ink = np.maximum(1 - r1[a][:, None, :], 1 - l1[None]).sum(axis=2)
        out[a] = d.sum(axis=2) / (ink + 1.0)
    return out


def _profile_distance(nodes: list[np.ndarray]) -> np.ndarray:
    """1 - correlation of row ink profiles: strips of one page share text line positions."""
    profs = np.stack([gaussian_filter1d((1 - im).mean(axis=1), 1.5) for im in nodes])
    std = profs.std(axis=1, keepdims=True)
    valid = std[:, 0] > 1e-3
    z = (profs - profs.mean(axis=1, keepdims=True)) / np.maximum(std, 1e-6)
    dist = 1 - (z @ z.T) / profs.shape[1]
    dist[~valid, :] = 1.0  # featureless strip: neutral
    dist[:, ~valid] = 1.0
    return dist


def _dark_fraction(x: np.ndarray) -> float:
    return float((x < 0.5).mean())


def _per_height(nodes: list[np.ndarray], fn) -> np.ndarray:
    """Apply a pairwise fn within groups of equal height; other pairs are inf."""
    n = len(nodes)
    out = np.full((n, n), np.inf)
    heights = np.array([im.shape[0] for im in nodes])
    for h in np.unique(heights):
        idx = np.flatnonzero(heights == h)
        out[np.ix_(idx, idx)] = fn([nodes[i] for i in idx])
    return out


def _classic(nodes: list[np.ndarray], cfg: ReconstructConfig) -> tuple[np.ndarray, np.ndarray]:
    edge = _per_height(nodes, lambda g: _edge_cost(g, cfg))
    prof = _per_height(nodes, _profile_distance)
    blank_r = np.array([_dark_fraction(im[:, -1]) < cfg.blank_threshold for im in nodes])
    blank_l = np.array([_dark_fraction(im[:, 0]) < cfg.blank_threshold for im in nodes])
    blank_seam = blank_r[:, None] & blank_l[None, :]
    # Two blank edges match perfectly yet say nothing; only the row profile can tell
    # whether the strips belong to the same page.
    cost = np.where(blank_seam, prof, edge + cfg.profile_weight * prof)
    split = np.where(blank_seam, prof, -np.inf)  # inked seams are never split
    return cost, split


def _cnn(nodes: list[np.ndarray], cfg: ReconstructConfig) -> tuple[np.ndarray, np.ndarray]:
    from . import deep  # torch is only imported when needed

    C = deep.compatibility(nodes, cfg)
    cost = C.max() - C
    return cost, cost


def _softmax_with_null(S: np.ndarray, null: float, axis: int) -> np.ndarray:
    m = np.maximum(S.max(axis=axis, keepdims=True), null)
    e = np.exp(S - m)
    return e / (e.sum(axis=axis, keepdims=True) + np.exp(null - m))


def _seamnet(nodes: list[np.ndarray], cfg: ReconstructConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from . import seamnet  # torch is only imported when needed

    S, null = seamnet.scores(nodes, cfg.model_path, cfg.max_shift, cfg.device)
    S = S.copy()
    np.fill_diagonal(S, -np.inf)
    nv = float(null.flat[0])
    # Link confidence: a's right edge choosing b (over all candidates and "no neighbour")
    # AND b's left edge choosing a. The smaller of the two is used, so confident links
    # are mutual.
    prob = np.minimum(_softmax_with_null(S, nv, 1), _softmax_with_null(S, nv, 0))
    finite = np.where(np.isfinite(S), S, np.nan)
    # split_score > 0: the learned "no neighbour" option beats this link
    return np.nanmax(finite) - S, null - S, prob


def compute(images: list[np.ndarray], cfg: ReconstructConfig) -> Costs:
    if cfg.normalize:
        images = [normalize_strip(im) for im in images]
    n_orient = 2 if cfg.allow_flips else 1
    nodes = oriented(images, cfg.allow_flips)
    link_prob = None
    if cfg.compat == "classic":
        cost, split = _classic(nodes, cfg)
    elif cfg.compat == "seamnet":
        cost, split, link_prob = _seamnet(nodes, cfg)
    elif cfg.compat == "cnn":
        cost, split = _cnn(nodes, cfg)
    else:
        raise ValueError(f"unknown compat {cfg.compat!r}")

    # A strip can't neighbour itself (in either orientation).
    strip_of = np.arange(len(nodes)) // n_orient
    cost = cost.astype(np.float64)
    cost[strip_of[:, None] == strip_of[None, :]] = np.inf
    blank_strip = np.array([_dark_fraction(im) < cfg.blank_threshold for im in images])
    return Costs(cost, split, blank_strip, n_orient, link_prob)
