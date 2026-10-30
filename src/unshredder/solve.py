"""Turn pairwise costs into ordered chains of strips (one chain per document)."""

from __future__ import annotations

import numpy as np

Chain = list[tuple[int, int]]  # [(strip, orient), ...] left to right


def _revflip(chain: Chain) -> Chain:
    """The same chain viewed rotated 180 degrees."""
    return [(s, 1 - o) for s, o in reversed(chain)]


class Linker:
    """Kruskal-style chain building: repeatedly join the right end of one chain to the
    left end of another. With flips, a chain may be rotated 180 degrees to make a join fit."""

    def __init__(self, strips: list[int], n_orient: int):
        self.n_orient = n_orient
        self.chains: dict[int, Chain] = {s: [(s, 0)] for s in strips}
        self.where = {s: s for s in strips}

    def try_link(self, a: int, b: int) -> bool:
        (i, oi), (j, oj) = divmod(a, self.n_orient), divmod(b, self.n_orient)
        if i not in self.where or j not in self.where:
            return False
        ca, cb = self.where[i], self.where[j]
        if ca == cb:
            return False
        A, B = self.chains[ca], self.chains[cb]
        if A[-1] != (i, oi):
            if self.n_orient == 2 and A[0] == (i, 1 - oi):
                A = _revflip(A)
            else:
                return False
        if B[0] != (j, oj):
            if self.n_orient == 2 and B[-1] == (j, 1 - oj):
                B = _revflip(B)
            else:
                return False
        self.chains[ca] = A + B
        del self.chains[cb]
        for s, _ in B:
            self.where[s] = ca
        return True

    def link_in_order(self, cost: np.ndarray, allowed: np.ndarray, max_cost: float, min_chains: int) -> None:
        masked = np.where(allowed, cost, np.inf)
        order = np.argsort(masked, axis=None)
        for k in order:
            if len(self.chains) <= min_chains:
                return
            a, b = divmod(int(k), cost.shape[0])
            if not masked[a, b] <= max_cost:
                return
            self.try_link(a, b)


def chain_cost(chain: Chain, cost: np.ndarray, n_orient: int) -> float:
    if len(chain) < 2:
        return 0.0
    nodes = np.array([s * n_orient + o for s, o in chain])
    return float(cost[nodes[:-1], nodes[1:]].sum())


def refine(chain: Chain, cost: np.ndarray, n_orient: int, max_seg: int = 3, max_passes: int = 20) -> Chain:
    """Local search: move short segments (or-opt) and, with flips, reverse segments (2-opt)."""
    best = list(chain)
    best_cost = chain_cost(best, cost, n_orient)
    n = len(best)

    def candidates():
        for L in range(1, min(max_seg, n - 1) + 1):
            for i in range(n - L + 1):
                seg, rest = best[i : i + L], best[:i] + best[i + L :]
                for j in range(len(rest) + 1):
                    if j != i:
                        yield rest[:j] + seg + rest[j:]
        if n_orient == 2:
            for i in range(n):
                for j in range(i + 2, n + 1):
                    yield best[:i] + _revflip(best[i:j]) + best[j:]

    for _ in range(max_passes):
        for cand in candidates():
            c = chain_cost(cand, cost, n_orient)
            if c < best_cost - 1e-9:
                best, best_cost = cand, c
                break
        else:
            break
    return best


def canonical_orientation(chain: Chain) -> Chain:
    """With no way to read the text, assume most strips were scanned upright."""
    flipped = sum(o for _, o in chain)
    return _revflip(chain) if flipped * 2 > len(chain) else chain


def tsp_path(cost: np.ndarray, nodes: list[int], runs: int = 10) -> list[int]:
    """Minimum-cost Hamiltonian path over `nodes` (asymmetric), as in Paixão et al.:
    add a dummy vertex joined to everything at zero cost, solve the ATSP, cut at the dummy.
    Solved with LKH (elkai) instead of Concorde."""
    import elkai

    n = len(nodes)
    if n <= 2:
        if n == 2 and cost[nodes[1], nodes[0]] < cost[nodes[0], nodes[1]]:
            return nodes[::-1]
        return list(nodes)
    sub = cost[np.ix_(nodes, nodes)]
    finite = sub[np.isfinite(sub)]
    top = float(finite.max()) if finite.size else 1.0
    # LKH works on integers: scale to a fixed resolution, impossible links get a big penalty.
    scaled = np.where(np.isfinite(sub), sub, 10 * top + 1) * (1e6 / max(10 * top + 1, 1e-9))
    m = np.zeros((n + 1, n + 1), dtype=np.int64)
    m[1:, 1:] = np.round(scaled).astype(np.int64)
    np.fill_diagonal(m, 0)
    tour = elkai.DistanceMatrix(m.tolist()).solve_tsp(runs=runs)[:-1]
    k = tour.index(0)
    return [nodes[i - 1] for i in tour[k + 1 :] + tour[:k]]


def split_path(path: list[int], score: np.ndarray, num_docs: int | None, threshold: float) -> list[list[int]]:
    """Cut a single path into documents at its least plausible links: the num_docs-1
    worst ones when the count is known, else every link scoring above threshold."""
    if len(path) < 2:
        return [path] if path else []
    links = np.array([score[a, b] for a, b in zip(path[:-1], path[1:])])
    if num_docs:
        cuts = set(np.argsort(-links)[: num_docs - 1].tolist())
    else:
        cuts = set(np.flatnonzero(links > threshold).tolist())
    docs, cur = [], [path[0]]
    for k, node in enumerate(path[1:]):
        if k in cuts:
            docs.append(cur)
            cur = []
        cur.append(node)
    docs.append(cur)
    return docs


def link_by_confidence(prob: np.ndarray, threshold: float, n_orient: int = 1,
                       strips: list[int] | None = None) -> list[Chain]:
    """Scalable assembly: accept links in order of decreasing confidence, each strip at most
    one left and one right neighbour, no cycles, and never below `threshold`. Pages end
    where no confident link exists, so a pile never collapses into one giant chain."""
    n = prob.shape[0] // n_orient
    linker = Linker(strips if strips is not None else list(range(n)), n_orient)
    p = np.where(np.isfinite(prob), prob, -1.0)
    order = np.argsort(-p, axis=None)
    for k in order:
        a, b = divmod(int(k), p.shape[0])
        if p[a, b] < threshold:
            break
        linker.try_link(a, b)
    return list(linker.chains.values())
