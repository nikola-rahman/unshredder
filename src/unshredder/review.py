"""Human-in-the-loop review: confident joins are committed automatically, a reviewer
answers the uncertain ones from top-K suggestions, and every answer cascades.

`ReviewEngine` is shared by the web app (a person answers) and `simulate` (an oracle
answers from ground truth, optionally with an error rate), so what is measured offline is
exactly what a reviewer experiences.

Loop:
  1. auto-commit links whose (mutual) confidence is >= `auto_threshold`
  2. pick the most useful open edge (highest best-candidate confidence: likely a quick
     answer, and every commit removes competitors elsewhere)
  3. show its top-K candidates (+ "no neighbour"); the reviewer picks one, or asks for
     more; in the simulation a miss beyond `deep_k` counts as a manual search
  4. the answer becomes a hard constraint; confidences are recomputed on what is left;
     repeat until every strip's right edge is decided.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

END = -1  # "this strip has no right neighbour here" (page border or missing strip)


class ReviewSession:
    """Decided links over a pile. S: scores (n, n), higher = more likely b right of a;
    null: score of "no neighbour" (same scale)."""

    def __init__(self, S: np.ndarray, null: float, candidates: int = 50,
                 forbidden: set[tuple[int, int]] = frozenset()):
        n = S.shape[0]
        self.n, self.null = n, null
        S = S.copy()
        np.fill_diagonal(S, -np.inf)
        self.no_end = {a for a, b in forbidden if b == END}  # "page ends here" was rejected
        self.forbidden = {(a, b) for a, b in forbidden if b != END}
        for a, b in self.forbidden:
            S[a, b] = -np.inf
        self.S = S
        m = max(1, min(candidates, n - 1))
        # sparse candidate lists keep every update cheap on piles of thousands of strips
        top = np.argpartition(-S, m - 1, axis=1)[:, :m] if n > 1 else np.zeros((n, 0), int)
        order = np.argsort(-np.take_along_axis(S, top, 1), axis=1)
        self.cand = np.take_along_axis(top, order, 1)  # (n, m), best first
        self.cand_S = np.take_along_axis(S, self.cand, 1)
        self.null_col = np.full(n, float(null))
        self.null_col[list(self.no_end)] = -np.inf
        self.right: list[int | None] = [None] * n
        self.left: list[int | None] = [None] * n
        self.has_left = np.zeros(n, bool)
        # chain id of every strip (to forbid cycles), relabelled on merge (small into large)
        self.root = np.arange(n)
        self.members = {a: [a] for a in range(n)}

    def _root(self, a: int) -> int:
        return int(self.root[a])

    def option_matrix(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """For edges `rows`: candidate ids (len(rows), m) and probabilities (len(rows), m + 1)
        over the still-available candidates + "no neighbour" (last column); 0 = unavailable."""
        C = self.cand[rows]
        sc = np.where(~self.has_left[C] & (self.root[C] != self.root[rows][:, None]), self.cand_S[rows], -np.inf)
        sc = np.concatenate([sc, self.null_col[rows][:, None]], 1)
        mx = sc.max(1, keepdims=True)
        p = np.exp(sc - np.where(np.isfinite(mx), mx, 0))
        tot = p.sum(1, keepdims=True)
        return C, np.divide(p, tot, out=np.zeros_like(p), where=tot > 0)

    def available(self, a: int) -> list[int]:
        """Candidates b that a could still link to (b's left is free, no cycle)."""
        return self.options(a)[0]

    def options(self, a: int) -> tuple[list[int], np.ndarray]:
        """Available candidates and their probabilities (softmax over them + "no neighbour")."""
        C, p = self.option_matrix(np.array([a]))
        keep = (~self.has_left[C[0]]) & (self.root[C[0]] != self.root[a]) & np.isfinite(self.cand_S[a])
        return [int(b) for b in C[0][keep]], np.append(p[0, :-1][keep], p[0, -1])  # p[-1] is "no neighbour"

    def can_link(self, a: int, b: int) -> bool:
        """Any free strip may be chosen (a reviewer can search beyond the shortlist), as
        long as it has no left neighbour yet and the link would not close a loop."""
        if b == END:
            return a not in self.no_end
        return b != a and self.left[b] is None and self._root(a) != self._root(b) and (a, b) not in self.forbidden

    def commit(self, a: int, b: int) -> None:
        self.right[a] = b
        if b != END:
            self.left[b] = a
            self.has_left[b] = True
            ra, rb = int(self.root[a]), int(self.root[b])
            if len(self.members[ra]) < len(self.members[rb]):
                ra, rb = rb, ra
            moved = self.members.pop(rb)
            self.root[moved] = ra
            self.members[ra] += moved

    def open_edges(self) -> list[int]:
        return [a for a in range(self.n) if self.right[a] is None]

    def chains(self) -> list[list[int]]:
        """Current assembled pieces, left to right."""
        out = []
        for a in range(self.n):
            if self.left[a] is None:
                ch, cur = [a], a
                while self.right[cur] not in (None, END):
                    cur = self.right[cur]
                    ch.append(cur)
                out.append(ch)
        return out

    def left_context(self, a: int, k: int) -> list[int]:
        """Up to k strips ending at a (a last)."""
        ctx = [a]
        while len(ctx) < k and self.left[ctx[0]] is not None:
            ctx.insert(0, self.left[ctx[0]])
        return ctx

    def right_context(self, b: int, k: int) -> list[int]:
        """Up to k strips starting at b."""
        ctx = [b]
        while len(ctx) < k and self.right[ctx[-1]] not in (None, END):
            ctx.append(self.right[ctx[-1]])
        return ctx


class ReviewEngine:
    """Auto-commit + question selection + answers + rejections, with undo by replaying the
    reviewer's actions. Actions: ("link", a, b) and ("reject", a, b), b may be END."""

    def __init__(self, S: np.ndarray, null: float, auto_threshold: float = 0.9999, candidates: int = 50):
        self.S, self.null, self.auto_threshold, self.candidates = S, null, auto_threshold, candidates
        self.actions: list[tuple[str, int, int]] = []
        self._rebuild()

    @property
    def decisions(self) -> list[tuple[int, int]]:
        return [(a, b) for kind, a, b in self.actions if kind == "link"]

    def _rebuild(self) -> None:
        rejected = {(a, b) for kind, a, b in self.actions if kind == "reject"}
        self.session = ReviewSession(self.S, self.null, self.candidates, rejected)
        self.auto: list[tuple[int, int]] = []
        self.auto_commit()
        for kind, a, b in self.actions:
            if kind == "link" and self.session.right[a] is None and self.session.can_link(a, b):
                self.session.commit(a, b)
                self.auto_commit()

    def auto_commit(self) -> int:
        sess, added = self.session, 0
        changed = True
        while changed:
            changed = False
            best = {}
            rows = np.array(sess.open_edges(), int)
            if len(rows):
                C, P = sess.option_matrix(rows)
                i = P.argmax(1)
                pm = P[np.arange(len(rows)), i]
                for r in np.flatnonzero(pm >= self.auto_threshold):
                    best[int(rows[r])] = (END if i[r] == C.shape[1] else int(C[r, i[r]]), float(pm[r]))
            claimed: dict[int, int] = {}
            for a, (b, p) in best.items():  # mutual: b's strongest open claimant must be a
                if b != END and p >= self.auto_threshold and (b not in claimed or p > best[claimed[b]][1]):
                    claimed[b] = a
            for a, (b, p) in best.items():
                if p < self.auto_threshold or sess.right[a] is not None:
                    continue
                if (b == END and sess.can_link(a, END)) or (claimed.get(b) == a and sess.can_link(a, b)):
                    sess.commit(a, b)
                    self.auto.append((a, b))
                    added += 1
                    changed = True
        return added

    def next_question(self, later: list[int] = ()) -> int | None:
        """The open edge to ask about, or None when everything is decided. Edges in `later`
        (the reviewer was not sure) are asked only when nothing else is left, oldest first."""
        open_now = self.session.open_edges()
        held = set(later)
        if open_now and all(a in held for a in open_now):
            return next(a for a in later if self.session.right[a] is None)
        best_a = None
        rows = np.array([a for a in open_now if a not in held], int)
        if len(rows):
            pm = self.session.option_matrix(rows)[1].max(1)
            if pm.max() > 0:
                best_a = int(rows[int(np.argmax(pm))])
        if best_a is None:  # open edges with no option left (every choice rejected)
            remaining = self.session.open_edges()
            return remaining[0] if remaining else None
        return best_a

    def suggestions(self, a: int, k: int) -> list[tuple[int, float]]:
        """Top-k options for a, best first, as (strip or END, probability)."""
        avail, p = self.session.options(a)
        ranked = [(END if i == len(avail) else avail[i], float(p[i])) for i in np.argsort(-p)]
        if a in self.session.no_end:
            ranked = [r for r in ranked if r[0] != END]
        return ranked[:k]

    @property
    def starts(self) -> set[int]:
        """Strips the reviewer marked as a page's left edge ("page starts here")."""
        return {a for kind, a, _ in self.actions if kind == "start"}

    def left_suggestions(self, b: int, k: int) -> list[tuple[int, float]]:
        """Top-k strips that could go immediately LEFT of b (their right edge is free, no
        loop), plus END = "page starts here", best first, as (strip or END, probability)."""
        sess = self.session
        n = sess.n
        free = np.array([r is None for r in sess.right]) & (sess.root != sess.root[b]) & np.isfinite(sess.S[:, b])
        free[b] = False
        cands = np.flatnonzero(free)
        sc = np.append(sess.S[cands, b], self.null)
        p = np.exp(sc - sc.max())
        p /= p.sum()
        order = np.argsort(-p)[:k]
        return [(END if i == len(cands) else int(cands[i]), float(p[i])) for i in order]

    def batch(self, n: int, later: list[int] = ()) -> list[tuple[int, int, float]]:
        """Up to n questions for quick confirmation: (a, the model's best option b, its
        probability), most confident first, never proposing the same strip twice (so all
        verdicts in a batch can be applied together). Questions saved for later are skipped."""
        held = set(later)
        rows = np.array([a for a in self.session.open_edges() if a not in held], int)
        if not len(rows):
            return []
        C, P = self.session.option_matrix(rows)
        best = P.argmax(1)
        pm = P[np.arange(len(rows)), best]
        out, used = [], set()
        for r in np.argsort(-pm, kind="stable"):
            if pm[r] <= 0:
                break
            b = END if best[r] == C.shape[1] else int(C[r, best[r]])
            if b != END and b in used:
                continue
            used.add(b)
            out.append((int(rows[r]), b, float(pm[r])))
            if len(out) >= n:
                break
        return out

    def ranked(self, a: int, side: str = "right") -> list[tuple[int, float]]:
        """EVERY strip that could still go right of a (side="right") or left of a ("left"),
        ranked by the model, with probabilities over all of them + the page edge (END, last
        place it ranks). Unlike `suggestions`, not limited to the shortlist."""
        sess = self.session
        if side == "right":
            free = ~sess.has_left
            sc_all = sess.S[a]
        else:
            free = np.array([r is None for r in sess.right])
            sc_all = sess.S[:, a]
        free = free & (sess.root != sess.root[a]) & np.isfinite(sc_all)
        free[a] = False
        cands = np.flatnonzero(free)
        edge_ok = not (side == "right" and a in sess.no_end) and not (side == "left" and a in self.starts)
        sc = np.append(sc_all[cands], self.null if edge_ok else -np.inf)
        p = np.exp(sc - sc.max())
        p /= p.sum()
        order = np.argsort(-p, kind="stable")
        return [(END if i == len(cands) else int(cands[i]), float(p[i])) for i in order
                if i < len(cands) or edge_ok]

    def mark_start(self, b: int) -> None:
        if b not in self.starts:
            self.actions.append(("start", b, END))

    def answer(self, a: int, b: int) -> int:
        """Apply a reviewer decision; returns how many links were auto-committed after it."""
        if self.session.right[a] is not None:
            raise ValueError("this edge is already decided")
        if not self.session.can_link(a, b):
            raise ValueError("that strip is no longer available")
        self.actions.append(("link", a, b))
        self.session.commit(a, b)
        return self.auto_commit()

    def reject(self, pairs: list[tuple[int, int]]) -> None:
        """The reviewer marks decided links as wrong (seen on the assembled page); they are
        undone and forbidden, and the affected edges go back into the question queue."""
        self.actions += [("reject", a, b) for a, b in pairs]
        self._rebuild()

    def remove(self, i: int) -> bool:
        """Take back one earlier action (not just the last); the rest is replayed."""
        if not 0 <= i < len(self.actions):
            return False
        del self.actions[i]
        self._rebuild()
        return True

    def undo(self) -> bool:
        if not self.actions:
            return False
        self.actions.pop()
        self._rebuild()
        return True

    def progress(self) -> dict:
        n = self.session.n
        open_edges = len(self.session.open_edges())
        return {"strips": n, "decided": n - open_edges, "open": open_edges, "your_decisions": len(self.actions),
                "auto_links": len(self.auto), "pieces": len(self.session.chains())}


# ---------------------------------------------------------------- offline simulation


@dataclass
class ReviewResult:
    clicks: int = 0
    searches: int = 0
    rejections: int = 0
    auto_links: int = 0
    auto_wrong: int = 0
    asked: int = 0
    in_top_k: int = 0
    per_page: dict = field(default_factory=dict)  # page -> reviewer decisions spent on it
    curve: list[tuple[int, float]] = field(default_factory=list)  # (decisions, pages complete)

    def summary(self, pages: int, seed: int = 0) -> dict:
        decisions = self.clicks + self.searches + self.rejections
        per = np.array(list(self.per_page.values()), dtype=float)
        rng = np.random.default_rng(seed)
        boot = [per[rng.integers(0, len(per), len(per))].mean() for _ in range(2000)] if len(per) else [0.0]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return {
            "pages": pages,
            "reviewer_decisions": decisions,
            "decisions_per_doc_to_100": round(decisions / max(1, pages), 3),
            "decisions_per_doc_ci95": [round(float(lo), 3), round(float(hi), 3)],
            "rejections": self.rejections,
            "searches": self.searches,
            "auto_links": self.auto_links,
            "auto_link_precision": round(1 - self.auto_wrong / max(1, self.auto_links), 4),
            "questions": self.asked,
            "answer_in_top_k": round(self.in_top_k / max(1, self.asked), 4),
            "pages_complete_final": self.curve[-1][1] if self.curve else 0.0,
        }


def _blocking_wrong_links(sess: ReviewSession, a: int, truth: int, truth_right: list[int]) -> list[tuple[int, int]]:
    """Wrong decided links that stop a from linking to its true partner: whatever holds the
    partner, plus wrong links along the chains of a and of the partner (loop case)."""
    if truth == END:
        return [(a, END)] if a in sess.no_end else []
    out = set()
    if sess.left[truth] is not None and sess.left[truth] != a:
        out.add((sess.left[truth], truth))
    for start in (a, truth):
        for x in sess.left_context(start, sess.n) + sess.right_context(start, sess.n):
            y = sess.right[x]
            if y is not None and y != END and y != truth_right[x]:
                out.add((x, y))
    return sorted(out)


def simulate(S: np.ndarray, null: float, truth_right: list[int], page_of: list, k: int = 5, deep_k: int = 20,
             auto_threshold: float = 0.9999, error_rate: float = 0.0, seed: int = 0, curve_every: int = 25,
             max_rounds: int = 50) -> ReviewResult:
    """Oracle review until every page is 100% correct. truth_right[a] = index of a's true
    right neighbour in the pile, or END. Decisions: picking an answer (1 click, 2 if it was
    beyond the top-k, +1 manual search if beyond deep_k) and rejecting a wrong join seen on
    the assembled page (1 each)."""
    rng = np.random.default_rng(seed)
    eng = ReviewEngine(S, null, auto_threshold)
    res = ReviewResult()
    pages = sorted(set(page_of), key=str)
    res.per_page = {p: 0 for p in pages}
    members = {p: [a for a in range(eng.session.n) if page_of[a] == p] for p in pages}

    def pages_complete() -> float:
        sess = eng.session
        done = sum(all(sess.right[a] == truth_right[a] for a in members[p]) for p in pages)
        return round(done / max(1, len(pages)), 4)

    def count_auto(start: int) -> None:
        for a, b in eng.auto[start:]:
            res.auto_links += 1
            res.auto_wrong += b != truth_right[a]

    count_auto(0)  # auto-join statistics describe the model's own (first-pass) confidence
    res.curve.append((0, pages_complete()))
    for _ in range(max_rounds):
        while (a := eng.next_question()) is not None:
            truth = truth_right[a]
            # The true partner may be held by a wrong join (or linking would close a loop through
            # one): the reviewer sees it misplaced and rejects the wrong joins in the way. The
            # rebuild can auto-join the partner elsewhere again, hence the loop.
            while eng.session.right[a] is None and not eng.session.can_link(a, truth):
                blocking = _blocking_wrong_links(eng.session, a, truth, truth_right)
                if not blocking:
                    raise RuntimeError(f"strip {a}: true partner blocked by correct links only")
                res.rejections += len(blocking)
                res.per_page[page_of[a]] += len(blocking)
                eng.reject(blocking)
            if eng.session.right[a] is not None:  # the rebuild re-decided this edge
                continue
            sess = eng.session
            order = [b for b, _ in eng.suggestions(a, sess.n + 1)]
            res.asked += 1
            shown = order[: k + 1]
            cost, searched = 1, False
            if rng.random() < error_rate:  # reviewer slip: picks the top suggestion that is wrong
                wrong = [c for c in shown if c != truth and sess.can_link(a, c)]
                answer = wrong[0] if wrong else truth
            elif truth in shown:
                answer = truth
                res.in_top_k += 1
            elif truth in order[: deep_k + 1]:
                answer, cost = truth, 2  # "more" + pick from the deeper list
            else:
                answer, searched = truth, True
                res.searches += 1  # found by searching the pile by hand
            res.clicks += cost
            res.per_page[page_of[a]] += cost + searched
            before = len(eng.auto)
            eng.answer(a, answer)
            if res.rejections == 0:
                count_auto(before)
            if res.asked % curve_every == 0:
                res.curve.append((res.clicks + res.searches + res.rejections, pages_complete()))
        # page check: wrong joins are visible on the assembled pages; reject each (1 decision)
        sess = eng.session
        wrong = [(a, sess.right[a]) for a in range(sess.n)
                 if sess.right[a] is not None and sess.right[a] != truth_right[a]]
        if not wrong:
            break
        res.rejections += len(wrong)
        for a, _ in wrong:
            res.per_page[page_of[a]] += 1
        eng.reject(wrong)  # rebuilds; auto_links/auto_wrong keep describing the first pass
    res.curve.append((res.clicks + res.searches + res.rejections, pages_complete()))
    return res


TEMPERATURES = tuple(2.0 ** k for k in range(-4, 6))  # 1/16 … 32


def decisions_to_100(S: np.ndarray, null: float, truth_right: list[int], page_of: list, pages: int,
                     temperatures=TEMPERATURES, auto_threshold: float = 0.9999, **kw) -> dict:
    """Reviewer decisions per document to reach 100%, at the model's best calibration.

    Raw confidences can be over- or under-confident; both distort the review cost (too
    confident: wrong auto-joins; not confident enough: no automation). Scaling every score
    (and the "no neighbour" score) by a temperature fixes calibration without changing the
    ranking, so the cost is reported for the best temperature in `temperatures` (the
    operating point one would deploy). Chosen on the same documents, hence slightly
    optimistic, equally so for every model. (Likelihood-fitted temperature scaling was
    tried: it lands far more cautious, ~3x the decisions, so it is not the operating point.)"""
    best = None
    for t in temperatures:
        res = simulate(S * t, null * t, truth_right, page_of, auto_threshold=auto_threshold, **kw)
        r = res.summary(pages) | {"temperature": t, "per_doc": {str(k): v for k, v in res.per_page.items()}}
        if best is None or r["reviewer_decisions"] < best["reviewer_decisions"]:
            best = r
    return best
