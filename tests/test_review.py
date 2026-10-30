import numpy as np

from unshredder.review import END, simulate


def _pile(pages=4, strips=6):
    truth, page_of = [], []
    for p in range(pages):
        base = p * strips
        for i in range(strips):
            truth.append(base + i + 1 if i + 1 < strips else END)
            page_of.append(p)
    return truth, page_of


def _scores(truth, noise, scale=1.0, seed=0):
    """True neighbour gets +4, everything else 0, plus gaussian noise; times `scale`."""
    n = len(truth)
    S = np.random.default_rng(seed).normal(0, noise, (n, n))
    for a, b in enumerate(truth):
        if b != END:
            S[a, b] += 4.0
    return S * scale


def test_perfect_scores_need_no_reviewer():
    from unshredder.review import decisions_to_100

    truth, page_of = _pile()
    r = decisions_to_100(_scores(truth, noise=0.0), 2.0, truth, page_of, pages=4)
    assert r["reviewer_decisions"] == 0 and r["pages_complete_final"] == 1.0


def test_useless_scores_still_reach_100_percent_with_oracle():
    truth, page_of = _pile()
    n = len(truth)
    S = np.random.default_rng(0).normal(size=(n, n))
    r = simulate(S, null=0.0, truth_right=truth, page_of=page_of, auto_threshold=0.999).summary(4)
    assert r["pages_complete_final"] == 1.0
    assert r["reviewer_decisions"] >= n * 0.5  # nearly every edge needed a person


def test_noisier_scores_cost_more():
    from unshredder.review import decisions_to_100

    truth, page_of = _pile(pages=8, strips=10)
    costs = []
    for noise in (0.3, 1.0, 2.0, 4.0):
        runs = [decisions_to_100(_scores(truth, noise, seed=s), 2.0, truth, page_of, pages=8)["reviewer_decisions"]
                for s in range(3)]
        costs.append(np.mean(runs))
    assert all(a < b for a, b in zip(costs, costs[1:])), costs


def test_metric_ignores_raw_overconfidence():
    """Same ranking, scores scaled up 2x (overconfident): best-calibration cost is unchanged
    (the optimum stays inside the temperature grid for both)."""
    from unshredder.review import decisions_to_100

    truth, page_of = _pile(pages=8, strips=10)
    for seed in range(3):
        S = _scores(truth, noise=1.5, seed=seed)
        a = decisions_to_100(S, 2.0, truth, page_of, pages=8)
        b = decisions_to_100(S * 2, 2.0 * 2, truth, page_of, pages=8)
        assert a["pages_complete_final"] == b["pages_complete_final"] == 1.0
        assert a["reviewer_decisions"] == b["reviewer_decisions"], (a, b)


def test_engine_answer_cascade_and_undo():
    from unshredder.review import ReviewEngine

    truth, _ = _pile(pages=2, strips=4)
    n = len(truth)
    S = np.full((n, n), -5.0)
    for a, b in enumerate(truth):
        if b != END:
            S[a, b] = 1.0  # right answer favoured, but not enough to auto-commit
    eng = ReviewEngine(S, null=0.0, auto_threshold=0.9999)
    assert eng.progress()["auto_links"] == 0
    a = eng.next_question()
    best = eng.suggestions(a, 5)[0][0]
    eng.answer(a, best)
    assert eng.progress()["your_decisions"] == 1
    assert eng.undo() and eng.progress()["your_decisions"] == 0
    assert eng.session.right[a] is None


def test_answer_beyond_shortlist_is_allowed():
    from unshredder.review import ReviewEngine

    n = 80
    S = np.random.default_rng(1).normal(size=(n, n))
    eng = ReviewEngine(S, null=5.0, auto_threshold=0.99999, candidates=10)
    a = eng.next_question()
    outside = next(b for b in range(n) if b != a and b not in eng.session.cand[a])
    eng.answer(a, outside)  # a reviewer found it by searching the pile
    assert eng.session.right[a] == outside


def test_true_partner_taken_by_wrong_join_is_recovered():
    """Overconfident wrong scores make auto-joins steal true partners; the oracle must still
    finish at 100% by rejecting the joins in the way."""
    from unshredder.review import decisions_to_100

    truth, page_of = _pile(pages=6, strips=8)
    n = len(truth)
    rng = np.random.default_rng(3)
    S = _scores(truth, noise=2.5, seed=3)
    for a in range(n):  # plant confident wrong joins
        if rng.random() < 0.3:
            S[a, rng.integers(0, n)] += 12
    r = decisions_to_100(S, 2.0, truth, page_of, pages=6)
    assert r["pages_complete_final"] == 1.0


def test_later_questions_come_last():
    from unshredder.review import ReviewEngine

    truth, page_of = _pile(pages=2, strips=6)
    S = _scores(truth, noise=3.0, seed=1)
    eng = ReviewEngine(S, 2.0, auto_threshold=0.999999)
    first = eng.next_question()
    assert eng.next_question([first]) != first
    others = [a for a in eng.session.open_edges() if a != first]
    assert eng.next_question(others + [first]) == others[0]  # all held: oldest first
    assert eng.next_question([first] + others) == first
