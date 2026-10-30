"""Score a reconstruction against the shredding ground truth."""

from __future__ import annotations

from collections import Counter

import numpy as np


def evaluate(reconstruction: list[list[dict]], truth: dict[str, dict]) -> dict:
    """
    accuracy:          Paixão et al.'s metric, over non-blank strips with the chains laid
                       end to end: fraction of adjacent pairs that are either true
                       neighbours or a document's last strip followed by another's first.
    neighbor_accuracy: fraction of true left-right neighbour pairs recovered (all strips).
    perfect_docs:      documents whose non-blank strips form one chain in exact order.
    purity:            fraction of strips sitting in a chain dominated by their own document.
    """
    doc_sizes = Counter(t["doc"] for t in truth.values())
    true_pairs = sum(n - 1 for n in doc_sizes.values())

    correct = pure = 0
    oriented = []
    for chain in reconstruction:
        # A strip ends up upright if our predicted flip undoes its scan flip.
        placed = [(truth[e["name"]], truth[e["name"]]["flipped"] == e["flipped"], e.get("blank", False)) for e in chain]
        if sum(u for _, u, _ in placed) * 2 < len(placed):  # whole chain upside down: view rotated
            placed = [(t, not u, b) for t, u, b in reversed(placed)]
        oriented.append(placed)
        for (ta, ua, _), (tb, ub, _) in zip(placed, placed[1:]):
            if ua and ub and ta["doc"] == tb["doc"] and tb["index"] == ta["index"] + 1:
                correct += 1
        pure += Counter(t["doc"] for t, _, _ in placed).most_common(1)[0][1]

    # Paixão accuracy and perfect docs ignore blank strips, which carry no information.
    seq = [(t, u) for placed in oriented for t, u, b in placed if not b]
    content: dict[int, list[int]] = {}
    for t, _ in seq:
        content.setdefault(t["doc"], []).append(t["index"])
    first = {d: min(v) for d, v in content.items()}
    last = {d: max(v) for d, v in content.items()}
    order = {d: sorted(v) for d, v in content.items()}

    def ok(a, b):
        (ta, ua), (tb, ub) = a, b
        if not (ua and ub):
            return False
        if ta["doc"] == tb["doc"]:
            pos = order[ta["doc"]]
            return pos.index(tb["index"]) == pos.index(ta["index"]) + 1
        return ta["index"] == last[ta["doc"]] and tb["index"] == first[tb["doc"]]

    paixao = sum(ok(a, b) for a, b in zip(seq, seq[1:])) / max(1, len(seq) - 1)

    perfect = 0
    for placed in oriented:
        items = [(t, u) for t, u, b in placed if not b]
        docs = {t["doc"] for t, _ in items}
        if len(docs) == 1:
            d = docs.pop()
            if all(u for _, u in items) and [t["index"] for t, _ in items] == order[d]:
                perfect += 1

    return {
        "num_strips": len(truth),
        "docs_true": len(doc_sizes),
        "docs_found": len(reconstruction),
        "accuracy": round(paixao, 4),
        "neighbor_accuracy": round(correct / true_pairs, 4) if true_pairs else 1.0,
        "perfect_docs": perfect,
        "perfect_doc_rate": round(perfect / len(doc_sizes), 4),
        "purity": round(pure / len(truth), 4),
    }


def link_metrics(reconstruction: list[list[dict]], truth: dict[str, dict]) -> dict:
    """Metrics for big piles where almost every pair is a non-match (accuracy over pairs
    would be meaningless). Each output link carries the model's confidence ("conf");
    committing only links above a threshold trades recall for precision.

    recall_at_p99:  fraction of all true links recovered while >= 99% of committed links
                    are correct (the headline number: how much is automatic and safe)
    bcubed_*:       strip -> page grouping quality (chains vs true pages)
    """
    by_name = truth
    index = {(t["doc"], t["index"]): name for name, t in by_name.items()}
    true_links = {(index[k], index[(k[0], k[1] + 1)]) for k in index if (k[0], k[1] + 1) in index}

    scored = []
    for chain in reconstruction:
        for e, nxt in zip(chain, chain[1:]):
            if "conf" in e:
                scored.append((e["conf"], (e["name"], nxt["name"]) in true_links))
    out = {"true_links": len(true_links), "output_links": len(scored)}
    if scored:
        scored.sort(key=lambda x: -x[0])
        correct = np.cumsum([c for _, c in scored])
        k = np.arange(1, len(scored) + 1)
        precision, recall = correct / k, correct / max(1, len(true_links))
        for target in (0.99, 0.95):
            ok = precision >= target
            out[f"recall_at_p{int(target * 100)}"] = round(float(recall[ok].max()) if ok.any() else 0.0, 4)
        out["link_precision_all"] = round(float(precision[-1]), 4)
        out["link_recall_all"] = round(float(recall[-1]), 4)
        hits = np.array([c for _, c in scored], dtype=float)
        out["average_precision"] = round(float((precision * hits).sum() / max(1, len(true_links))), 4)

    # B-cubed: per strip, how pure its chain is (precision) and how complete (recall).
    chain_of = {e["name"]: i for i, ch in enumerate(reconstruction) for e in ch}
    chain_pages: dict[int, Counter] = {}
    for name, i in chain_of.items():
        chain_pages.setdefault(i, Counter())[by_name[name]["doc"]] += 1
    page_size = Counter(t["doc"] for t in by_name.values())
    chain_size = {i: sum(c.values()) for i, c in chain_pages.items()}
    p = r = 0.0
    for name, i in chain_of.items():
        same = chain_pages[i][by_name[name]["doc"]]
        p += same / chain_size[i]
        r += same / page_size[by_name[name]["doc"]]
    n = max(1, len(chain_of))
    out["bcubed_precision"], out["bcubed_recall"] = round(p / n, 4), round(r / n, 4)
    return out


def page_recovery_inked(reconstruction: list[list[dict]], truth: dict[str, dict]) -> dict:
    """page_recovery with strips flagged blank removed and pages re-indexed without them,
    so bags are scored exactly like the validation pile."""
    keep = [[e for e in ch if not e.get("blank")] for ch in reconstruction]
    keep = [ch for ch in keep if ch]
    names = {e["name"] for ch in keep for e in ch}
    by_page: dict = {}
    for name in names:
        by_page.setdefault(truth[name]["doc"], []).append(name)
    t = {}
    for page, members in by_page.items():
        for r, name in enumerate(sorted(members, key=lambda n: truth[n]["index"])):
            t[name] = {"doc": page, "index": r, "flipped": truth[name]["flipped"]}
    return page_recovery(keep, t)


def page_recovery(reconstruction: list[list[dict]], truth: dict[str, dict], per_page: bool = False) -> dict:
    """End-user metric: for each original page, the largest correctly assembled piece
    (consecutive strips of that page, in order, upright, nothing foreign in between) as
    a fraction of the page's strips, averaged over pages.

    page_recovery:      mean fraction of each page recovered intact (1.0 = all perfect)
    seam_recovery:      mean fraction of each page's seams reconstructed correctly (lenient:
                        one wrong seam costs one seam, not half the page)
    pages_complete:     fraction of pages recovered 100%
    """
    page_size = Counter(t["doc"] for t in truth.values())
    best = Counter()
    good_seams = Counter()
    for chain in reconstruction:
        run_page, run_len, prev = None, 0, None
        for e in chain:
            t = truth[e["name"]]
            upright = t["flipped"] == e["flipped"]
            if (upright and prev is not None and t["doc"] == run_page
                    and t["index"] == prev["index"] + 1):
                run_len += 1
                good_seams[t["doc"]] += 1
            else:
                run_page, run_len = (t["doc"], 1) if upright else (None, 0)
            prev = t if upright else None
            if run_page is not None:
                best[run_page] = max(best[run_page], run_len)
    fractions = [best[p] / n for p, n in page_size.items()]
    seams = [good_seams[p] / (n - 1) for p, n in page_size.items() if n > 1]
    out = {"page_recovery": round(float(np.mean(fractions)), 4),
           "seam_recovery": round(float(np.mean(seams)), 4) if seams else 1.0,
           "pages_complete": round(float(np.mean([f == 1.0 for f in fractions])), 4)}
    if per_page:  # fraction recovered intact, per original page
        out["per_page"] = {p: round(best[p] / n, 4) for p, n in page_size.items()}
    return out
