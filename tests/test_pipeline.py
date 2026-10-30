import numpy as np
import pytest

from unshredder import config, solve
from unshredder.evaluate import evaluate


def _truth(sizes):
    return {f"d{d}_{i}": {"doc": d, "index": i, "flipped": False} for d, n in enumerate(sizes) for i in range(n)}


def _chain(names, flipped=False):
    return [{"name": n, "flipped": flipped} for n in names]


def test_evaluate_perfect():
    truth = _truth([3, 2])
    m = evaluate([_chain(["d0_0", "d0_1", "d0_2"]), _chain(["d1_0", "d1_1"])], truth)
    assert m["accuracy"] == 1.0 and m["neighbor_accuracy"] == 1.0 and m["perfect_docs"] == 2


def test_evaluate_counts_document_boundary_as_correct():
    # Paixão metric: last strip of one doc followed by the first of another is correct.
    truth = _truth([2, 2])
    m = evaluate([_chain(["d0_0", "d0_1", "d1_0", "d1_1"])], truth)
    assert m["accuracy"] == 1.0
    assert m["neighbor_accuracy"] == 1.0
    assert m["perfect_docs"] == 0  # both docs in one chain


def test_evaluate_rotated_chain_is_upright():
    truth = {k: v | {"flipped": True} for k, v in _truth([3]).items()}
    # every strip scanned upside down and the chain read right-to-left: undoing flips fixes it
    m = evaluate([_chain(["d0_0", "d0_1", "d0_2"], flipped=True)], truth)
    assert m["neighbor_accuracy"] == 1.0


def test_evaluate_ignores_blank_strips_in_paixao_accuracy():
    truth = _truth([4])
    chain = _chain(["d0_1", "d0_2"]) + [{"name": "d0_0", "flipped": False, "blank": True}]
    m = evaluate([chain], truth)
    assert m["accuracy"] == 1.0


def test_tsp_path_recovers_order():
    rng = np.random.default_rng(0)
    n = 12
    order = rng.permutation(n)
    cost = np.full((n, n), 10.0)
    for a, b in zip(order[:-1], order[1:]):
        cost[a, b] = 0.1
    np.fill_diagonal(cost, np.inf)
    assert solve.tsp_path(cost, list(range(n))) == order.tolist()


def test_split_path_known_count_and_threshold():
    score = np.zeros((4, 4))
    score[1, 2] = 5.0
    assert solve.split_path([0, 1, 2, 3], score, 2, 0.0) == [[0, 1], [2, 3]]
    assert solve.split_path([0, 1, 2, 3], score, None, 1.0) == [[0, 1], [2, 3]]
    assert solve.split_path([0, 1, 2, 3], score, None, 10.0) == [[0, 1, 2, 3]]


def test_config_overrides():
    cfg = config.load(None, ["shred.num_strips=7", "generate.font_size=[10, 12]", "seamnet.page_dirs=[a, b]"])
    assert cfg.shred.num_strips == 7
    assert cfg.generate.font_size == (10, 12)
    assert cfg.seamnet.page_dirs == ["a", "b"]
    assert config.load(None, ["seamnet.lr=3e-4"]).seamnet.lr == 3e-4
    with pytest.raises(KeyError):
        config.load(None, ["shred.nope=1"])


def test_link_metrics_precision_recall_and_bcubed():
    from unshredder.evaluate import link_metrics

    truth = _truth([3, 2])  # true links: d0_0>d0_1, d0_1>d0_2, d1_0>d1_1
    recon = [
        [{"name": "d0_0", "flipped": False, "conf": 0.99}, {"name": "d0_1", "flipped": False, "conf": 0.2},
         {"name": "d1_0", "flipped": False, "conf": 0.9}, {"name": "d1_1", "flipped": False}],
        [{"name": "d0_2", "flipped": False}],
    ]
    m = link_metrics(recon, truth)
    assert m["true_links"] == 3 and m["output_links"] == 3
    # ranked by confidence: 0.99 correct, 0.9 correct, 0.2 wrong -> 2/3 recall at 100% precision
    assert m["recall_at_p99"] == round(2 / 3, 4)
    assert m["link_precision_all"] == round(2 / 3, 4)
    # strip d0_2 alone: precision 1, recall 1/3; chain of 4 mixes pages 2+2
    assert 0 < m["bcubed_precision"] < 1 and 0 < m["bcubed_recall"] < 1


def test_list_docs_slices(tmp_path):
    from unshredder.datasets import list_docs

    for k in range(6):
        (tmp_path / "mechanical" / f"D{k:03d}" / "strips").mkdir(parents=True)
    assert [d.name for d in list_docs(f"{tmp_path}[1:3]")] == ["D001", "D002"]
    assert len(list_docs(str(tmp_path))) == 6 and len(list_docs(f"{tmp_path}[4:]")) == 2


def test_page_recovery():
    from unshredder.evaluate import page_recovery

    truth = _truth([4, 2])
    perfect = [_chain(["d0_0", "d0_1", "d0_2", "d0_3"]), _chain(["d1_0", "d1_1"])]
    assert page_recovery(perfect, truth) == {"page_recovery": 1.0, "seam_recovery": 1.0, "pages_complete": 1.0}
    # page 0 split by a foreign strip: largest intact piece is 2 of 4; page 1 intact
    broken = [_chain(["d0_0", "d0_1", "d1_0", "d1_1", "d0_2", "d0_3"])]
    # page 0: 2 of its 3 seams intact; page 1: 1 of 1
    assert page_recovery(broken, truth) == {"page_recovery": 0.75, "seam_recovery": round((2 / 3 + 1) / 2, 4),
                                            "pages_complete": 0.5}


def test_page_recovery_inked_ignores_blank_strips():
    from unshredder.evaluate import page_recovery_inked

    truth = _truth([4])  # d0_0 will be a blank margin strip
    recon = [[{"name": "d0_0", "flipped": False, "blank": True}],
             [{"name": "d0_1", "flipped": False}, {"name": "d0_2", "flipped": False}, {"name": "d0_3", "flipped": False}]]
    assert page_recovery_inked(recon, truth)["page_recovery"] == 1.0
