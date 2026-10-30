import numpy as np
import torch

from unshredder.seamnet import SparseScores, pair_scores, sparse_pair_scores


def _emb(n, d=8, h=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, h, generator=g), torch.randn(n, d, h, generator=g)


def _dense(R, L, shift, scale):
    S = pair_scores(R, L, shift, torch.tensor(scale)).numpy().astype(np.float64)
    np.fill_diagonal(S, -np.inf)
    return S


def _lse(M, axis):
    m = M.max(axis)
    return m + np.log(np.exp(M - np.expand_dims(m, axis)).sum(axis))


def test_blockwise_topk_matches_dense_exactly():
    R, L = _emb(70)
    S = _dense(R, L, 2, 0.3)
    k = 9
    for block, devs in [(16, None), (70, None), (13, [torch.device("cpu")] * 3)]:  # uneven blocks, "3 devices"
        ri, rv, ci, cv, rl, cl = sparse_pair_scores(R, L, 2, 0.3, k, devices=devs, block=block)
        for a in range(70):
            assert set(ri[a]) == set(np.argsort(-S[a])[:k])
            assert np.allclose(rv[a], np.sort(S[a])[::-1][:k], atol=1e-4)
            assert set(ci[a]) == set(np.argsort(-S[:, a])[:k])
            assert np.allclose(cv[a], np.sort(S[:, a])[::-1][:k], atol=1e-4)
        assert np.allclose(rl, _lse(S, 1), atol=1e-4) and np.allclose(cl, _lse(S, 0), atol=1e-4)


def test_from_dense_matches_blockwise():
    R, L = _emb(40, seed=1)
    S = _dense(R, L, 1, 1.0)
    a = SparseScores.from_dense(S, 3.0, 7)
    b = SparseScores(*sparse_pair_scores(R, L, 1, 1.0, 7, block=11), null=3.0)
    assert (a.right_idx == b.right_idx).all() and (a.left_idx == b.left_idx).all()
    assert np.allclose(a.row_lse, b.row_lse, atol=1e-4) and np.allclose(a.col_lse, b.col_lse, atol=1e-4)
