"""Unit tests for the shared evaluation helpers (eval_common.py)."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eval_common import (auc_rank, full_rank_recall, precision_at_k,
                         sample_class_negatives, sample_neg_excluding,
                         sampled_mrr_auc)  # noqa: E402


def test_auc_rank_perfect_separation():
    pos = torch.tensor([0.9, 0.8, 0.7])
    neg = torch.tensor([0.1, 0.2, 0.3])
    assert auc_rank(pos, neg) == 1.0


def test_auc_rank_worst_separation():
    pos = torch.tensor([0.1, 0.2])
    neg = torch.tensor([0.8, 0.9, 0.95])
    assert auc_rank(pos, neg) == 0.0


def test_auc_rank_all_ties_is_half():
    assert auc_rank(torch.zeros(4), torch.zeros(10)) == 0.5


def test_auc_rank_tie_correction():
    # pos values {1, 1}, neg values {1, 0} -> the three 1s tie at ranks 2,3,4
    # (avg 3.0 each); the 0 gets rank 1.
    # AUC = (3+3 - 2*3/2) / (2*2) = 3/4
    assert abs(auc_rank(torch.tensor([1.0, 1.0]), torch.tensor([1.0, 0.0])) - 0.75) < 1e-9


def test_sampled_mrr_auc_known_case():
    # one positive, two negatives below it -> MRR 1.0, AUC 1.0
    pos = torch.tensor([0.5])
    neg = torch.tensor([[0.1, 0.2]])
    mrr, auc = sampled_mrr_auc(pos, neg)
    assert mrr == 1.0 and auc == 1.0
    # negative above positive -> rank 2 -> MRR 0.5, AUC 0.5
    mrr, auc = sampled_mrr_auc(pos, torch.tensor([[0.9, 0.2]]))
    assert abs(mrr - 0.5) < 1e-9 and abs(auc - 0.5) < 1e-9


def test_full_rank_recall_all_zero_tiebreak_is_deterministic():
    # all-zero scores: deterministic index tie-break, NOT trivial rank 1.
    # with a zero pos, candidate w beats v iff idx_w > idx_v.
    N = 10
    sim = torch.zeros(N, N)
    cand = ~torch.eye(N, dtype=torch.bool)
    # v=1: candidates 2..9 (8 of them) all tie-break above it -> rank 8
    r = full_rank_recall(sim, cand, torch.tensor([[0], [1]]), (1, 9))
    assert r["recall@1"] == 0.0 and r["recall@9"] == 1.0
    # v=9 (largest index): nothing ties above it -> rank 0
    r2 = full_rank_recall(sim, cand, torch.tensor([[0], [9]]), (1, 9))
    assert r2["recall@1"] == 1.0 and r2["recall@9"] == 1.0


def test_full_rank_recall_basic():
    N = 5
    sim = torch.tensor([
        [0.0, 0.9, 0.1, 0.05, 0.02],
        [0.9, 0.0, 0.1, 0.05, 0.02],
        [0.1, 0.1, 0.0, 0.99, 0.98],
        [0.05, 0.05, 0.99, 0.0, 0.01],
        [0.02, 0.02, 0.98, 0.01, 0.0],
    ])
    cand = ~torch.eye(5, dtype=torch.bool)
    # for src=0: ranking of others: 1(0.9), 2(0.1), 3(0.05), 4(0.02)
    r = full_rank_recall(sim, cand, torch.tensor([[0], [1]]), (1, 2))
    assert r["recall@1"] == 1.0
    r = full_rank_recall(sim, cand, torch.tensor([[0], [3]]), (2, 3))
    assert r["recall@2"] == 0.0 and r["recall@3"] == 1.0


def test_precision_at_k_perfect():
    N = 6
    sim = torch.zeros(N, N)
    cand = ~torch.eye(N, dtype=torch.bool)
    latent = torch.zeros(N, N, dtype=torch.bool)
    latent[0, 1] = latent[1, 0] = True
    latent[2, 3] = latent[3, 2] = True
    sim[0, 1] = sim[1, 0] = 0.99
    sim[2, 3] = sim[3, 2] = 0.98
    p = precision_at_k(sim, cand, latent, (1, 2, 3))
    assert p["precision@1"] == 1.0 and p["precision@2"] == 1.0
    assert abs(p["precision@3"] - 2 / 3) < 1e-6


# ------------------------------------------------------------------ samplers
def _assert_negatives_valid(out, src, dst, adj, K):
    """Core invariant: no candidate is src itself, the positive dst, or a
    train-graph neighbor of src."""
    assert out.shape == (src.numel(), K)
    assert bool((out != src.unsqueeze(1)).all()), "self-pair in negatives"
    assert bool((out != dst.unsqueeze(1)).all()), "positive dst in negatives"
    assert bool((~adj[src.unsqueeze(1).expand_as(out), out]).all()), \
        "train neighbor in negatives (false negative)"


def test_sample_neg_excluding_invariants_sparse_graph():
    N, m, K = 200, 500, 10
    g = torch.Generator().manual_seed(0)
    adj = (torch.rand(N, N, generator=g) < 0.15).to(torch.bool)
    adj = adj | adj.T
    src = torch.randint(0, N, (m,), generator=g)
    dst = torch.randint(0, N, (m,), generator=g)
    out = sample_neg_excluding(src, dst, adj, K, g, N)
    _assert_negatives_valid(out, src, dst, adj, K)


def test_sample_neg_excluding_invariants_dense_graph():
    """Dense graph (60% edges): stochastic rounds often fail, so this
    exercises the explicit-valid top-up path (never an unrestricted
    randint fallback)."""
    N, m, K = 200, 500, 10
    g = torch.Generator().manual_seed(1)
    adj = (torch.rand(N, N, generator=g) < 0.6).to(torch.bool)
    adj = adj | adj.T
    src = torch.randint(0, N, (m,), generator=g)
    dst = torch.randint(0, N, (m,), generator=g)
    out = sample_neg_excluding(src, dst, adj, K, g, N, rounds=2)
    _assert_negatives_valid(out, src, dst, adj, K)


def test_sample_neg_excluding_raises_when_too_few_valid():
    """Complete graph: zero valid nodes for any row -> must raise, not
    silently return an invalid candidate."""
    N = 12
    adj = torch.ones(N, N, dtype=torch.bool)
    src = torch.tensor([0])
    dst = torch.tensor([1])
    with pytest.raises(RuntimeError):
        sample_neg_excluding(src, dst, adj, 5, torch.Generator(), N, rounds=2)


def test_sample_class_negatives_invariants():
    """Hard negatives: exclusions ALWAYS hold; when the class pool has
    enough valid members, every candidate comes from the dst's class."""
    N, m, K = 300, 400, 10
    g = torch.Generator().manual_seed(2)
    cls = torch.randint(0, 3, (N,), generator=g)
    pools = [torch.nonzero(cls == c).squeeze(1).contiguous() for c in range(3)]
    adj = (torch.rand(N, N, generator=g) < 0.2).to(torch.bool)
    adj = adj | adj.T
    src = torch.randint(0, N, (m,), generator=g)
    dst = torch.randint(0, N, (m,), generator=g)
    out = sample_class_negatives(src, dst, cls[dst], pools, adj, K, g, N)
    _assert_negatives_valid(out, src, dst, adj, K)
    # class membership where the pool has >= K+2 valid members
    in_cls = torch.zeros(N, N, dtype=torch.bool)
    for c, p in enumerate(pools):
        in_cls[:, c] = False
        in_cls[p, c] = True
    dst_cls = cls[dst]
    pool_valid_count = torch.zeros(m, dtype=torch.long)
    for i in range(m):
        u, v = int(src[i]), int(dst[i])
        mask = ~adj[u]
        mask[v] = False
        mask[u] = False
        pool = pools[int(dst_cls[i])]
        pool_valid_count[i] = int(mask[pool].sum().item())
    enough = pool_valid_count >= K + 2
    for i in torch.nonzero(enough).squeeze(1).tolist():
        for c in out[i].tolist():
            assert bool(in_cls[c, dst_cls[i]]), "candidate outside dst class"


def test_sample_class_negatives_empty_pool_fallback_still_valid():
    """Rows whose class pool is empty fall back to uniform-over-all WITH
    the same exclusions (never an unrestricted draw)."""
    N, K = 100, 10
    g = torch.Generator().manual_seed(3)
    # two classes, class 1 empty
    cls = torch.zeros(N, dtype=torch.long)
    pools = [torch.arange(1, N), torch.empty(0, dtype=torch.long)]
    adj = (torch.rand(N, N, generator=g) < 0.3).to(torch.bool)
    adj = adj | adj.T
    src = torch.randint(0, N, (200,), generator=g)
    dst = torch.ones(200, dtype=torch.long)  # dst in empty class 1
    out = sample_class_negatives(src, dst, cls[dst], pools, adj, K, g, N)
    _assert_negatives_valid(out, src, dst, adj, K)


def test_topup_path_forced_by_high_neighbor_fraction():
    """85% neighbor fraction: stochastic rounds almost never fill K valid
    candidates, so nearly every row goes through the explicit-valid
    enumeration top-up. Invariants must still hold."""
    N, m, K = 120, 300, 8
    g = torch.Generator().manual_seed(4)
    adj = (torch.rand(N, N, generator=g) < 0.425).to(torch.bool)
    adj = adj | adj.T
    src = torch.randint(0, N, (m,), generator=g)
    dst = torch.randint(0, N, (m,), generator=g)
    out = sample_neg_excluding(src, dst, adj, K, g, N, rounds=3)
    _assert_negatives_valid(out, src, dst, adj, K)
