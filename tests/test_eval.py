"""Unit tests for the shared evaluation helpers (eval_common.py)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eval_common import (auc_rank, full_rank_recall, precision_at_k,
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
