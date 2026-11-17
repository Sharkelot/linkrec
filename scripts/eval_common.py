"""Shared link-prediction evaluation helpers (exact rank-based metrics).

All metrics are device-agnostic; pass CUDA tensors for speed.
"""
import torch


def sampled_mrr_auc(pos_scores, neg_scores):
    """MRR and AUC from per-edge positive scores vs K negative scores."""
    K = neg_scores.size(1)
    rank = (neg_scores > pos_scores.unsqueeze(1)).sum(dim=1)
    mrr = (1.0 / (rank + 1)).mean().item()
    auc = ((neg_scores < pos_scores.unsqueeze(1)).sum(dim=1)
           + 0.5 * (neg_scores == pos_scores.unsqueeze(1)).sum(dim=1)).mean().item() / K
    return mrr, auc


def full_rank_recall(sim, cand_mask, edges, ks):
    """Recall@k by full ranking: for each edge (u,v), is v in u's top-k
    rankable candidates? sim: N×N scores; cand_mask: N×N bool (True =
    rankable); edges: (2,E) long.
    Exact ties are broken deterministically by candidate index (smaller
    first) so a zero-scored edge does not trivially rank at 1."""
    rows = sim[edges[0]].masked_fill(~cand_mask[edges[0]], float("-inf"))
    pos = sim[edges[0], edges[1]]
    n = sim.size(0)
    tiebreak = (torch.arange(n, device=sim.device, dtype=sim.dtype).unsqueeze(0) * 1e-9)
    # tie-break both sides: candidate w beats v iff (score_w + idx_w) > (score_v + idx_v)
    rank = (rows + tiebreak > (pos + tiebreak.squeeze(0)[edges[1]]).unsqueeze(1)).sum(dim=1)  # 0-based
    return {f"recall@{k}": float((rank < k).float().mean().item()) for k in ks}


def auc_rank(pos_scores, neg_scores):
    """Exact Mann-Whitney AUC via average ranks (tie-corrected)."""
    pos = pos_scores.float().cpu()
    neg = neg_scores.float().cpu()
    all_s = torch.cat([pos, neg])
    s, order = torch.sort(all_s)
    vals, inv = torch.unique(s, return_inverse=True)
    ranks = torch.arange(1, s.numel() + 1, dtype=torch.float)
    cnt = torch.bincount(inv)
    sum_r = torch.zeros(vals.numel())
    sum_r.scatter_add_(0, inv, ranks)
    avg_r = sum_r / cnt
    r_sorted = avg_r[inv]
    r_orig = torch.empty_like(r_sorted)
    r_orig[order] = r_sorted
    n_pos, n_neg = pos.numel(), neg.numel()
    r_pos_sum = r_orig[:n_pos].sum().item()
    return (r_pos_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def precision_at_k(sim, cand_mask, latent_mask, ks):
    """Rank all candidate undirected pairs; Precision@k = fraction of the
    top-k that are true (latent) edges. sim: N×N; masks: N×N bool."""
    tri = torch.triu_indices(sim.size(0), sim.size(0), offset=1, device=sim.device)
    scores = sim[tri[0], tri[1]].masked_fill(~cand_mask[tri[0], tri[1]], float("-inf"))
    order = torch.argsort(scores, descending=True)
    lat = latent_mask[tri[0], tri[1]]
    return {f"precision@{k}": float(lat[order[:k]].float().mean().item()) for k in ks}


def sample_neg_excluding(src, dst, adj, K, gen, N, rounds=2):
    """K uniform negatives per (src, dst) that are NOT src's adj-neighbors,
    not src itself, and not the positive dst. Vectorized; `rounds` of
    2K-candidate draws cover sparse-to-moderate neighbor fractions
    (Cora ~2%, B2B ~20%); a final uniform top-up covers pathological rows.
    src/dst: (m,) long; adj: N×N bool -> (m, K) long."""
    m = src.numel()
    dev = src.device
    out = None
    still_bad = None
    for r in range(rounds):
        cand = torch.randint(0, N, (m, 2 * K), device=dev, generator=gen)
        ok = ~adj[src.unsqueeze(1).expand(m, 2 * K), cand]
        ok &= (cand != dst.unsqueeze(1)) & (cand != src.unsqueeze(1))
        sel = torch.argsort(~ok, dim=1)[:, :K]  # ok (False) sorts first
        chosen = cand.gather(1, sel)
        good = ok.gather(1, sel)
        if r == 0:
            out = chosen
            still_bad = ~good
        else:
            out[still_bad] = chosen[still_bad]
            still_bad = still_bad & ~good
    if still_bad.any():
        out[still_bad] = torch.randint(0, N, (int(still_bad.sum().item()),),
                                      device=dev, generator=gen)
    return out
