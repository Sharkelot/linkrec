"""Shared link-prediction evaluation helpers (exact rank-based metrics) and
negative samplers with hard validity invariants.

All metrics are device-agnostic; pass CUDA tensors for speed.

Negative-sampling invariants (all samplers in this module):
  * a returned candidate is NEVER src itself, the positive dst, or a
    train-graph neighbor of src (no false negatives from the observed graph);
  * the samplers never use an unrestricted uniform fallback: when stochastic
    rounds fail to find K valid candidates, an explicit enumeration pass
    scans a random permutation of all nodes and keeps only valid ones;
  * a RuntimeError is raised if a row has fewer than K valid nodes in total
    (graph too dense) instead of silently returning an invalid candidate.
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
    # deterministic tie-break: exact score ties (e.g. zero-scored pairs) are
    # broken by candidate index, so top-k lists are reproducible run to run
    scores = scores + torch.arange(scores.numel(), device=sim.device, dtype=scores.dtype) * 1e-9
    order = torch.argsort(scores, descending=True, stable=True)
    lat = latent_mask[tri[0], tri[1]]
    return {f"precision@{k}": float(lat[order[:k]].float().mean().item()) for k in ks}


def _explicit_valid_topup(bad_rows, src, dst, adj, K, gen, N, prefer=None):
    """Fill K valid candidates for the given rows by explicit enumeration:
    scan a random permutation of all N nodes, keep valid ones only.
    `prefer` (optional): per-row (n_c,) index tensors whose members are
    tried first (e.g. the dst's class pool); the rest of the permutation
    fills the remainder. Raises RuntimeError if a row has < K valid nodes.
    Returns (m, K) long."""
    m = bad_rows.numel()
    dev = src.device
    out = torch.empty(m, K, dtype=torch.long, device=dev)
    for i in range(m):
        i0 = int(bad_rows[i])
        u, v = int(src[i0]), int(dst[i0])
        mask = ~adj[u].clone()
        mask[v] = False
        mask[u] = False
        perm = torch.randperm(N, device=dev, generator=gen)
        if prefer is not None:
            pool = prefer[i]  # loop position, not the original row index
            pool_valid = pool[mask[pool]]  # node-based: mask indexed by node id
            rest_valid = perm[mask[perm] & ~torch.isin(perm, pool)]
            cand_t = torch.cat([pool_valid, rest_valid])
        else:
            # NOTE: mask must be indexed by NODE id, not permutation position:
            # perm[mask[perm]] = random order over exactly the valid nodes
            cand_t = perm[mask[perm]]
        if cand_t.numel() < K:
            raise RuntimeError(
                f"node {u}: only {cand_t.numel()} valid negatives < K={K} "
                f"(graph too dense for the requested exclusions)")
        out[i] = cand_t[:K]
    return out


def sample_neg_excluding(src, dst, adj, K, gen, N, rounds=8):
    """K uniform negatives per (src, dst) that are NOT src's adj-neighbors,
    not src itself, and not the positive dst. Vectorized stochastic rounds
    (only currently-bad slots are refreshed each round); any remaining rows
    are filled by explicit valid enumeration (see module docstring - never
    an unrestricted fallback).
    src/dst: (m,) long; adj: N×N bool -> (m, K) long."""
    m = src.numel()
    dev = src.device
    out = None
    slot_bad = None
    for r in range(rounds):
        cand = torch.randint(0, N, (m, 2 * K), device=dev, generator=gen)
        ok = ~adj[src.unsqueeze(1).expand(m, 2 * K), cand]
        ok &= (cand != dst.unsqueeze(1)) & (cand != src.unsqueeze(1))
        sel = torch.argsort(~ok, dim=1, stable=True)  # valid (False) sorts first; stable = deterministic tie order
        if r == 0:
            selk = sel[:, :K]
            out = cand.gather(1, selk)
            slot_bad = ~ok.gather(1, selk)
        else:
            # refresh ONLY the bad slots with this round's best candidate;
            # validity of what was just written is tracked per slot
            first = cand.gather(1, sel[:, :1]).expand(m, K)
            first_ok = ok.gather(1, sel[:, :1]).expand(m, K)
            bad = slot_bad
            out[bad] = first[bad]
            slot_bad = bad & ~first_ok
        if not slot_bad.any():
            break
    if slot_bad.any():
        bad_rows = torch.nonzero(slot_bad.any(dim=1)).squeeze(1)
        out[bad_rows] = _explicit_valid_topup(bad_rows, src, dst, adj, K, gen, N)
    return out


def sample_class_negatives(src, dst, cls_of_dst, pools, adj, K, gen, N, rounds=8):
    """K hard negatives per (src, dst): drawn from the dst's class pool
    (pools[c] = node indices of class c), excluding src, dst and src's
    adj-neighbors. Rows whose pool has no nodes fall back to uniform over
    all nodes WITH THE SAME EXCLUSIONS. Remaining rows get explicit valid
    enumeration (class pool first, then the rest).
    src/dst/cls_of_dst: (m,) long; adj: N×N bool -> (m, K) long."""
    m = src.numel()
    dev = src.device
    sizes = torch.tensor([p.numel() for p in pools], device=dev)
    cls_start = torch.zeros(len(pools) + 1, dtype=torch.long, device=dev)
    cls_start[1:] = torch.cumsum(sizes, 0)
    flat = torch.cat(pools) if any(p.numel() for p in pools) else \
        torch.empty(0, dtype=torch.long, device=dev)
    has = sizes[cls_of_dst] > 0
    out = None
    slot_bad = None
    for r in range(rounds):
        off = (torch.rand(m, 2 * K, device=dev, generator=gen)
               * sizes[cls_of_dst].clamp(min=1).unsqueeze(1)).long()
        if flat.numel() == 0:
            pool_cand = torch.zeros(m, 2 * K, dtype=torch.long, device=dev)
        else:
            # clamp start for empty pools: the index must stay in-range
            # even though torch.where discards that branch
            safe_start = torch.clamp(cls_start[cls_of_dst],
                                     max=flat.numel() - 1)
            pool_cand = flat[safe_start.unsqueeze(1) + off]
        cand = torch.where(has.unsqueeze(1), pool_cand,
                           torch.randint(0, N, (m, 2 * K), device=dev, generator=gen))
        ok = ~adj[src.unsqueeze(1).expand(m, 2 * K), cand]
        ok &= (cand != dst.unsqueeze(1)) & (cand != src.unsqueeze(1))
        sel = torch.argsort(~ok, dim=1, stable=True)  # valid (False) sorts first; stable = deterministic tie order
        if r == 0:
            selk = sel[:, :K]
            out = cand.gather(1, selk)
            slot_bad = ~ok.gather(1, selk)
        else:
            first = cand.gather(1, sel[:, :1]).expand(m, K)
            first_ok = ok.gather(1, sel[:, :1]).expand(m, K)
            bad = slot_bad
            out[bad] = first[bad]
            slot_bad = bad & ~first_ok
        if not slot_bad.any():
            break
    if slot_bad.any():
        bad_rows = torch.nonzero(slot_bad.any(dim=1)).squeeze(1)
        prefer = [pools[int(cls_of_dst[i0])] for i0 in bad_rows.tolist()]
        out[bad_rows] = _explicit_valid_topup(
            bad_rows, src, dst, adj, K, gen, N, prefer=prefer)
    return out
