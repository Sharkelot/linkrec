#!/usr/bin/env python3
"""Train + evaluate link prediction on the synthetic B2B graph (ground truth).

Models compared (identical protocol, identical splits):
  1. feature_cosine  - cosine of OBSERVED node features (no graph)
  2. adamic_adaar    - Adamic-Adar: sum over common neighbors c of
                       1/log(deg(c)) (weights the COMMON neighbors, not
                       the endpoints), on the train graph
  3. sage_cosine     - GraphSAGE embeddings, cosine score
  4. pair_ranker     - MLP over relationship-specific pair features:
                       [sage_cosine, sector_match, region_match, size_diff,
                        log1p(deg_u), log1p(deg_v), log1p(common_neighbors)]

Metrics (ground truth = held-out latent edges):
  * AUC (latent vs true non-edge, exact rank-based over ALL ~4.5M candidates)
  * MRR / AUC with uniform and hard (same-sector) sampled negatives
  * Recall@k by full ranking (k = 1,5,10,20)
  * Precision@k over the TOP-k unconnected pairs (k = 1,5,10,20,50,100)
    <- the headline metric: fraction of the model's top "likely useful
       unconnected relationships" that truly are latent edges

Outputs:
  outputs/b2b_metrics.json, b2b_model.pt, b2b_ranker.pt, b2b_top_pairs.json
"""
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from eval_common import (auc_rank, full_rank_recall, precision_at_k,
                         sampled_mrr_auc, sample_class_negatives,
                         sample_neg_excluding)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
SEED = int(os.environ.get("LINKREC_SEED", "42"))
HIDDEN = 128
EPOCHS = 120
LR = 1e-3
NEG_K = 10
EVAL_NEGS = 99
RECALL_KS = (1, 5, 10, 20)
PREC_KS = (1, 5, 10, 20, 50, 100)
TOP_K = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SAGE(torch.nn.Module):
    def __init__(self, in_dim, hidden, out):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.head = torch.nn.Linear(hidden, out)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.head(x)


class PairRanker(torch.nn.Module):
    def __init__(self, n_feat=7, hidden=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_feat, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, feats):
        return self.net(feats).squeeze(-1)


def main():
    # bit-level reproducibility: PyG scatter on CUDA is atomic-based and
    # otherwise drifts ~1e-4 run to run; deterministic mode pins the kernels
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    rng = torch.Generator(device=DEVICE).manual_seed(SEED)
    t0 = time.time()
    meta = json.loads((OUT / "b2b_meta.json").read_text())
    G = torch.load(OUT / "b2b_graph.pt", map_location=DEVICE)
    N = G["x"].shape[0]
    x = G["x"].float()
    sector, region, size = G["sector"], G["region"], G["size"]
    train_adj = G["train_adj"]
    latent = G["latent_mask"]
    train_pairs = G["train_pairs"]
    latent_pairs = G["latent_pairs"]
    n_train = train_pairs.size(1)
    n_latent = latent_pairs.size(1)
    print(f"[b2b] N={N} train={n_train} latent={n_latent} (device={DEVICE})")

    # ---- edge-disjoint protocol (out-of-fold ranker features) ----
    # Split the observed train edges into:
    #   MP  (80%): the message-passing graph - SAGE trains here (graph,
    #              loss and negative exclusion all on MP only).
    #   SUP (20%): held out from the GNN entirely; used as the ranker's
    #              positive labels, so the ranker's SAGE-embedding / degree
    #              / common-neighbor features for a positive pair were
    #              computed WITHOUT that edge present (edge-level
    #              out-of-fold; no degree correction needed).
    perm_e = torch.randperm(n_train, device=DEVICE, generator=rng)
    n_mp = int(n_train * 0.8)
    mp_pairs = train_pairs[:, perm_e[:n_mp]]
    sup_pairs = train_pairs[:, perm_e[n_mp:]]
    mp_adj = torch.zeros(N, N, dtype=torch.bool, device=DEVICE)
    mp_adj[mp_pairs[0], mp_pairs[1]] = True
    mp_adj = mp_adj | mp_adj.T
    edge_index_mp = torch.cat([mp_pairs, mp_pairs.flip(0)], dim=1)  # both directions
    print(f"[b2b] edge-disjoint: MP={mp_pairs.size(1)} SUP={sup_pairs.size(1)}")

    sector_pools = [torch.nonzero(sector == c).squeeze(1) for c in range(meta["n_sectors"])]
    tri_i, tri_j = torch.triu_indices(N, N, offset=1, device=DEVICE)

    # candidate space for ranking: every unobserved pair (latent + non-edge)
    cand_mask = ~(train_adj | torch.eye(N, dtype=torch.bool, device=DEVICE))

    # ---------------------------------------------------------------- SAGE
    # memory-lean: one forward per epoch, per-block backward with
    # gradient accumulation (GPU is shared). Negatives sampled via the
    # hardened vectorized samplers in eval_common (exclusions + explicit
    # valid top-up, never an unrestricted fallback).
    model = SAGE(x.shape[1], HIDDEN, HIDDEN).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    margin = 1.0
    BLOCK = 8192

    n_mp = mp_pairs.size(1)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()
        emb = F.normalize(model(x, edge_index_mp), dim=-1)
        # negatives exclude ALL observed (train) neighbors / self / dst
        # (stricter than the MP graph - no false negatives); cat along
        # dim=1 so each edge keeps ITS OWN negatives
        neg_u = sample_neg_excluding(mp_pairs[0], mp_pairs[1], train_adj,
                                     NEG_K, rng, N)
        neg_h = sample_class_negatives(mp_pairs[0], mp_pairs[1],
                                       sector[mp_pairs[1]], sector_pools,
                                       train_adj, NEG_K, rng, N)
        total_loss, nb = 0.0, 0
        for b in range(0, n_mp, BLOCK):
            e = min(b + BLOCK, n_mp)
            sb, eb = mp_pairs[0, b:e], mp_pairs[1, b:e]
            m = sb.numel()
            pos = F.cosine_similarity(emb[sb], emb[eb], dim=-1)
            neg_dst = torch.cat([neg_u[b:e], neg_h[b:e]], dim=1).view(-1)
            neg = F.cosine_similarity(emb[sb.repeat_interleave(2 * NEG_K)],
                                      emb[neg_dst], dim=-1)
            neg = neg.view(m, 2, NEG_K).mean(dim=(1, 2))
            loss = F.relu(margin - pos + neg).mean()
            loss.backward(retain_graph=True)  # accumulate; emb graph kept
            total_loss += loss.item()
            nb += 1
        opt.step()
        if epoch % 20 == 0 or epoch == 1:
            print(f"[sage] epoch {epoch:3d} loss={total_loss / nb:.4f}", flush=True)

    # ---------------------------------------------------------------- scores
    with torch.no_grad():
        emb = F.normalize(model(x, edge_index_mp), dim=-1)
        sim_sage = emb @ emb.T
        sim_feat = x @ x.T  # features are already ~unit; cosine below
        sim_feat = F.normalize(x, dim=-1) @ F.normalize(x, dim=-1).T

        # Adamic-Adar on the train graph: weight the COMMON neighbors
        # aa[u,v] = sum_{c in N(u) cap N(v)} 1/log(deg(c))
        deg = train_adj.float().sum(dim=1)
        w = torch.where(deg > 1, 1.0 / torch.log(deg.clamp(min=2)), torch.zeros_like(deg))
        A = train_adj.float()
        aa = A @ torch.diag_embed(w) @ A.t()
        aa = aa.masked_fill(torch.eye(N, dtype=torch.bool, device=DEVICE), 0.0)
        aa_max = aa.max().clamp(min=1.0)
        sim_aa = aa / aa_max

    # shared sampled negatives for MRR/AUC (LATENT pairs are the positives).
    # Negatives exclude train neighbors / self / dst. NOTE: sampled from
    # unobserved-space nodes, so a latent edge of the same src CAN appear
    # among the negatives (realistic deployment condition, conservative).
    lat_pos = latent_pairs
    neg_u = sample_neg_excluding(lat_pos[0], lat_pos[1], train_adj,
                                 EVAL_NEGS, rng, N)
    neg_h = sample_class_negatives(lat_pos[0], lat_pos[1], sector[lat_pos[1]],
                                   sector_pools, train_adj, EVAL_NEGS, rng, N)
    neg_src = lat_pos[0].repeat_interleave(2 * EVAL_NEGS)

    def evaluate(name, sim):
        pos = sim[lat_pos[0], lat_pos[1]]
        # dim=1 cat keeps each src's own negatives grouped; reshape to
        # (n, 2, K): column 0 = uniform, column 1 = hard
        neg_s = sim[neg_src, torch.cat([neg_u, neg_h], dim=1).view(-1)].view(
            n_latent, 2, EVAL_NEGS)
        mrr_u, auc_u = sampled_mrr_auc(pos, neg_s[:, 0])
        mrr_h, auc_h = sampled_mrr_auc(pos, neg_s[:, 1])
        # exact AUC over the whole candidate space
        tri_scores = sim[tri_i, tri_j]
        tri_lat = latent[tri_i, tri_j]
        tri_cand = cand_mask[tri_i, tri_j]
        s_pos = tri_scores[tri_cand & tri_lat]
        s_neg = tri_scores[tri_cand & ~tri_lat]
        auc_full = auc_rank(s_pos, s_neg)
        recall = full_rank_recall(sim, cand_mask, lat_pos, RECALL_KS)
        prec = precision_at_k(sim, cand_mask, latent, PREC_KS)
        out = {
            "auc_sampled_uniform": round(auc_u, 4), "mrr_sampled_uniform": round(mrr_u, 4),
            "auc_sampled_hard": round(auc_h, 4), "mrr_sampled_hard": round(mrr_h, 4),
            "auc_full_ranking": round(auc_full, 4),
            **{k: round(v, 4) for k, v in recall.items()},
            **{k: round(v, 4) for k, v in prec.items()},
        }
        print(f"[{name}] AUC_full={out['auc_full_ranking']:.4f} "
              f"P@10={out['precision@10']:.4f} P@100={out['precision@100']:.4f} "
              f"R@10={out['recall@10']:.4f} MRR_u={out['mrr_sampled_uniform']:.4f}")
        return out

    results = {
        "feature_cosine": evaluate("feature_cosine", sim_feat),
        "adamic_adaar": evaluate("adamic_adaar", sim_aa),
        "sage_cosine": evaluate("sage_cosine", sim_sage),
    }

    # ---------------------------------------------------------------- ranker
    # Features come from the MP graph only (edge-disjoint protocol): the
    # SUP edges are the positive labels and were NEVER present in the SAGE
    # graph, so sim_sage / degrees / common-neighbors for a positive pair
    # are genuine out-of-fold predictions (no degree correction needed).
    with torch.no_grad():
        cn = (mp_adj.float() @ mp_adj.float())  # common neighbors (MP graph)
        cn.fill_diagonal_(0)
        deg = mp_adj.sum(dim=1)

        def pair_feats(a, b):
            """Relationship-specific features for index pairs (a, b),
            computed on the MP graph only (see above)."""
            return torch.stack([
                sim_sage[a, b],
                (sector[a] == sector[b]).float(),
                (region[a] == region[b]).float(),
                (size[a] - size[b]).abs(),
                torch.log1p(deg[a].float()),
                torch.log1p(deg[b].float()),
                torch.log1p(cn[a, b].float()),
            ], dim=-1)

        # train samples: positives = SUP edges (held out from the GNN).
        # negatives = UNOBSERVED pairs sampled from training-time information
        # only (~1.6% of them are in fact latent test positives - a
        # realistic, slightly conservative negative distribution; NOT an
        # oracle non-edge filter). SUP edges can't appear here: they are
        # part of train_adj.
        unobs_idx = (~train_adj)[tri_i, tri_j].nonzero().squeeze(1)
        n_neg = min(sup_pairs.size(1) * 2, unobs_idx.numel())
        sel = torch.randperm(unobs_idx.numel(), device=DEVICE, generator=rng)[:n_neg]
        neg_pairs = torch.stack([tri_i[unobs_idx[sel]], tri_j[unobs_idx[sel]]])
        X = torch.cat([pair_feats(sup_pairs[0], sup_pairs[1]),
                       pair_feats(neg_pairs[0], neg_pairs[1])])
        y = torch.cat([torch.ones(sup_pairs.size(1), device=DEVICE),
                       torch.zeros(n_neg, device=DEVICE)])

    ranker = PairRanker(n_feat=7).to(DEVICE)
    ropt = torch.optim.Adam(ranker.parameters(), lr=5e-3)
    for epoch in range(1, 11):
        ropt.zero_grad()
        pred = ranker(X)
        loss = F.binary_cross_entropy_with_logits(pred, y)
        loss.backward()
        ropt.step()
        if epoch % 2 == 0:
            print(f"[ranker] epoch {epoch:2d} bce={loss.item():.4f}")
    # chunked full-grid scoring (9M pairs would OOM in one MLP pass)
    with torch.no_grad():
        score_ranker = torch.empty(N, N, device=DEVICE)
        RB = 200
        b_all = torch.arange(N, device=DEVICE)
        for ib in range(0, N, RB):
            a = torch.arange(ib, min(ib + RB, N), device=DEVICE)
            A = a.unsqueeze(1).expand(-1, N).reshape(-1)
            B = b_all.unsqueeze(0).expand(len(a), N).reshape(-1)
            score_ranker[ib:ib + len(a)] = torch.sigmoid(
                ranker(pair_feats(A, B))).view(-1, N)
    results["pair_ranker"] = evaluate("pair_ranker", score_ranker)

    # ---------------------------------------------------------------- top pairs
    with torch.no_grad():
        sim = score_ranker
        masked = sim.masked_fill(~cand_mask, float("-inf"))
        cand_flat = masked[tri_i, tri_j]
        # deterministic tie-break (torch.topk has no stable option)
        tie = torch.arange(cand_flat.numel(), device=DEVICE, dtype=cand_flat.dtype) * 1e-9
        vals, idx = torch.topk(cand_flat + tie, TOP_K)
    top = []
    for j in range(TOP_K):
        a, b = int(tri_i[idx[j]]), int(tri_j[idx[j]])
        top.append({
            "src": a, "dst": b, "score": round(float(vals[j]), 4),
            "src_sector": meta["sector_names"][int(sector[a])],
            "dst_sector": meta["sector_names"][int(sector[b])],
            "src_region": meta["region_names"][int(region[a])],
            "dst_region": meta["region_names"][int(region[b])],
            "src_size": round(float(size[a]), 3), "dst_size": round(float(size[b]), 3),
            "common_neighbors": int(cn[a, b]),
            "is_latent_ground_truth": bool(latent[a, b]),
        })

    torch.save({"state_dict": model.state_dict(),
                "args": {"in_dim": x.shape[1], "hidden": HIDDEN, "out": HIDDEN}},
               OUT / "b2b_model.pt")
    torch.save({"state_dict": ranker.state_dict()}, OUT / "b2b_ranker.pt")
    (OUT / "b2b_metrics.json").write_text(json.dumps(
        {"n_companies": N, "n_train_edges": n_train, "n_latent_edges": n_latent,
         "protocol": "edge-disjoint: SAGE trains on MP edges only; ranker "
                    "positives = SUP edges (features computed without them)",
         "n_mp_edges": int(mp_pairs.size(1)), "n_sup_edges": int(sup_pairs.size(1)),
         "models": results, "train_seconds": round(time.time() - t0, 2), "seed": SEED},
        indent=2))
    (OUT / "b2b_top_pairs.json").write_text(json.dumps(top, indent=2))

    print("\n=== B2B headline: Precision@k over top unconnected pairs ===")
    for m, r in results.items():
        print(f"  {m:15s} P@1={r['precision@1']:.3f} P@10={r['precision@10']:.3f} "
              f"P@100={r['precision@100']:.3f} AUC={r['auc_full_ranking']:.3f}")
    hit = sum(t["is_latent_ground_truth"] for t in top)
    print(f"top-{TOP_K} (pair_ranker): {hit}/{TOP_K} are true latent edges")


if __name__ == "__main__":
    main()
