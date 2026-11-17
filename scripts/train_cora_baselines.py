#!/usr/bin/env python3
"""Non-learned baselines for the Cora link-prediction task.

Models:
  * feature_cosine  - cosine of the raw (normalized) bag-of-words features,
    no graph information at all
  * adamic_adaar    - sum over common neighbors of 1/log(deg), on the
    TRAIN graph only (classic non-embedding link-prediction baseline)

Identical protocol to train_linkpred.py:
  * same seed-42 edge split (20% held out)
  * MRR/AUC with 99 uniform and 99 hard (same-topic) sampled negatives
  * Recall@1/5/10/20 by full ranking among all unconnected candidates

Output: outputs/cora_baselines.json
"""
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from train_linkpred import (EVAL_NEGS, RECALL_KS, build_edge_splits, to_index,
                            sample_negatives)
from eval_common import full_rank_recall, sampled_mrr_auc

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DATA = ROOT / "data"
SEED = 42
HOLDOUT_FRAC = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    gen = torch.Generator(device=DEVICE).manual_seed(SEED)
    t0 = time.time()

    from torch_geometric.datasets import Planetoid
    dataset = Planetoid(root=str(DATA / "Planetoid"), name="Cora")
    data = dataset[0].to(DEVICE)
    N = data.num_nodes
    y = data.y

    train_edges, holdout_edges = build_edge_splits(data, HOLDOUT_FRAC, rng)
    train_idx, holdout_idx = to_index(train_edges), to_index(holdout_edges)
    print(f"[splits] train={train_idx.size(1)} heldout={holdout_idx.size(1)} (same as train_linkpred)")

    x = F.normalize(data.x, dim=-1)
    adj = torch.zeros(N, N, dtype=torch.bool, device=DEVICE)
    adj[train_idx[0], train_idx[1]] = True
    adj = adj | adj.T
    pools = [torch.nonzero(y == c).squeeze(1).contiguous() for c in range(int(y.max()) + 1)]

    held = holdout_idx
    neg_u = torch.randint(0, N, (held.size(1), EVAL_NEGS), device=DEVICE, generator=gen)
    neg_h = sample_negatives(held[0], held[1], y, pools, adj, EVAL_NEGS, DEVICE, gen)
    neg_src = held[0].repeat_interleave(2 * EVAL_NEGS)

    # candidate mask for full ranking: not connected in train, not held-out, not self
    held_adj = torch.zeros(N, N, dtype=torch.bool, device=DEVICE)
    held_adj[held[0], held[1]] = True
    held_adj[held[1], held[0]] = True
    cand_mask = ~(adj | held_adj | torch.eye(N, dtype=torch.bool, device=DEVICE))

    sim_feat = x @ x.T
    deg = adj.float().sum(dim=1)
    w = torch.where(deg > 1, 1.0 / torch.log(deg.clamp(min=2)), torch.zeros_like(deg))
    A = adj.float()
    aa = torch.diag_embed(w) @ A @ A.t() @ torch.diag_embed(w)
    aa = aa * (aa > 0).float()
    sim_aa = aa / aa.max().clamp(min=1.0)

    results = {}
    for name, sim in (("feature_cosine", sim_feat), ("adamic_adaar", sim_aa)):
        pos = sim[held[0], held[1]]
        neg = sim[neg_src, torch.cat([neg_u, neg_h]).view(-1)]
        mrr_u, auc_u = sampled_mrr_auc(pos, neg[:held.size(1) * EVAL_NEGS].view(-1, EVAL_NEGS))
        mrr_h, auc_h = sampled_mrr_auc(pos, neg[held.size(1) * EVAL_NEGS:].view(-1, EVAL_NEGS))
        recall = full_rank_recall(sim, cand_mask, held, RECALL_KS)
        results[name] = {
            "auc_uniform": round(auc_u, 4), "mrr_uniform": round(mrr_u, 4),
            "auc_hard": round(auc_h, 4), "mrr_hard": round(mrr_h, 4),
            **{k: round(v, 4) for k, v in recall.items()},
        }
        print(f"[{name}] AUC_u={auc_u:.4f} MRR_u={mrr_u:.4f} "
              f"AUC_h={auc_h:.4f} MRR_h={mrr_h:.4f} R@10={recall['recall@10']:.4f}")

    (OUT / "cora_baselines.json").write_text(json.dumps(
        {"dataset": "Cora", "train_edges": int(train_idx.size(1)),
         "heldout_edges": int(holdout_idx.size(1)), "models": results,
         "train_seconds": round(time.time() - t0, 2), "seed": SEED}, indent=2))
    print(f"[saved] outputs/cora_baselines.json ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
