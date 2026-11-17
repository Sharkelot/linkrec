#!/usr/bin/env python3
"""PyTorch Geometric link prediction on the Cora citation graph.

Follows the PyG "Link Prediction" tutorial pattern, with a stricter
evaluation protocol:
  * train a GraphSAGE encoder with a margin-rank contrastive objective
  * hold out 20% of (undirected) edges, never shown during training
  * training negatives: mix of uniform-random AND hard (same-topic)
    negatives, so the model cannot cheat by only separating topics
  * evaluation (held-out edges):
      - MRR / AUC with uniform random negatives (comparable to the
        canonical PyG tutorial numbers)
      - MRR / AUC with hard (same-topic, non-neighbor) negatives
      - Recall@k by FULL RANKING: rank the true held-out destination
        among ALL nodes unconnected to the source (k = 1, 5, 10, 20);
        no negative sampling involved, the most robust metric
  * rank ALL non-edges and report the top-K unconnected pairs as
    "likely useful relationships that aren't explicitly connected",
    EXCLUDING pairs whose raw feature rows are identical (bag-of-words
    duplicates produce near-cosine-1.0 embeddings; an artifact, not a
    discovered relationship)

Outputs (outputs/):
  linkpred_model.pt          - GraphSAGE state dict + args
  linkpred_metrics.json      - full metric panel + protocol
  linkpred_top_edges.json    - top unconnected pairs with labels
  linkpred_data.pt           - x, edge splits, node labels (for serving)
"""
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SAGEConv

from eval_common import (full_rank_recall, sample_class_negatives,
                         sample_neg_excluding)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DATA = ROOT / "data"
OUT.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

SEED = int(os.environ.get("LINKREC_SEED", "42"))
HIDDEN = 128
EPOCHS = 80
LR = 1e-3
NEG_K = 10          # negatives per positive edge per type during training
EVAL_NEGS = 99      # negatives per held-out edge for MRR/AUC
HOLDOUT_FRAC = 0.2
TOP_K = 20
RECALL_KS = (1, 5, 10, 20)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CORA_CLASSES = ["case_based", "genetic_algorithms", "neural_networks",
                "probabilistic", "reinforcement", "rule_learning", "theory"]


def build_edge_splits(data, holdout_frac, rng):
    """Split Cora's directed edges into train / linkpred (undirected, unique)."""
    src, dst = data.edge_index
    undir = set()
    for a, b in zip(src.tolist(), dst.tolist()):
        undir.add((min(a, b), max(a, b)))
    edges = sorted(undir)
    rng.shuffle(edges)
    n_hold = max(1, int(len(edges) * holdout_frac))
    holdout, train = edges[:n_hold], edges[n_hold:]
    return train, holdout


def to_index(edges):
    if not edges:
        return torch.empty(2, 0, dtype=torch.long, device=DEVICE)
    return torch.tensor([[a for a, _ in edges], [b for _, b in edges]],
                        device=DEVICE)


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


def score_pairs(emb, src, dst):
    return F.cosine_similarity(emb[src], emb[dst], dim=-1)


def sampled_mrr_auc(pos_scores, neg_scores):
    """MRR and AUC from per-edge positive score vs K negative scores."""
    K = neg_scores.size(1)
    rank = (neg_scores > pos_scores.unsqueeze(1)).sum(dim=1)
    mrr = (1.0 / (rank + 1)).mean().item()
    auc = ((neg_scores < pos_scores.unsqueeze(1)).sum(dim=1)
           + 0.5 * (neg_scores == pos_scores.unsqueeze(1)).sum(dim=1)).mean().item() / K
    return mrr, auc


def main():
    # bit-level reproducibility: PyG scatter on CUDA is atomic-based and
    # otherwise drifts ~1e-4 run to run; deterministic mode pins the kernels
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    gen = torch.Generator(device=DEVICE).manual_seed(SEED)
    t0 = time.time()

    dataset = Planetoid(root=str(DATA / "Planetoid"), name="Cora")
    data = dataset[0].to(DEVICE)
    N = data.num_nodes
    y = data.y
    print(f"[data] Cora: {N} nodes, {data.edge_index.size(1)} directed edges, "
          f"{data.x.shape[1]} features (device={DEVICE})")

    train_edges, holdout_edges = build_edge_splits(data, HOLDOUT_FRAC, rng)
    train_idx, holdout_idx = to_index(train_edges), to_index(holdout_edges)
    print(f"[splits] train edges: {train_idx.size(1)}, held-out linkpred edges: {holdout_idx.size(1)}")

    data.x = F.normalize(data.x, dim=-1)
    # bidirectional message passing: each undirected train edge both ways
    data.edge_index = torch.cat([train_idx, train_idx.flip(0)], dim=1)

    # per-class node pools (for hard negatives)
    pools = [torch.nonzero(y == c).squeeze(1).contiguous() for c in range(int(y.max()) + 1)]
    adj = torch.zeros(N, N, dtype=torch.bool, device=DEVICE)
    adj[train_idx[0], train_idx[1]] = True
    adj = adj | adj.T  # undirected neighbor mask

    model = SAGE(data.x.shape[1], HIDDEN, HIDDEN).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    margin = 1.0
    hist = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()
        emb = model(data.x, data.edge_index)
        emb = F.normalize(emb, dim=-1)
        pos = score_pairs(emb, train_idx[0], train_idx[1])

        # negatives: uniform (excluding train neighbors/src/dst) + hard
        # (same class); cat along dim=1 so each edge keeps ITS OWN negatives
        neg_uniform = sample_neg_excluding(train_idx[0], train_idx[1], adj,
                                           NEG_K, gen, N)
        neg_hard = sample_class_negatives(train_idx[0], train_idx[1], y[train_idx[1]],
                                          pools, adj, NEG_K, gen, N)
        neg_src = train_idx[0].repeat_interleave(2 * NEG_K)
        neg_dst = torch.cat([neg_uniform, neg_hard], dim=1).view(-1)
        neg = score_pairs(emb, neg_src, neg_dst)
        neg = neg.view(train_idx.size(1), 2, NEG_K).mean(dim=(1, 2))
        loss = F.relu(margin - pos + neg).mean()
        loss.backward()
        opt.step()
        hist.append(loss.item())
        if epoch % 10 == 0 or epoch == 1:
            print(f"[train] epoch {epoch:3d} loss={loss.item():.4f}")

    # ---------------- evaluation on held-out edges ----------------
    model.eval()
    with torch.no_grad():
        emb = model(data.x, data.edge_index)
        emb = F.normalize(emb, dim=-1)
        held = holdout_idx
        pos_scores = score_pairs(emb, held[0], held[1]).cpu()

        # eval negatives: uniform with the same exclusions as training
        neg_uniform = sample_neg_excluding(held[0], held[1], adj, EVAL_NEGS, gen, N)
        mrr_uniform, auc_uniform = sampled_mrr_auc(
            pos_scores, score_pairs(emb, held[0].repeat_interleave(EVAL_NEGS),
                                    neg_uniform.view(-1)).cpu().view(-1, EVAL_NEGS))

        neg_hard = sample_class_negatives(held[0], held[1], y[held[1]],
                                          pools, adj, EVAL_NEGS, gen, N)
        mrr_hard, auc_hard = sampled_mrr_auc(
            pos_scores, score_pairs(emb, held[0].repeat_interleave(EVAL_NEGS),
                                    neg_hard.view(-1)).cpu().view(-1, EVAL_NEGS))

        # full-ranking Recall@k: rank true dst among all non-train-neighbors
        # of src. Other held-out edges DO compete (they are unconnected to
        # src in the train graph); the positive itself cancels under the
        # tie-break (score_v + tb_v is never > itself).
        sim = emb @ emb.T
        cand = ~(adj | torch.eye(N, device=DEVICE, dtype=torch.bool))
        recall_at = full_rank_recall(sim, cand, held, RECALL_KS)

        # top unconnected pairs: mask KNOWN edges (train + held + self) and
        # identical-feature (duplicate) pairs
        held_adj = torch.zeros(N, N, dtype=torch.bool, device=DEVICE)
        held_adj[held[0], held[1]] = True
        held_adj[held[1], held[0]] = True
        known = adj | held_adj | torch.eye(N, device=DEVICE, dtype=torch.bool)
        _, inv = torch.unique(data.x, dim=0, return_inverse=True)
        dup = inv.unsqueeze(0) == inv.unsqueeze(1)
        tri = torch.triu_indices(N, N, offset=1, device=DEVICE)
        cand_flat = sim[tri[0], tri[1]]
        cand_flat = cand_flat.masked_fill(
            known[tri[0], tri[1]] | dup[tri[0], tri[1]], float("-inf"))
        n_dup_excluded = int((dup[tri[0], tri[1]] & ~known[tri[0], tri[1]]).sum().item())
        topk = min(TOP_K, int((~cand_flat.isinf()).sum().item()))
        tie = torch.arange(cand_flat.numel(), device=DEVICE, dtype=cand_flat.dtype) * 1e-9
        vals, idx = torch.topk(cand_flat + tie, topk)
        top_pairs = [(int(tri[0][idx[j]]), int(tri[1][idx[j]]), float(vals[j]))
                     for j in range(topk)]

    names = [f"paper {i} ({CORA_CLASSES[int(y[i])]})" for i in range(N)]
    top_out = [
        {"src": a, "dst": b, "score": round(s, 4),
         "src_title": names[a], "dst_title": names[b]}
        for a, b, s in top_pairs
    ]

    torch.save({"state_dict": model.state_dict(),
                "args": {"in_dim": data.x.shape[1], "hidden": HIDDEN, "out": HIDDEN}},
               OUT / "linkpred_model.pt")
    torch.save({"x": data.x.cpu(),
                "edge_index_train": data.edge_index.cpu(),  # bidirectional (as trained)
                "edge_index_heldout": holdout_idx.cpu(),
                "num_nodes": N,
                "names": names}, OUT / "linkpred_data.pt")

    metrics = {
        "dataset": "Cora",
        "num_nodes": N,
        "train_edges": int(train_idx.size(1)),
        "heldout_edges": int(holdout_idx.size(1)),
        "model": "GraphSAGE(2x SAGEConv, hidden=128, cosine margin-rank, mixed uniform+hard negatives)",
        # sampled-negative metrics
        "auc_uniform": round(auc_uniform, 4),
        "mrr_uniform": round(mrr_uniform, 4),
        "auc_hard": round(auc_hard, 4),
        "mrr_hard": round(mrr_hard, 4),
        # full-ranking metrics (no negative sampling)
        **recall_at,
        # compatibility aliases for the original dashboard keys
        "auc": round(auc_uniform, 4),
        "mrr": round(mrr_uniform, 4),
        # top-pair ranking hygiene
        "duplicate_feature_pairs_excluded": n_dup_excluded,
        "negative_sampling": {
            "training": f"{NEG_K} uniform (excluding train neighbors/src/dst) + {NEG_K} same-class (hard) per edge",
            "eval_uniform": f"{EVAL_NEGS} random non-neighbor nodes per held-out edge (same exclusions)",
            "eval_hard": f"{EVAL_NEGS} same-class non-neighbor nodes per held-out edge",
            "full_ranking": "Recall@k ranks the true edge among all non-train-neighbors (other held-out edges compete)",
            "message_passing": "bidirectional (both orientations of each undirected train edge)",
        },
        "train_seconds": round(time.time() - t0, 2),
        "final_loss": round(hist[-1], 4),
        "seed": SEED,
    }
    (OUT / "linkpred_metrics.json").write_text(json.dumps(metrics, indent=2))
    (OUT / "linkpred_top_edges.json").write_text(json.dumps(top_out, indent=2))

    print("\n=== Link prediction results (held-out edges) ===")
    print(f"AUC/MRR (uniform negatives): {auc_uniform:.4f} / {mrr_uniform:.4f}")
    print(f"AUC/MRR (hard same-topic):   {auc_hard:.4f} / {mrr_hard:.4f}")
    print("Recall@k (full ranking):     " +
          ", ".join(f"R@{k}={recall_at[f'recall@{k}']:.4f}" for k in RECALL_KS))
    print(f"duplicate-feature unconnected pairs excluded from top list: {n_dup_excluded}")
    print(f"\nTop {len(top_pairs)} UNCONNECTED pairs with highest predicted similarity:")
    for p in top_out:
        print(f"  {p['score']:.4f}  [{p['src']}] {p['src_title'][:48]}  <->  [{p['dst']}] {p['dst_title'][:48]}")


if __name__ == "__main__":
    main()
