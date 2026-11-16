#!/usr/bin/env python3
"""PyTorch Geometric link prediction on the Cora citation graph.

Follows the PyG "Link Prediction" tutorial pattern:
  * train a GraphSAGE encoder with a margin-rank contrastive objective
  * hold out 20% of (undirected) edges, never shown during training
  * evaluate AUC + MRR (sampled negatives) on held-out edges
  * rank ALL non-edges and report the top-K unconnected pairs as
    "likely useful relationships that aren't explicitly connected"

Outputs (outputs/):
  linkpred_model.pt          - GraphSAGE state dict + args
  linkpred_metrics.json      - AUC / MRR / train time
  linkpred_top_edges.json    - top unconnected pairs with paper titles
"""
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SAGEConv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DATA = ROOT / "data"
OUT.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

SEED = 42
HIDDEN = 128
EPOCHS = 80
LR = 1e-3
NEG_K = 10          # negatives per positive edge during training
EVAL_NEGS = 99      # negatives per held-out edge for MRR/AUC
HOLDOUT_FRAC = 0.2
TOP_K = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_edge_splits(data, holdout_frac, rng):
    """Split Cora's directed edges into train / linkpred (undirected, unique)."""
    src, dst = data.edge_index
    u, v = src.tolist(), dst.tolist()
    undir = set()
    for a, b in zip(u, v):
        undir.add((min(a, b), max(a, b)))
    edges = sorted(undir)
    rng.shuffle(edges)
    n_hold = max(1, int(len(edges) * holdout_frac))
    holdout, train = edges[:n_hold], edges[n_hold:]
    return train, holdout


def to_index(edges):
    if not edges:
        return torch.empty(2, 0, dtype=torch.long)
    return torch.tensor([[a for a, _ in edges], [b for _, b in edges]]).to(DEVICE)


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


def score_pairs(model, emb, src, dst):
    return F.cosine_similarity(emb[src], emb[dst], dim=-1)


def main():
    torch.manual_seed(SEED)
    rng = torch.Generator().manual_seed(SEED)
    gen = torch.Generator(device=DEVICE).manual_seed(SEED)
    t0 = time.time()

    dataset = Planetoid(root=str(DATA / "Planetoid"), name="Cora")
    data = dataset[0].to(DEVICE)
    print(f"[data] Cora: {data.num_nodes} nodes, {data.edge_index.size(1)} directed edges, {data.x.shape[1]} features (device={DEVICE})")

    import random
    train_edges, holdout_edges = build_edge_splits(data, HOLDOUT_FRAC, random.Random(SEED))
    train_idx, holdout_idx = to_index(train_edges), to_index(holdout_edges)
    print(f"[splits] train edges: {train_idx.size(1)}, held-out linkpred edges: {holdout_idx.size(1)}")

    # training edge_index keeps original direction (informational), both orientations available
    data.edge_index = train_idx
    data.x = F.normalize(data.x, dim=-1)

    model = SAGE(data.x.shape[1], HIDDEN, HIDDEN).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    margin = 1.0
    hist = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()
        emb = model(data.x, data.edge_index)
        emb = F.normalize(emb, dim=-1)
        pos = score_pairs(model, emb, train_idx[0], train_idx[1])
        # sample negatives: keep src, random dst
        neg_dst = torch.randint(0, data.num_nodes, (train_idx.size(1), NEG_K), device=DEVICE)
        neg_src = train_idx[0].repeat_interleave(NEG_K)
        neg = score_pairs(model, emb, neg_src, neg_dst.view(-1))
        neg = neg.view(train_idx.size(1), NEG_K).mean(dim=1)
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
        held = to_index(holdout_edges)
        pos_scores = score_pairs(model, emb, held[0], held[1]).cpu()

        # MRR + AUC with sampled negatives
        neg_dst = torch.randint(0, data.num_nodes, (held.size(1), EVAL_NEGS), device=DEVICE, generator=gen)
        neg_src = held[0].repeat_interleave(EVAL_NEGS)
        neg_scores = score_pairs(model, emb, neg_src, neg_dst.view(-1)).cpu().view(-1, EVAL_NEGS)

        rank = (neg_scores > pos_scores.unsqueeze(1)).sum(dim=1)  # 0-based rank
        mrr = (1.0 / (rank + 1)).mean().item()
        # AUC: fraction of negative scores below positive score
        auc = ((neg_scores < pos_scores.unsqueeze(1)).sum(dim=1) + 0.5 * (neg_scores == pos_scores.unsqueeze(1)).sum(dim=1)).mean().item() / EVAL_NEGS

    # ---------------- rank all non-edges ----------------
    with torch.no_grad():
        emb = model(data.x, data.edge_index)
        emb = F.normalize(emb, dim=-1)
        sim = emb @ emb.T  # full similarity matrix
    eye = torch.eye(data.num_nodes, device=DEVICE, dtype=torch.bool)
    adj = torch.zeros(data.num_nodes, data.num_nodes, device=DEVICE, dtype=torch.bool)
    adj[held[0], held[1]] = 1  # held-out edges are "unknown" too: we want edges the model didn't train on
    train_adj = torch.zeros(data.num_nodes, data.num_nodes, device=DEVICE, dtype=torch.bool)
    train_adj[train_idx[0], train_idx[1]] = 1
    adj = adj | train_adj | eye
    sim = sim.masked_fill(adj, float("-inf"))
    # top pairs (symmetric)
    tri = torch.triu_indices(data.num_nodes, data.num_nodes, offset=1, device=DEVICE)
    cand = sim[tri[0], tri[1]]
    topk = min(TOP_K, cand.numel())
    vals, idx = torch.topk(cand, topk)
    top_pairs = [(int(tri[0][idx[j]]), int(tri[1][idx[j]]), float(vals[j])) for j in range(topk)]

    # human-readable node labels: Cora's 7 topic categories (standard mapping)
    CORA_CLASSES = ["case_based", "genetic_algorithms", "neural_networks",
                    "probabilistic", "reinforcement", "rule_learning", "theory"]
    y = data.y.cpu()
    names = [f"paper {i} ({CORA_CLASSES[int(y[i])]})" for i in range(data.num_nodes)]
    top_out = [
        {"src": a, "dst": b, "score": round(s, 4),
         "src_title": names[a], "dst_title": names[b]}
        for a, b, s in top_pairs
    ]

    model_path = OUT / "linkpred_model.pt"
    torch.save({"state_dict": model.state_dict(),
                "args": {"in_dim": data.x.shape[1], "hidden": HIDDEN, "out": HIDDEN}}, model_path)

    data_path = OUT / "linkpred_data.pt"
    torch.save({"x": data.x.cpu(),
                "edge_index_train": train_idx.cpu(),
                "edge_index_heldout": holdout_idx.cpu(),
                "num_nodes": data.num_nodes,
                "names": names}, data_path)

    metrics = {
        "dataset": "Cora",
        "num_nodes": int(data.num_nodes),
        "train_edges": int(train_idx.size(1)),
        "heldout_edges": int(holdout_idx.size(1)),
        "model": "GraphSAGE(2x SAGEConv, hidden=128, cosine margin-rank)",
        "auc": round(auc, 4),
        "mrr": round(mrr, 4),
        "train_seconds": round(time.time() - t0, 2),
        "final_loss": round(hist[-1], 4),
        "seed": SEED,
    }
    (OUT / "linkpred_metrics.json").write_text(json.dumps(metrics, indent=2))
    (OUT / "linkpred_top_edges.json").write_text(json.dumps(top_out, indent=2))

    print("\n=== Link prediction results (held-out edges) ===")
    print(f"AUC  = {auc:.4f}")
    print(f"MRR  = {mrr:.4f}")
    print(f"\nTop {len(top_pairs)} UNCONNECTED pairs with highest predicted similarity:")
    for p in top_out:
        print(f"  {p['score']:.4f}  [{p['src']}] {p['src_title'][:48]}  <->  [{p['dst']}] {p['dst_title'][:48]}")


if __name__ == "__main__":
    main()
