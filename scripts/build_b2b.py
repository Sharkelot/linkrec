#!/usr/bin/env python3
"""Build a synthetic B2B company relationship graph with GROUND TRUTH.

Design
------
N companies, each with:
  * sector  (12 discrete)   - observed (feature)
  * region  (16 discrete)   - observed (feature)
  * size    (log10 revenue) - observed (feature)
  * latent affinity vector z in R^8 - NOT observed

True relationship probability between companies a,b:
  p_ab = sigmoid(0.9*1[same sector] + 0.6*1[same region]
                 + 1.2*cos(z_a, z_b) + 0.5*1[|size_a-size_b| < 1.0] - b0)
with b0 chosen so the graph has ~2% edge density.

All pairs are sampled from p_ab -> the FULL ground-truth adjacency G.
  * observed (train) edges  = 80% of G's edges
  * held-out latent edges   = 20% of G's edges  <- the ground-truth
    "likely useful relationships that aren't explicitly connected"
  * G=False pairs           = true non-edges (negative space)

This gives a benchmark where "useful unconnected relationship" is a
measurable ground truth, unlike Cora where it is only a hypothesis.

Outputs:
  outputs/b2b_graph.pt   - x, sector, region, size, z, train/latent edges, masks
  outputs/b2b_meta.json  - sector/region names, counts, density
"""
import json
import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

SEED = int(os.environ.get("LINKREC_SEED", "42"))
N = 3000
N_SECTORS = 12
N_REGIONS = 16
Z_DIM = 8
LATENT_FRAC = 0.2

# Two calibrations of the same generative process:
#   observable   - observable attributes (sector/region/size) dominate edge
#                  probability; the unobserved z is a secondary signal.
#                  This is the regime where "find likely useful unconnected
#                  relationships" is answerable from observed data.
#   unlearnable  - the unobserved z dominates at low density (v1): even the
#                  best observable profile has only ~15-25% edge probability,
#                  so no model can produce precise top-k unconnected pairs.
#                  Kept as the control regime (measured: AUC 0.51-0.59,
#                  top-20 precision 0/20 - see experiments/b2b_v1_metrics.json).
REGIMES = {
    "observable": dict(density=0.08, w_sector=1.5, w_region=1.0, w_z=1.5, w_size=0.8),
    "unlearnable": dict(density=0.02, w_sector=0.9, w_region=0.6, w_z=1.2, w_size=0.5),
}
import argparse as _ap
_ap = _ap.ArgumentParser()
_ap.add_argument("--regime", choices=sorted(REGIMES), default="observable")
ARGS = _ap.parse_args()
REGIME = REGIMES[ARGS.regime]
TARGET_DENSITY = REGIME["density"]

SECTOR_NAMES = ["SaaS", "Fintech", "Healthcare", "Manufacturing", "Retail",
                "Logistics", "Energy", "Media", "Telecom", "Agriculture",
                "Construction", "Consulting"]
REGION_NAMES = ["North", "South", "East", "West", "Central", "Coastal",
                "Highland", "Delta", "Metro A", "Metro B", "Metro C",
                "Rural East", "Rural West", "Island N", "Island S", "Border"]


def main():
    g = torch.Generator().manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sector = torch.randint(0, N_SECTORS, (N,), generator=g)
    region = torch.randint(0, N_REGIONS, (N,), generator=g)
    size = torch.exp(torch.normal(0.0, 0.8, (N,), generator=g))  # lognormal
    z = torch.randn(N, Z_DIM, generator=g)

    # pairwise logits (upper triangle, N*(N-1)/2 pairs)
    tri_i, tri_j = torch.triu_indices(N, N, offset=1)
    sec_same = (sector[tri_i] == sector[tri_j]).to(torch.float32)
    reg_same = (region[tri_i] == region[tri_j]).to(torch.float32)
    zc = z
    cos_z = torch.nn.functional.cosine_similarity(zc[tri_i], zc[tri_j], dim=-1)
    size_close = (size[tri_i] - size[tri_j]).abs() < 1.0
    size_close = size_close.to(torch.float32)
    logit = (REGIME["w_sector"] * sec_same + REGIME["w_region"] * reg_same
             + REGIME["w_z"] * cos_z + REGIME["w_size"] * size_close).to(device)
    # choose b0 so mean(p) == TARGET_DENSITY (bisection on the bias)
    lo, hi = -8.0, 6.0
    for _ in range(60):
        mid = (lo + hi) / 2
        p = torch.sigmoid(logit + mid)
        if p.mean().item() > TARGET_DENSITY:
            hi = mid
        else:
            lo = mid
    b0 = (lo + hi) / 2
    p = torch.sigmoid(logit + b0)

    G = torch.bernoulli(p.cpu(), generator=g)
    n_true = int(G.sum().item())
    latent_mask_all = torch.zeros(N, N, dtype=torch.bool)

    edge_pairs = torch.stack([tri_i[G.bool()], tri_j[G.bool()]])
    perm = torch.randperm(edge_pairs.size(1), generator=g)
    edge_pairs = edge_pairs[:, perm]
    n_latent = max(1, int(edge_pairs.size(1) * LATENT_FRAC))
    latent_pairs = edge_pairs[:, :n_latent]
    train_pairs = edge_pairs[:, n_latent:]

    # full adjacency masks (symmetric)
    latent_mask_all[latent_pairs[0], latent_pairs[1]] = True
    latent_mask_all[latent_pairs[1], latent_pairs[0]] = True
    train_adj = torch.zeros(N, N, dtype=torch.bool)
    train_adj[train_pairs[0], train_pairs[1]] = True
    train_adj = train_adj | train_adj.t()
    full_adj = train_adj | latent_mask_all
    nonedge_mask = ~full_adj
    nonedge_mask.fill_diagonal_(False)

    # observed features: sector one-hot + region one-hot + normalized size
    x = torch.cat([
        torch.nn.functional.one_hot(sector, N_SECTORS).float(),
        torch.nn.functional.one_hot(region, N_REGIONS).float(),
        ((size - size.min()) / (size.max() - size.min())).unsqueeze(1),
    ], dim=1)

    torch.save({
        "x": x.cpu(),
        "sector": sector.cpu(),
        "region": region.cpu(),
        "size": size.cpu(),
        "z": z.cpu(),
        "edge_index_train": train_adj.nonzero().t().contiguous().cpu(),  # both directions
        "train_pairs": train_pairs.cpu(),
        "latent_pairs": latent_pairs.cpu(),
        "latent_mask": latent_mask_all.cpu(),
        "train_adj": train_adj.cpu(),
        "nonedge_mask": nonedge_mask.cpu(),
    }, OUT / "b2b_graph.pt")

    meta = {
        "n_companies": N,
        "n_sectors": N_SECTORS,
        "n_regions": N_REGIONS,
        "n_true_edges": n_true,
        "n_train_edges_undirected": int(train_pairs.size(1)),
        "n_latent_edges": int(latent_pairs.size(1)),
        "edge_density": round(n_true * 2 / (N * (N - 1)), 5),
        "bias_b0": round(b0, 4),
        "sector_names": SECTOR_NAMES,
        "region_names": REGION_NAMES,
        "seed": SEED,
        "regime": ARGS.regime,
        "feature_dims": {"sector_onehot": N_SECTORS, "region_onehot": N_REGIONS, "size": 1, "total": x.shape[1]},
        "latent_vector_dims": Z_DIM,
        "generative_weights": dict(REGIME, target_density=REGIME["density"]),
    }
    (OUT / "b2b_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"[b2b] N={N} true edges={n_true} density={meta['edge_density']} b0={b0:.4f}")
    print(f"[b2b] train(undirected)={meta['n_train_edges_undirected']} latent(held-out)={meta['n_latent_edges']}")
    print(f"[b2b] feature dims={x.shape[1]} (sector {N_SECTORS} + region {N_REGIONS} + size 1)")


if __name__ == "__main__":
    main()
