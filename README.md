# linkrec — embeddings & link prediction

**Live (2026-11-16):** https://liberal-myspace-soil-connectors.trycloudflare.com
(local: http://127.0.0.1:8620) — ephemeral quick-tunnel hostname, see caveats.

## Measured results (2026-11-16, seed 42, RTX 4060 Ti)

| metric | value |
|---|---|
| LightGCN **NDCG@10** (chronological test) | **0.1244** |
| LightGCN **Recall@10** (chronological test) | **0.1190** |
| LightGCN **MRR@10** (chronological test) | **0.2168** |
| validation (chronological) | NDCG@10 0.1239 · Recall@10 0.1286 · MRR@10 0.2108 |
| GraphSAGE link pred. **AUC** (1055 held-out Cora edges) | **0.8931** |
| GraphSAGE link pred. **MRR** (99 sampled negatives) | **0.3619** |
| inference latency (local, warm) | recommend ≈0.4 ms · linkpred ≈1–2 ms (cold full-matrix ≈33 ms) |
| end-to-end via Cloudflare edge (this host's RTT) | ≈0.12–0.24 s |

The model's top *unconnected* Cora pairs are dominated by same-topic paper
pairs (e.g. two reinforcement-learning papers, cosine ≈1.0, no citation
between them) — i.e. it does recover plausible missing links, mostly within
topic clusters.

Experiment: can a model identify *likely useful relationships that are not yet
explicitly connected*? Two parts, one served app:

1. **PyTorch Geometric link prediction** (Cora citation graph) — GraphSAGE
   embeddings + margin-rank contrastive training; 20% of edges are held out and
   never seen. We report AUC/MRR on held-out edges and rank **all non-edges**
   to surface the top unconnected pairs the model thinks should be linked.
   (Follows the PyG link-prediction tutorial pattern.)
2. **RecBole recommendation** (ML-100k) — LightGCN, trained and evaluated with
   a **chronological (time-ordered) split**: each user's earliest 80% of
   interactions train the model, the next 10% validate, the last 10% test.
   Full top-k ranking evaluation → **NDCG@10, Recall@10, MRR**. No future
   information leaks into predictions.
3. **Served + measured** — FastAPI on `127.0.0.1:8620`, exposed through a
   Cloudflare quick tunnel. Per-request inference latency is tracked
   (p50/p90/p99) and shown on the dashboard.

## Layout

```
scripts/train_linkpred.py   PyG GraphSAGE link prediction (Cora)
scripts/train_recsys.py     RecBole LightGCN (ml-100k, TO split)
serve/app.py                FastAPI app (dashboard + /recommend + /linkpred + /metrics)
outputs/                    models, metrics JSON, embeddings, top-edge candidates
data/                       downloaded datasets (Cora, ml-100k)
```

## Endpoints

| path                | description |
|---------------------|-------------|
| `/`                 | dashboard: ranking metrics, latency percentiles, top unconnected pairs, rec form |
| `/recommend?user_id=196&k=10` | top-k items for a user (LightGCN embeddings, known items excluded) |
| `/linkpred`         | top unconnected Cora edge candidates (GraphSAGE cosine) |
| `/linkpred/score?src=1&dst=2` | predicted similarity for a specific pair |
| `/metrics`          | eval metrics (NDCG@10, Recall@10, MRR, AUC) + latency stats |

## Reproduce

```bash
.venv/bin/python scripts/train_linkpred.py   # ~2 min on GPU
.venv/bin/python scripts/train_recsys.py     # few min on GPU
.venv/bin/python -m uvicorn serve.app:app --host 127.0.0.1 --port 8620
```

Tunnel (bundled cloudflared, quick tunnel):

```bash
/home/coreys/.dsh/mobile-access-desktop/components/cloudflared/2026.9.1/cloudflared \
  tunnel --url http://127.0.0.1:8620 --no-autoupdate --logfile logs/cloudflared.log
# URL of the form https://*.trycloudflare.com is printed in the log
```

## Notes & caveats

- Quick-tunnel URLs are ephemeral (new hostname on each start). For a stable
  hostname, run a named tunnel on the `afterlifehigh.com` Cloudflare account
  (needs a tunnel token) and point `linkrec.*.afterlifehigh.com` at `:8620`.
- `eval_args.order: TO` (RecBole 1.2.0) sorts each user's interactions by the
  `timestamp` field before the 80/10/10 ratio split — the split is strictly
  chronological per user.
- Latency numbers on the dashboard are **local** (loopback, warm model).
  End-to-end latency through the Cloudflare edge is higher (RTT + edge hop)
  and is measured separately at verification time.
