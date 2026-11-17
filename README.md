# linkrec — embeddings, link prediction & recommendations

Experiment: **can a model identify likely useful relationships that are not yet
explicitly connected?** Three parts, one served app, all measured with
repeatable, seed-42 pipelines:

1. **PyTorch Geometric link prediction** (Cora citation graph) — GraphSAGE
   embeddings, margin-rank contrastive training with mixed uniform + hard
   (same-topic) negatives; 20% of edges held out, never trained on.
2. **RecBole recommendation** (ML-100k) — LightGCN with a **chronological
   per-user 80/10/10 split** (no future leakage), full top-10 ranking
   evaluation: NDCG@10, Recall@10, MRR.
3. **Synthetic B2B graph with ground truth** — 3,000 companies, 90k true
   relationships; 20% held out as *latent* edges. This is where
   "likely useful but unconnected" is a measurable ground truth. Four models
   compete, including a ranker over **relationship-specific pair features**.

Served through FastAPI behind a Cloudflare tunnel, with per-request latency
tracking and a local-LLM `/explain` endpoint.

## Live

- **Dashboard:** `http://127.0.0.1:8620` (systemd: `linkrec.service`)
- **Public (quick tunnel, ephemeral hostname):** see `logs/cloudflared.log`
  for the current `https://*.trycloudflare.com` URL (systemd:
  `linkrec-tunnel.service`)

## Results (seed 42, RTX 4060 Ti, 2026-11-16)

### Link prediction — Cora (1,055 held-out edges)

| model | AUC (uniform negs) | MRR (uniform) | AUC (hard same-topic) | MRR (hard) | R@1 | R@5 | R@10 | R@20 |
|---|---|---|---|---|---|---|---|---|
| GraphSAGE (trained) | **0.8874** | 0.3545 | 0.7716 | 0.2364 | 0.032 | 0.116 | 0.168 | 0.244 |
| feature cosine (no graph) | 0.7976 | 0.3481 | 0.7860 | 0.3602 | 0.063 | 0.134 | 0.177 | 0.225 |
| Adamic-Adar (common nbrs) | 0.6495 | **0.7658** | 0.6493 | **0.8512** | 0.051 | 0.141 | 0.194 | **0.241** |

Read: the GNN embedding improves *discrimination* (AUC) over both non-learned
baselines; on this sparse graph, common-neighbor structure is a surprisingly
strong top-k locator (full-ranking R@10). Full-ranking Recall@k (no negative
sampling) is the most robust column; sampled MRR treats ties in favor of the
positive (a known convention that flatters sparse zero-scored baselines).

### Recommendations — ML-100k (chronological split, full ranking, top-10)

| model | NDCG@10 (test) | Recall@10 (test) | MRR@10 (test) | NDCG@10 (val) |
|---|---|---|---|---|
| LightGCN (RecBole) | **0.1228** | **0.1148** | **0.2170** | 0.1236 |
| PopRec (item frequency) | 0.0789 | 0.0704 | 0.1515 | 0.0673 |

Each user's interactions are split by timestamp (earliest 80% train, next 10%
validate, latest 10% test). Test items are strictly later than train items —
verified per user (0 invariant violations across 943 users). LightGCN beats
the popularity baseline on all three metrics.

### B2B synthetic ground truth (two calibrations of one generative process)

The graph has 3,000 companies; edge probability is driven by sector match,
region match, size similarity (all **observed** features) plus a latent
affinity vector **z** (unobserved). 20% of the true edges are held out as
*latent* edges — the ground truth for "likely useful but unconnected".

| model | regime | AUC (full) | P@1 | P@10 | P@100 | R@10 |
|---|---|---|---|---|---|---|
| **pair ranker (rel.-feature MLP)** | observable | **0.646** | **1.000** | **0.400** | 0.140 | 0.023 |
| GraphSAGE cosine | observable | 0.590 | 0.000 | 0.200 | 0.160 | 0.022 |
| feature cosine | observable | 0.590 | 1.000 | 0.200 | 0.150 | 0.024 |
| Adamic-Adar | observable | 0.611 | 0.000 | 0.100 | 0.130 | 0.014 |
| pair ranker | unlearnable (control) | 0.588 | 0.000 | 0.000 | 0.010 | 0.013 |
| feature cosine | unlearnable (control) | 0.551 | 0.000 | 0.000 | 0.020 | 0.013 |
| GraphSAGE cosine | unlearnable (control) | 0.540 | 0.000 | 0.000 | 0.000 | 0.007 |
| Adamic-Adar | unlearnable (control) | 0.514 | 0.000 | 0.000 | 0.000 | 0.004 |

Base rate in the observable regime: 71,921 latent / 4.48M candidates ≈ 1.6%.

**Reading the results.**
- **Observable regime:** the feature-aware pair ranker beats everything on
  AUC (0.646) and top-k precision (P@10 = 0.40, i.e. 25× base rate; P@1 = 1.0).
  Its top unconnected pairs are exactly the "likely useful" profile:
  same sector + same region (e.g. Energy/Highland ↔ Energy/Highland).
  6/20 of its top-20 are true latent edges. The GNN embedding alone adds
  little beyond the raw features here (AUC 0.590 = feature cosine); the
  graph structure shows up mostly through common-neighbor signals.
- **Control (unlearnable) regime:** when the unobserved z dominates, the same
  models collapse to AUC 0.51-0.59 with 0/20 top-precision — the relationship
  is genuinely unidentifiable from observed data. Same models, same code;
  only the data calibration differs. That gap *is* the measurement of how
  much observable signal the task contains.

## Endpoints

| path | description |
|---|---|
| `/` | dashboard: ranking metrics, latency percentiles, top unconnected pairs (Cora + B2B), rec form, explain form |
| `/recommend?user_id=196&k=10` | top-k items (LightGCN; API speaks **raw ML-100k ids**; known items masked) |
| `/linkpred` | top unconnected Cora pairs (GraphSAGE cosine; duplicate-feature pairs excluded) |
| `/linkpred/score?src=1&dst=2` | predicted similarity for a specific pair |
| `/b2b` | top unconnected B2B pairs (pair-ranker scores) + ground-truth annotation |
| `/explain?src=0&dst=1&graph=b2b\|cora` | local-LLM (Chon-12B via Ollama) explanation of a prediction, with the model-derived facts; 503 + facts if the LLM is down |
| `/metrics` | eval metrics (both tasks) + per-endpoint latency stats |

## Layout

```
scripts/
  train_linkpred.py        PyG GraphSAGE link prediction (Cora) + full metric panel
  train_cora_baselines.py  feature_cosine + adamic_adaar baselines (same protocol)
  train_recsys.py          RecBole LightGCN on real ML-100k (chronological TO split,
                           audited per-user, saves split pairs/manifest + id map)
  train_poprec.py          popularity baseline (same split, same eval semantics)
  validate_recsys.py       ID-mapping + split + metric-recomputation audit (18 checks)
  build_b2b.py             synthetic B2B graph with ground-truth latent edges
  train_b2b.py             4 models: feature_cosine, adamic_adaar, sage_cosine,
                           pair_ranker (relationship-feature MLP)
  run_experiments.py       one-command suite -> experiments/results.{json,md}
  eval_common.py           exact rank-based metrics (AUC, MRR, Recall@k, Precision@k)
serve/app.py               FastAPI app (dashboard + all endpoints above)
tests/                     pytest: eval-math unit tests + API contract tests
outputs/                   models, metrics, embeddings, split manifests (gitignored)
data/                      downloaded datasets (gitignored)
experiments/               results.json / results.md + per-step logs (gitignored)
deploy/                    systemd units for app + tunnel
```

## Reproduce

```bash
# full suite (~15 min on GPU; deterministic, seed 42)
.venv/bin/python scripts/run_experiments.py

# or piece by piece
.venv/bin/python scripts/train_linkpred.py        # Cora GraphSAGE
.venv/bin/python scripts/train_cora_baselines.py  # Cora baselines
.venv/bin/python scripts/train_recsys.py          # LightGCN + split audit
.venv/bin/python scripts/train_poprec.py          # PopRec baseline
.venv/bin/python scripts/validate_recsys.py       # 18-check audit (all must PASS)
.venv/bin/python scripts/build_b2b.py             # synthetic B2B dataset
.venv/bin/python scripts/train_b2b.py             # 4 B2B models
.venv/bin/python -m pytest tests/ -v              # unit + API tests

# serve
.venv/bin/python -m uvicorn serve.app:app --host 127.0.0.1 --port 8620
```

## Deployment

- `deploy/linkrec.service` — runs the app on `127.0.0.1:8620` (systemd,
  restart-on-failure, enabled at boot).
- `deploy/linkrec-tunnel.service` — runs a Cloudflare **quick tunnel**
  (bundled cloudflared binary) to `:8620`; prints a
  `https://*.trycloudflare.com` URL to `logs/cloudflared.log`.
  Quick-tunnel hostnames are **ephemeral** (new hostname each restart).
- Stable hostname option: create a **named tunnel** on the
  `afterlifehigh.com` Cloudflare account, then point a DNS record
  (e.g. `linkrec.afterlifehigh.com`) at it with ingress
  `linkrec.afterlifehigh.com -> http://127.0.0.1:8620`.

## Environment notes (verified on this host)

- Python 3.14 + torch 2.14 (CUDA 13) + PyG 2.8.1 + RecBole 1.2.0 in `.venv`
  (`.venv/hermes-torch.pth` reuses the host's CUDA torch install).
- RecBole 1.2.0 specifics that matter here:
  - `data_path` is **ignored for the name `ml-100k`** (hardcoded to the
    bundled sample, which is *not* the real dataset) — this project trains on
    the real ML-100k under the dataset name `ml100k` (typed `.inter` file,
    `load_col: [user_id, item_id, rating, timestamp]`).
  - chronological split = `eval_args: {order: TO, split: {RS: [0.8,0.1,0.1]},
    group_by: user}`; `normalize_all: True` is required so float32
    timestamp quantization doesn't reorder ties.
  - model classes live in `recbole.model.general_recommender.*`; the
    full-sort evaluator masks item 0 + the user's history (train for valid,
    train+valid for test).
  - RecBole remaps raw ids in first-occurrence order; the raw↔remapped maps
    are saved (`recsys_id_map.json`) and the API converts at the boundary.
- torch 2.14 renames: `triu_indices` takes `(row, col, offset=)`;
  `torch.lognormal` removed (use `exp(normal)`); PyG 2.8: `Planetoid` (not
  `PlanetoidDataset`).
- GPU is shared (ComfyUI) — B2B training is chunked to stay under ~3 GB.

## Caveats

- Cora "unconnected but related" has no ground truth — the B2B part exists
  precisely to measure that claim against known latent edges.
- Top Cora pairs exclude pairs with **identical** bag-of-words feature rows
  (they score ≈1.0 without graph signal); near-duplicate papers can still
  dominate the top of the cosine ranking.
- Quick-tunnel URL changes on tunnel restart; the dashboard and API are
  stable at `127.0.0.1:8620`.
- Latency numbers on the dashboard are local (loopback, warm model);
  end-to-end through the Cloudflare edge adds RTT + a TLS hop.
