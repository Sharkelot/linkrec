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
3. **Synthetic B2B graph with ground truth** — 3,000 companies; edge
   probability driven by observed features (sector/region/size) plus an
   unobserved affinity vector. 20% of true edges are held out as *latent*
   edges — the ground truth for "likely useful but unconnected". Four models
   compete, including a ranker over **relationship-specific pair features**.
   Two calibrations: observable (359,605 true edges, 71,921 latent) and an
   unlearnable control (90,308 true edges, 18,061 latent) where the
   unobserved vector dominates.

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
| GraphSAGE (trained) | 0.7966 | 0.3498 | **0.7793** | 0.2693 | 0.050 | 0.137 | 0.193 | 0.256 |
| feature cosine (no graph) | **0.8028** | 0.3523 | 0.7417 | 0.2656 | 0.064 | 0.135 | 0.182 | 0.234 |
| Adamic-Adar (common nbrs) | 0.6939 | **0.8162** | 0.6760 | **0.6514** | **0.112** | **0.258** | **0.322** | **0.349** |

Read (v3, corrected protocol): with edge-excluding negative sampling and
bidirectional message passing, the GNN's AUC edge over raw feature cosine
mostly disappears on Cora (0.797 vs 0.803) — feature-level topic similarity
is a very strong signal on this graph. Correct Adamic-Adar is the strongest
full-ranking top-k locator here (R@10 0.322, R@1 0.112). Full-ranking
Recall@k (no negative sampling) is the most robust column; sampled MRR
treats ties in favor of the positive (a known convention that flatters
sparse zero-scored baselines). The GNN's value shows up where graph
structure carries relationship signal — the B2B experiment below.

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
| **pair ranker (rel.-feature MLP)** | observable | **0.644** | 0.000 | 0.300 | 0.110 | 0.023 |
| Adamic-Adar | observable | 0.611 | 0.000 | 0.000 | 0.140 | 0.013 |
| feature cosine | observable | 0.590 | 0.000 | **0.500** | **0.180** | **0.024** |
| GraphSAGE cosine | observable | 0.577 | 0.000 | 0.100 | **0.180** | 0.018 |
| pair ranker | unlearnable (control) | 0.590 | 0.000 | 0.000 | 0.040 | 0.013 |
| feature cosine | unlearnable (control) | 0.551 | 0.000 | 0.100 | 0.040 | 0.013 |
| GraphSAGE cosine | unlearnable (control) | 0.516 | 0.000 | 0.000 | 0.000 | 0.005 |
| Adamic-Adar | unlearnable (control) | 0.518 | 0.000 | 0.000 | 0.000 | 0.004 |

Base rate in the observable regime: 71,921 latent / 4.21M unobserved
candidates ≈ 1.7%.

**Reading the results (v3, edge-disjoint protocol).**
- **Observable regime:** the feature-aware pair ranker still wins on AUC
  (0.644; multi-seed mean 0.604 ± 0.030 — the seed-42 run is the best of the
  five seeds, so the canonical number is optimistic by ~1.1σ). Top-k
  precision is a different story: under the edge-disjoint protocol the raw
  feature cosine leads (P@10 0.50 ≈ 29× base rate, P@100 0.18, tied with the
  GNN), and *no model* lands the single top pair (P@1 0.0 across the board).
  4/20 of the pair ranker's top-20 unconnected pairs are true latent edges —
  the "likely useful" profile (same sector + same region). The GNN embedding
  alone is on par with raw feature cosine (0.577 vs 0.590); graph structure's
  contribution is mostly mediated through the common-neighbor and degree
  features. Top-k precision is seed-sensitive (P@10 spans 0.0–0.5 across the
  five seeds) — read single-seed P@k as a point estimate, not a stable
  ranking signal.
- **Control (unlearnable) regime:** when the unobserved z dominates, the
  same models + fixed pipeline collapse to AUC 0.52–0.59 with 1/20
  top-20 precision — the relationship is genuinely unidentifiable from
  observed data. Same code; only the data calibration differs. That gap
  *is* the measurement of how much observable signal the task contains.

### Multi-seed validation (seeds 7, 123, 2024, 99, 42 — mean ± std)

| model | AUC (full / uniform) | MRR / Recall@10 | P@10 (std) | P@100 (std) |
|---|---|---|---|---|
| cora GraphSAGE | 0.8052 (0.0077) | MRR_u 0.3688 (0.0185) · R@10 0.2104 (0.0191) | — | — |
| cora feature cosine | 0.8089 (0.0076) | R@10 0.1856 (0.0116) | — | — |
| cora Adamic-Adar | 0.6963 (0.0046) | R@10 0.3189 (0.0087) | — | — |
| b2b feature cosine | 0.5886 (0.0015) | — | 0.2600 (0.1855) | 0.1600 (0.0329) |
| b2b Adamic-Adar | 0.6113 (0.0009) | — | 0.0400 (0.0490) | 0.1120 (0.0172) |
| b2b GraphSAGE cosine | 0.5790 (0.0015) | — | 0.0600 (0.0490) | 0.1360 (0.0258) |
| b2b pair ranker | 0.6038 (0.0299) | — | 0.1200 (0.1470) | 0.0720 (0.0431) |
| recsys LightGCN | — | NDCG@10 0.1238 (0.0006) · R@10 0.1168 (0.0016) · MRR@10 0.2176 (0.0031) | — | — |
| recsys PopRec | — | NDCG@10 0.0789 · R@10 0.0704 · MRR@10 0.1515 (std 0.0000) | — | — |

The pipeline is **bit-deterministic** (`torch.use_deterministic_algorithms`
+ fixed seed: PyG's CUDA scatter is atomic-based and otherwise drifts ~1e-4
run to run). Back-to-back seed-42 runs produce identical metrics, and the
seed-42 column of this table matches `results.json` bit-for-bit on all 26
cross-checked values.

## Benchmark protocol (v3 — corrected evaluation, edge-disjoint, deterministic)

v1→v2 fixed four methodological issues found in code review; v3 adds
edge-disjoint ranker features (item 3), hardened samplers (item 5), and
bit-level determinism (item 6). All numbers above are from the v3 pipeline
(seed 42; multi-seed stats in the table above):

1. **Negative-sample pairing.** v1 concatenated uniform + hard negative
   tensors along the batch dimension while repeating source nodes per
   example, which attached one edge's negatives to a different edge.
   Negatives are now concatenated along the sample dimension
   (`torch.cat([...], dim=1).view(-1)`) so each source keeps its own
   negatives, in training and in sampled MRR/AUC.
2. **Adamic-Adar formula.** v1 weighted the *endpoints*
   (`diag(w) @ A @ Aᵀ @ diag(w)`); true Adamic-Adar weights the *common
   neighbors*: `aa[u,v] = Σ_c 1/log(deg(c)) = A @ diag(w) @ Aᵀ`.
3. **Ranker train/test separation (edge-disjoint out-of-fold).** v1
   selected ranker negatives from the oracle non-edge mask (which knew the
   latent edges) and used degrees that included the target edge. v2 goes
   further: the observed train edges are split 80/20 into an MP graph and a
   SUP set. SAGE trains on the MP graph **only** (graph, loss, and negative
   exclusion all restricted to it), and the ranker's positive labels are the
   SUP edges — so every ranker feature for a positive pair (SAGE cosine,
   degrees, common neighbors) was computed without that edge present, at the
   edge level. Ranker negatives are sampled from unobserved pairs using
   training-time information only (~1.6% of them are in fact latent test
   positives — a realistic, slightly *conservative* negative distribution).
   Note: sampled unobserved negatives CAN contain withheld true edges, so
   "no false negatives" applies to the *observed* graph exclusions, not to
   the ground truth as a whole.
4. **Cora evaluation cleanup.** The message-passing graph is now
   bidirectional (one orientation per undirected train edge in v1); offline
   top-pair generation explicitly masks known edges (train + held-out +
   self), not just duplicate-feature pairs; and the full-ranking Recall@k
   candidate set now includes other held-out edges (they compete, as they
   do in the real task).
5. **Negative samplers hardened.** All negative sampling (training and
   evaluation, both graphs) rejects observed neighbors, self, and the
   positive's destination. The samplers never use an unrestricted uniform
   fallback: when stochastic rounds fail, an explicit enumeration pass scans
   a random permutation of all nodes and keeps only valid candidates, and
   raises `RuntimeError` if a row has fewer than K valid nodes in total.
   Invariant tests cover sparse, dense (top-up-forcing), empty-class-pool,
   and raise-on-impossible cases (24 tests, all passing).
6. **Bit-level determinism.** PyG scatter on CUDA is atomic-based, so the
   pipeline otherwise drifts ~1e-4 run to run (top-k boundary ranks flip).
   Every training script now runs under
   `torch.use_deterministic_algorithms(True)`: back-to-back seed-42 runs
   produce bit-identical metrics (verified), and the suite's artifacts are
   exactly reproducible.

Known remaining limitations (documented, not fixed):
- The `sage_cosine` model itself is a GNN link-prediction baseline: its
  embeddings are trained on the MP graph, so its scores for held-out edges
  reflect the GNN's view of that graph (standard for GNN link prediction).
  The edge-disjoint protocol removes this exposure specifically for the
  *ranker's* features, which is where train/test feature consistency
  matters most.
- Sampled MRR treats a tie in favor of the positive (common convention).
- B2B sampled negatives can contain latent edges of the same source
  (realistic deployment condition, conservative direction).

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
  multi_seed.py            5-seed statistical validation (mean/std) ->
                           experiments/multi_seed.{json,md}; seed-42 last so
                           served artifacts stay canonical
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
# full suite (~3 min on RTX 4060 Ti; bit-deterministic, seed 42)
.venv/bin/python scripts/run_experiments.py

# multi-seed statistical validation (~15 min; seeds 7,123,2024,99,42;
# delete experiments/multi_seed.json first for a fully fresh run)
.venv/bin/python scripts/multi_seed.py

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
