#!/usr/bin/env python3
"""One-command repeatable experiment suite.

    .venv/bin/python scripts/run_experiments.py            # full retrain (~15 min)
    .venv/bin/python scripts/run_experiments.py --fast     # reuse model artifacts,
                                                           # recompute baselines only

Steps (all deterministic, SEED=42):
  1. Cora link prediction:
       - train_linkpred.py        (GraphSAGE, mixed uniform+hard negatives)
       - train_cora_baselines.py  (feature_cosine, adamic_adaar)
  2. ML-100k recommendations (chronological 80/10/10 split):
       - train_recsys.py          (RecBole LightGCN; --fast reuses artifacts)
       - train_poprec.py          (popularity baseline)
       - validate_recsys.py       (ID mapping + metric recomputation audit)
  3. Synthetic B2B graph (ground-truth latent edges):
       - build_b2b.py             (dataset)
       - train_b2b.py             (feature_cosine, adamic_adaar, sage_cosine, pair_ranker)

Outputs:
  experiments/results.json   - machine-readable, everything
  experiments/results.md     - human-readable tables
  (plus all per-task artifacts under outputs/)
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
OUT = ROOT / "outputs"
EXP = ROOT / "experiments"
EXP.mkdir(exist_ok=True)


def run(script: str, log_name: str):
    t0 = time.time()
    print(f"\n=== {script} ===", flush=True)
    with open(EXP / log_name, "w") as log:
        r = subprocess.run([PY, str(ROOT / "scripts" / script)],
                           stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    status = "ok" if r.returncode == 0 else f"FAILED ({r.returncode})"
    print(f"[{script}] {status} in {time.time() - t0:.0f}s (log: experiments/{log_name})")
    return r.returncode == 0


def load(name: str):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true",
                    help="reuse existing model artifacts (skip train_linkpred/train_recsys/train_b2b model training)")
    args = ap.parse_args()

    t0 = time.time()
    steps = []

    # 1. Cora
    if not args.fast:
        steps.append(("train_linkpred", run("train_linkpred.py", "train_linkpred.log")))
    steps.append(("cora_baselines", run("train_cora_baselines.py", "cora_baselines.log")))

    # 2. Recsys
    if not args.fast:
        steps.append(("train_recsys", run("train_recsys.py", "train_recsys.log")))
    steps.append(("poprec", run("train_poprec.py", "poprec.log")))
    steps.append(("validate_recsys", run("validate_recsys.py", "validate_recsys.log")))

    # 3. B2B
    steps.append(("build_b2b", run("build_b2b.py", "build_b2b.log")))
    if not args.fast:
        steps.append(("train_b2b", run("train_b2b.py", "train_b2b.log")))

    lp = load("linkpred_metrics.json")
    cb = load("cora_baselines.json")
    rec = load("recsys_metrics.json")
    pop = load("poprec_metrics.json")
    b2b = load("b2b_metrics.json")
    b2b_meta = load("b2b_meta.json")

    results = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "seed": 42,
        "steps": steps,
        "linkpred_cora": {
            "graphsage": {k: v for k, v in (lp or {}).items()
                          if k in ("auc_uniform", "mrr_uniform", "auc_hard", "mrr_hard",
                                   "recall@1", "recall@5", "recall@10", "recall@20")},
            "baselines": (cb or {}).get("models", {}),
            "heldout_edges": (lp or {}).get("heldout_edges"),
        },
        "recsys_ml100k": {
            "split": "chronological 80/10/10 per user (order: TO, group_by: user)",
            "lightgcn": rec,
            "poprec": pop,
        },
        "b2b": {
            "dataset": b2b_meta,
            "models": (b2b or {}).get("models", {}),
            "n_latent_edges": (b2b or {}).get("n_latent_edges"),
        },
    }
    (EXP / "results.json").write_text(json.dumps(results, indent=2))

    # ---------------- markdown ----------------
    L = []
    L.append("# linkrec — experiment results\n")
    L.append(f"Generated {results['generated_utc']} · python {results['python']} · seed 42\n")
    L.append(f"Steps: " + ", ".join(f"**{n}** {'PASS' if ok else 'FAIL'}" for n, ok in steps) + "\n")

    L.append("\n## 1. Link prediction — Cora (held-out edges, %s)\n"
             % results["linkpred_cora"]["heldout_edges"])
    L.append("| model | AUC (uniform negs) | MRR (uniform) | AUC (hard) | MRR (hard) | R@1 | R@5 | R@10 | R@20 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for name, r in (("GraphSAGE (trained)", results["linkpred_cora"]["graphsage"]),
                    *[(n, m) for n, m in results["linkpred_cora"]["baselines"].items()]):
        L.append(f"| {name} | {r.get('auc_uniform','—')} | {r.get('mrr_uniform','—')} | "
                 f"{r.get('auc_hard','—')} | {r.get('mrr_hard','—')} | "
                 f"{r.get('recall@1','—')} | {r.get('recall@5','—')} | "
                 f"{r.get('recall@10','—')} | {r.get('recall@20','—')} |")

    L.append("\n## 2. Recommendations — ML-100k (chronological split, full ranking, top-10)\n")
    L.append("| model | NDCG@10 (test) | Recall@10 (test) | MRR@10 (test) | NDCG@10 (val) |")
    L.append("|---|---|---|---|---|")
    for name, m in (("LightGCN (RecBole)", (rec or {})), ("PopRec (baseline)", (pop or {}))):
        t = m.get("test", {}) or {}
        v = m.get("valid", {}) or {}
        L.append(f"| {name} | {t.get('NDCG@10','—')} | {t.get('Recall@10','—')} | "
                 f"{t.get('MRR@10','—')} | {v.get('NDCG@10','—')} |")

    L.append("\n## 3. B2B synthetic graph (ground-truth latent edges: %s)\n"
             % results["b2b"]["n_latent_edges"])
    L.append("| model | AUC (full) | P@1 | P@10 | P@100 | R@10 (full) |")
    L.append("|---|---|---|---|---|---|")
    for name, r in results["b2b"]["models"].items():
        L.append(f"| {name} | {r.get('auc_full_ranking','—')} | {r.get('precision@1','—')} | "
                 f"{r.get('precision@10','—')} | {r.get('precision@100','—')} | "
                 f"{r.get('recall@10','—')} |")
    L.append("\n_P@k = precision over the model's top-k unconnected pairs (fraction that are true latent edges). "
             "AUC/R@k are over all unobserved candidates._")

    (EXP / "results.md").write_text("\n".join(L) + "\n")
    print(f"\n[experiments] {time.time() - t0:.0f}s total")
    print(f"[experiments] wrote experiments/results.json + experiments/results.md")
    failed = [n for n, ok in steps if not ok]
    if failed:
        print(f"[experiments] FAILED steps: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
