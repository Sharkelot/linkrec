#!/usr/bin/env python3
"""Multi-seed statistical validation of the linkrec benchmark.

Runs the full pipeline for each seed in SEEDS (seed 42 last, so the
final served artifacts are the canonical seed-42 run) and aggregates
mean/std of the headline metrics. Writes:
  experiments/multi_seed.json   per-seed raw metrics + aggregates
  experiments/multi_seed.md     summary table

Headline metrics:
  Cora linkpred:  GraphSAGE auc_uniform, mrr_uniform, auc_hard, recall@10
                  + both baselines (auc_uniform, recall@10)
  B2B (observable): per model auc_full_ranking, precision@10, precision@100
  Recsys (chronological): LightGCN + PopRec NDCG@10 / Recall@10 / MRR@10 (test)
"""
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
EXP = ROOT / "experiments"
PY = sys.executable
SEEDS = [7, 123, 2024, 99, 42]
STEPS = [
    ("build_b2b", "build_b2b.py"),
    ("train_b2b", "train_b2b.py"),
    ("train_linkpred", "train_linkpred.py"),
    ("cora_baselines", "train_cora_baselines.py"),
    ("train_recsys", "train_recsys.py"),
    ("poprec", "train_poprec.py"),
]


def run_step(seed, name, script, log):
    env = {**os.environ, "LINKREC_SEED": str(seed)}
    t0 = time.time()
    with open(log, "w") as f:
        r = subprocess.run([PY, str(ROOT / "scripts" / script)],
                           stdout=f, stderr=subprocess.STDOUT,
                           env=env, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"[multi_seed] seed={seed} step={name} failed "
                         f"(log: {log})")
    print(f"  [{seed}] {name:15s} ok in {time.time() - t0:.0f}s", flush=True)


def load(p):
    return json.loads((OUT / p).read_text())


def collect(seed):
    lp = load("linkpred_metrics.json")
    cb = load("cora_baselines.json")["models"]
    b2b = load("b2b_metrics.json")["models"]
    rc = load("recsys_metrics.json")
    pr = load("poprec_metrics.json")
    return {
        "cora_graphsage": {
            "auc_uniform": lp["auc_uniform"], "mrr_uniform": lp["mrr_uniform"],
            "auc_hard": lp["auc_hard"], "recall@10": lp["recall@10"],
        },
        "cora_feature_cosine": {"auc_uniform": cb["feature_cosine"]["auc_uniform"],
                                "recall@10": cb["feature_cosine"]["recall@10"]},
        "cora_adamic_adaar": {"auc_uniform": cb["adamic_adaar"]["auc_uniform"],
                              "recall@10": cb["adamic_adaar"]["recall@10"]},
        **{f"b2b_{m}": {"auc": v["auc_full_ranking"], "p@10": v["precision@10"],
                        "p@100": v["precision@100"]}
           for m, v in b2b.items()},
        "recsys_lightgcn": {k: rc["test"][k] for k in ("NDCG@10", "Recall@10", "MRR@10")},
        "recsys_poprec": {k: pr["test"][k] for k in ("NDCG@10", "Recall@10", "MRR@10")},
    }


def aggregate(per_seed):
    models = list(per_seed.values())[0].keys()
    agg = {}
    for m in models:
        rows = [per_seed[s][m] for s in per_seed]
        for k in rows[0]:
            vals = [r[k] for r in rows]
            agg.setdefault(m, {})[k] = {
                "mean": round(statistics.fmean(vals), 4),
                "std": round(statistics.pstdev(vals), 4),
                "per_seed": [round(v, 4) for v in vals],
            }
    return agg


def main():
    EXP.mkdir(exist_ok=True)
    acc_path = EXP / "multi_seed.json"
    per_seed = {}
    if acc_path.exists():
        per_seed = {int(k): v for k, v in
                    json.loads(acc_path.read_text()).get("per_seed", {}).items()}
    for seed in SEEDS:
        if seed in per_seed:
            print(f"  [{seed}] already done, skipping", flush=True)
            continue
        print(f"=== seed {seed} ===", flush=True)
        for name, script in STEPS:
            run_step(seed, name, script, EXP / f"multi_seed_{name}_{seed}.log")
        per_seed[seed] = collect(seed)
        acc_path.write_text(json.dumps({"per_seed": per_seed}, indent=2))

    agg = aggregate(per_seed)
    (EXP / "multi_seed.json").write_text(json.dumps(
        {"seeds": SEEDS, "per_seed": per_seed, "aggregate": agg}, indent=2))

    lines = ["# Multi-seed benchmark (linkrec)", "",
             f"Seeds: {SEEDS} (42 last = canonical artifacts). "
             "mean (std) per metric.", ""]
    models = list(agg.keys())
    lines.append("| metric | " + " | ".join(models) + " |")
    lines.append("|" + "---|" * (len(models) + 1))
    rows = []  # union of metric names across all models, first-seen order
    for m in models:
        for k in agg[m]:
            if k not in rows:
                rows.append(k)
    for k in rows:
        row = [f"**{k}**"]
        for m in models:
            v = agg[m].get(k)
            row.append(f"{v['mean']:.4f} ({v['std']:.4f})" if v else "-")
        lines.append("| " + " | ".join(row) + " |")
    (EXP / "multi_seed.md").write_text("\n".join(lines) + "\n")
    print("\n=== multi-seed aggregate ===")
    for m, ks in agg.items():
        for k, v in ks.items():
            print(f"  {m:24s} {k:10s} mean={v['mean']:.4f} std={v['std']:.4f} "
                  f"seeds={[x for x in v['per_seed']]}")
    print(f"\n[saved] {acc_path} + {EXP / 'multi_seed.md'}")


if __name__ == "__main__":
    main()
