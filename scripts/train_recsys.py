#!/usr/bin/env python3
"""RecBole 1.2.0 LightGCN training on REAL ML-100k with a CHRONOLOGICAL split.

Data provenance (verified 2026-11-16):
  RecBole 1.2.0 hard-codes the dataset NAME 'ml-100k' to its bundled
  `dataset_example/ml-100k/` dir (config/configurator.py::_set_default_parameters),
  which contains a SYNTHETIC 100k-row placeholder (88,308 rows absent from the
  real GroupLens data). This script therefore builds a real atomic dataset
  `data/ml100k/ml100k.inter` from the raw zip (scripts/prep_ml100k.py) and uses
  the dataset name 'ml100k' so the configured data_path is respected.
  Raw file verified: 100,000 rows, 943 users, 1682 items, no duplicate (u,i),
  min 20 ratings/user, ts range 874724710..893286638, rating distribution
  6110/11370/27145/34174/21201 (matches documented GroupLens stats).

Chronological split (verified by scripts/probe_split.py, RESULT: PASS):
  * per-user 80/10/10 split of timestamp-ordered interactions (order=TO,
    split=RS, group_by=user): each user's EARLIEST 80% trains, next 10%
    validates, LAST 10% tests -> no future information leaks into training.
  * TWO RecBole 1.2.0 pitfalls are neutralized:
      1. Interaction stores float columns as float32. Raw epoch timestamps
         (~8.8e8) quantize to 64-second steps, so the TO sort (sort key =
         quantized value) destroys true time order. Fix: normalize_all=True
         (timestamps in [0,1] do not collide in float32).
      2. Even normalized, a few sub-second ties remain at bucket boundaries.
         Fix: pre-sort inter_feat by the EXACT float64 timestamp before
         data_preparation; build()'s stable sort then preserves exact order
         inside every quantization bucket.
  * After splitting, this script re-derives per-user RAW-timestamp bounds and
    asserts the chronology invariant (max train ts <= min valid ts <=
    min test ts) for all 943 users, and saves the audit manifest to
    outputs/recsys_split_manifest.json.

Evaluation: full (topk) ranking at k=10 -> NDCG@10, Recall@10, MRR.
Early stopping on validation MRR@10.

Outputs (outputs/):
  recsys_embeddings.pt    - final user/item ranking embeddings (for serving)
  recsys_known_train.pt   - bool [users, items]: interactions used for training
  recsys_metrics.json     - val/test NDCG@10, Recall@10, MRR + train time
  recsys_item_titles.json - item_id -> movie title
  recsys_split_manifest.json - per-user split sizes + raw-ts bounds (audit)
"""
import json
import time
from collections import defaultdict
from pathlib import Path

from compat import install_shims, quiet_recbole_noise
install_shims()
quiet_recbole_noise()

import io
import zipfile

import numpy as np
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_model, get_trainer, init_seed

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
RAW_ZIP = ROOT / "data" / "ml-100k-raw.zip"

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG = {
    "model": "LightGCN",
    "dataset": "ml100k",  # NOT 'ml-100k' (hard-coded to bundled sample data)
    "data_path": str(ROOT / "data"),
    # sample.yaml fallback only loads user_id,item_id - load the full inter
    "load_col": {"inter": ["user_id", "item_id", "rating", "timestamp"]},
    # see module docstring: prevents float32 timestamp quantization in the
    # TO sort (Interaction stores float cols as float32)
    "normalize_all": True,
    # model
    "embedding_size": 64,
    "layer_num": 3,
    # CHRONOLOGICAL split, per user, by timestamp
    "eval_args": {
        "group_by": "user",
        "order": "TO",
        "split": {"RS": [0.8, 0.1, 0.1]},
        "mode": "full",
    },
    # full ranking evaluation at k=10
    "topk_weights": [10],
    "metrics": {"NDCG": [10], "Recall": [10], "MRR": [10]},
    "metric_values_num": [10],
    "valid_metric": "MRR@10",
    # training
    "epochs": 300,
    "train_batch_size": 2048,
    "learning_rate": 1e-3,
    "eval_step": 5,
    "stopping_step": 30,
    "seed": SEED,
    "reproducibility": True,
    "device": DEVICE,
    "show_progress": False,
}


def raw_timestamps() -> dict:
    """(user, item) -> raw epoch timestamp, from the raw GroupLens zip."""
    ts = {}
    with zipfile.ZipFile(RAW_ZIP) as zf:
        with zf.open("ml-100k/u.data") as f:
            for line in io.TextIOWrapper(f, encoding="latin-1"):
                p = line.split("\t")
                ts[(int(p[0]), int(p[1]))] = int(p[3])
    return ts


def audit_splits(config, dataset, train_data, valid_data, test_data):
    """Invert RecBole's ID remap, re-derive RAW-timestamp bounds per user,
    assert the chronology invariant, and save the audit manifest."""
    uid_tok = dataset.field2id_token[config["USER_ID_FIELD"]]
    iid_tok = dataset.field2id_token[config["ITEM_ID_FIELD"]]
    ts_map = raw_timestamps()

    manifest = {"users": {}, "invariant_violations": 0}
    per_user = defaultdict(lambda: {"train": [], "valid": [], "test": []})
    for name, dl in (("train", train_data), ("valid", valid_data), ("test", test_data)):
        feat = dl._dataset.inter_feat
        nu = feat[config["USER_ID_FIELD"]].numpy()
        ni = feat[config["ITEM_ID_FIELD"]].numpy()
        ru = np.array([int(x) for x in uid_tok[nu]])
        ri = np.array([int(x) for x in iid_tok[ni]])
        for u, i in zip(ru.tolist(), ri.tolist()):
            per_user[u][name].append(i)

    # persist the actual split pairs (raw ids) for independent re-evaluation
    pairs = {name: [[u, i] for u in sorted(per_user) for i in per_user[u][name]]
             for name in ("train", "valid", "test")}
    (OUT / "recsys_split_pairs.json").write_text(json.dumps(pairs))

    # RecBole remaps ids to 1..N in first-occurrence order. Persist the
    # raw<->remapped maps so the serving API and validators can convert:
    # embeddings/known-mask live in REMAPPED space, the API speaks RAW ids.
    id_map = {
        "user_raw_to_remapped": {str(k): int(v) for k, v in dataset.field2token_id[config["USER_ID_FIELD"]].items() if k != "[PAD]"},
        "item_raw_to_remapped": {str(k): int(v) for k, v in dataset.field2token_id[config["ITEM_ID_FIELD"]].items() if k != "[PAD]"},
    }
    (OUT / "recsys_id_map.json").write_text(json.dumps(id_map))
    print(f"[audit] id map saved: {len(id_map['user_raw_to_remapped'])} users, "
          f"{len(id_map['item_raw_to_remapped'])} items")

    for u in sorted(per_user):
        parts = per_user[u]
        bounds = {}
        for name in ("train", "valid", "test"):
            items = parts[name]
            ts = [ts_map[(u, i)] for i in items]
            bounds[name] = {
                "n": len(items),
                "min_ts": min(ts),
                "max_ts": max(ts),
            }
        ok = (
            bounds["train"]["max_ts"] <= bounds["valid"]["min_ts"]
            and bounds["train"]["max_ts"] <= bounds["test"]["min_ts"]
            and bounds["valid"]["max_ts"] <= bounds["test"]["min_ts"]
        )
        if not ok:
            manifest["invariant_violations"] += 1
        manifest["users"][str(u)] = {"bounds": bounds, "invariant_ok": ok}

    (OUT / "recsys_split_manifest.json").write_text(json.dumps(manifest, indent=1))
    n_users = len(manifest["users"])
    print(f"[audit] users={n_users} chronology invariant violations="
          f"{manifest['invariant_violations']} -> manifest saved")
    assert manifest["invariant_violations"] == 0, (
        "chronological split invariant violated - refusing to train on a leaking split"
    )
    return manifest


def main():
    t0 = time.time()
    init_seed(SEED, True)
    print(f"[config] model=LightGCN device={DEVICE} split=TO/RS[0.8,0.1,0.1] group_by=user")

    config = Config(model="LightGCN", dataset="ml100k", config_dict=CONFIG)
    dataset = create_dataset(config)

    # Pre-sort by EXACT float64 timestamp. build()'s TO sort is stable but
    # sorts on a float32 (normalized) key; pre-ordering on the exact values
    # makes the final per-user order exactly timestamp-ordered.
    dataset.inter_feat = dataset.inter_feat.sort_values(
        "timestamp", kind="stable"
    ).reset_index(drop=True)

    train_data, valid_data, test_data = data_preparation(config, dataset)
    inter = dataset.inter_feat
    num_users = int(inter[config["USER_ID_FIELD"]].max()) + 1
    num_items = int(inter[config["ITEM_ID_FIELD"]].max()) + 1
    print(f"[data] users={num_users} items={num_items} "
          f"interactions: train={len(train_data._dataset.inter_feat)}, "
          f"valid={len(valid_data._dataset.inter_feat)}, test={len(test_data._dataset.inter_feat)}")

    audit_splits(config, dataset, train_data, valid_data, test_data)

    init_seed(SEED, True)
    model = get_model("LightGCN")(config, train_data._dataset).to(DEVICE)
    trainer = get_trainer(config["MODEL_TYPE"], "LightGCN")(config, model)

    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=False
    )
    print(f"\n=== Validation (chronological, top-10 full ranking) ===")
    print({k: round(v, 4) for k, v in best_valid_result.items()})

    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=False)
    print(f"\n=== Test (chronological, top-10 full ranking) ===")
    print({k: round(v, 4) for k, v in test_result.items()})

    # ---------- persist for serving ----------
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model.forward()
    torch.save({"user_emb": user_emb.cpu(), "item_emb": item_emb.cpu()},
               OUT / "recsys_embeddings.pt")

    # known interactions from the TRAIN split only (chronological: the first
    # 80% of each user's timestamp-ordered history)
    known = torch.zeros(num_users, num_items, dtype=torch.bool)
    ti = train_data._dataset.inter_feat
    known[ti[config["USER_ID_FIELD"]].long(), ti[config["ITEM_ID_FIELD"]].long()] = True
    torch.save(known.cpu(), OUT / "recsys_known_train.pt")

    # item titles from the raw zip (same zip the inter file was built from,
    # so item ids are consistent)
    titles = fetch_raw_titles()
    (OUT / "recsys_item_titles.json").write_text(json.dumps(titles))

    # RecBole remaps ids to 1..N in first-occurrence order here: users 1..943,
    # items 1..1682 (verified: loaded triples == raw triples, 100000/100000)
    def _canon(k: str) -> str:
        name, kpart = k.lower().split("@")
        return f"{name.upper() if name in ('ndcg', 'mrr') else name.capitalize()}@{kpart}"
    metrics = {
        "dataset": "ml-100k (real GroupLens data via data/ml100k/, verified against raw zip)",
        "model": "LightGCN(embedding_size=64, layer_num=3)",
        "split": "chronological: per-user 80/10/10 by raw timestamp (order=TO, RS, group_by=user); "
                 "audit manifest: outputs/recsys_split_manifest.json (0 invariant violations)",
        "evaluation": "full ranking (topk) at k=10",
        "valid": {_canon(k): round(float(v), 4) for k, v in best_valid_result.items()},
        "test": {_canon(k): round(float(v), 4) for k, v in test_result.items()},
        "train_seconds": round(time.time() - t0, 2),
        "seed": SEED,
    }
    (OUT / "recsys_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\n[saved] embeddings -> {OUT / 'recsys_embeddings.pt'}")
    print(f"[saved] metrics    -> {OUT / 'recsys_metrics.json'}")


def fetch_raw_titles() -> dict:
    """Item titles from the raw ML-100k zip (u.item)."""
    titles = {}
    with zipfile.ZipFile(RAW_ZIP) as zf:
        with zf.open("ml-100k/u.item") as f:
            for line in io.TextIOWrapper(f, encoding="latin-1"):
                parts = line.split("|")
                if len(parts) >= 2:
                    try:
                        titles[int(parts[0])] = parts[1].strip()
                    except ValueError:
                        pass
    return titles


if __name__ == "__main__":
    main()
