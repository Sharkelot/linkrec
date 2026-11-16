#!/usr/bin/env python3
"""RecBole 1.2.0 recommendation training on ML-100k with a CHRONOLOGICAL split.

Follows the RecBole training/evaluation guide:
  * model: LightGCN (GNN collaborative filtering)
  * chronological (time-ordered) split via eval_args:
      order: TO          -> interactions sorted by timestamp
      split: {RS: [0.8, 0.1, 0.1]} + group_by: user
      -> each user's EARLIEST 80% of interactions train the model,
         next 10% validate, LAST 10% test. Test items are strictly
         later than training items: no future information leaks in.
  * full (topk) ranking evaluation, k=10 -> NDCG@10, Recall@10, MRR
  * early stopping on validation MRR@10

Outputs (outputs/):
  recsys_embeddings.pt   - final user/item ranking embeddings (for serving)
  recsys_known_train.pt  - bool [users, items]: interactions used for training
  recsys_metrics.json    - val/test NDCG@10, Recall@10, MRR + train time
  recsys_item_titles.json- item_id -> movie title
"""
import json
import time
from pathlib import Path

import numpy as np
# RecBole 1.2.0 compatibility_settings() touches np.float_/np.complex_/
# np.object_/np.unicode_, removed in NumPy 2.0 - restore the legacy aliases
# before any recbole config is built.
if not hasattr(np, "float_"):
    np.float_ = np.float64  # type: ignore[attr-defined]
    np.float = np.float64  # type: ignore[attr-defined]
if not hasattr(np, "complex_"):
    np.complex_ = np.complex128  # type: ignore[attr-defined]
    np.complex = np.complex128  # type: ignore[attr-defined]
if not hasattr(np, "object_"):
    np.object_ = object  # type: ignore[attr-defined]
    np.object = object  # type: ignore[attr-defined]
if not hasattr(np, "unicode_"):
    np.unicode_ = str  # type: ignore[attr-defined]
    np.unicode = str  # type: ignore[attr-defined]

# scipy >= 1.14 removed the private dok_matrix._update used by RecBole's
# LightGCN.get_norm_adj_mat - restore it as a thin wrapper around .update()
import scipy.sparse as sp
if not hasattr(sp.dok_matrix, "_update"):
    sp.dok_matrix._update = lambda self, other: self.update(other)  # type: ignore[attr-defined]

import torch
# torch 2.6+ defaults torch.load to weights_only=True; RecBole's trainer
# internally reloads its own (just-saved, trusted) checkpoint with torch.load.
# Relax the default for this process only.
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load  # type: ignore[assignment]
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_model, get_trainer, init_seed

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG = {
    "model": "LightGCN",
    "dataset": "ml-100k",
    "data_path": str(ROOT / "data"),
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


def main():
    t0 = time.time()
    init_seed(SEED, True)
    print(f"[config] model=LightGCN device={DEVICE} split=TO/RS[0.8,0.1,0.1] group_by=user")

    config = Config(model="LightGCN", dataset="ml-100k", config_dict=CONFIG)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    inter = dataset.inter_feat
    num_users = int(inter[config["USER_ID_FIELD"]].max()) + 1
    num_items = int(inter[config["ITEM_ID_FIELD"]].max()) + 1
    print(f"[data] users={num_users} items={num_items} "
          f"interactions: train={len(train_data._dataset.inter_feat)}, "
          f"valid={len(valid_data._dataset.inter_feat)}, test={len(test_data._dataset.inter_feat)}")

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

    # item titles: from processed dataset item feature file if present,
    # otherwise from raw ML-100k u.item (downloaded below)
    titles = {}
    item_file = ROOT / "data" / "ml-100k" / "item.csv"
    if item_file.exists():
        import csv
        with item_file.open() as f:
            rd = csv.DictReader(f)
            for row in rd:
                if "item_id" in row and "title" in row and row["item_id"]:
                    try:
                        titles[int(row["item_id"])] = row["title"].strip()
                    except (ValueError, KeyError):
                        pass
    if not titles:
        titles = fetch_raw_titles()
    (OUT / "recsys_item_titles.json").write_text(json.dumps(titles))

    # RecBole keeps the raw 1-based ML-100k IDs here: users 1..943, items 1..1682
    def _canon(k: str) -> str:
        name, kpart = k.lower().split("@")
        return f"{name.upper() if name in ('ndcg', 'mrr') else name.capitalize()}@{kpart}"
    metrics = {
        "dataset": "ml-100k",
        "model": "LightGCN(embedding_size=64, layer_num=3)",
        "split": "chronological TIME-ordered: per-user 80/10/10 by timestamp (order=TO)",
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
    """Download raw ML-100k from GroupLens and parse u.item titles."""
    import io
    import urllib.request
    import zipfile

    z = ROOT / "data" / "ml-100k-raw.zip"
    url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
    if not z.exists():
        print("[data] downloading raw ml-100k for item titles ...")
        urllib.request.urlretrieve(url, z)
    titles = {}
    with zipfile.ZipFile(z) as zf:
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
