#!/usr/bin/env python3
"""Independent validation of the RecBole LightGCN pipeline (checklist item 2).

Re-derives everything from artifacts + raw data, without re-running training:
  1. ID mapping: raw ML-100k id ranges vs embedding table sizes and split
     pair ids (RecBole remaps to contiguous 1..N in first-occurrence order;
     index 0 unused).
  2. Split audit: from outputs/recsys_split_pairs.json + raw timestamps -
     disjointness, coverage of all 100k pairs, per-user 80/10/10 sizes,
     and the chronology invariant (no future leak).
  3. Metric recomputation: full top-10 ranking per test/valid user from the
     SAVED embeddings, compared against recsys_metrics.json (NDCG@10,
     Recall@10, MRR@10). A match proves embeddings, ID mapping, split and
     reported metrics are all mutually consistent.
  4. Serving mask: known_train[u,i] iff (u,i) in train pairs.
  5. Serving API (if the service is up): /recommend excludes the user's
     train interactions and matches a local masked recomputation; titles
     resolve for every recommended item.

Exit 0 = all checks pass.
"""
import io
import json
import math
import sys
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
RAW_ZIP = ROOT / "data" / "ml-100k-raw.zip"
K = 10
API = "http://127.0.0.1:8620"
FAILURES = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def raw_data():
    pairs, ts = set(), {}
    with zipfile.ZipFile(RAW_ZIP) as zf:
        with zf.open("ml-100k/u.data") as f:
            for line in io.TextIOWrapper(f, encoding="latin-1"):
                p = line.split("\t")
                u, i, t = int(p[0]), int(p[1]), int(p[3])
                pairs.add((u, i))
                ts[(u, i)] = t
    return pairs, ts


def topk_metrics(user_scores: dict, positives: list, k: int = K):
    """RecBole full-sort metrics for one user (positives may be >1)."""
    P = set(positives)
    order = sorted(user_scores.items(), key=lambda x: (-x[1], x[0]))
    topk = [i for i, _ in order[:k]]
    hits = [i for i in topk if i in P]
    recall = len(hits) / len(P)
    dcg = sum(1.0 / math.log2(r + 2) for r, i in enumerate(topk) if i in P)
    idcg = sum(1.0 / math.log2(r + 2) for r in range(min(len(P), k)))
    ndcg = dcg / idcg if idcg else 0.0
    mrr = 0.0
    for rank, (i, _) in enumerate(order, start=1):
        if i in P:
            mrr = 1.0 / rank
            break
    return ndcg, recall, mrr


def main():
    import torch

    emb = torch.load(OUT / "recsys_embeddings.pt", map_location="cpu", weights_only=True)
    user_emb, item_emb = emb["user_emb"], emb["item_emb"]
    known = torch.load(OUT / "recsys_known_train.pt", map_location="cpu", weights_only=True)
    metrics = json.loads((OUT / "recsys_metrics.json").read_text())
    pairs = json.loads((OUT / "recsys_split_pairs.json").read_text())
    manifest = json.loads((OUT / "recsys_split_manifest.json").read_text())
    titles = json.loads((OUT / "recsys_item_titles.json").read_text())
    id_map = json.loads((OUT / "recsys_id_map.json").read_text())
    u2r = {int(k): v for k, v in id_map["user_raw_to_remapped"].items()}
    i2r = {int(k): v for k, v in id_map["item_raw_to_remapped"].items()}
    r2i = {v: k for k, v in i2r.items()}
    raw_pairs, raw_ts = raw_data()

    print("== 1. ID mapping ==")
    max_u = max(u for u, _ in raw_pairs)
    max_i = max(i for _, i in raw_pairs)
    print(f"raw: users 1..{max_u} ({max_u}), items 1..{max_i} ({max_i})")
    print(f"emb: user_emb {tuple(user_emb.shape)}, item_emb {tuple(item_emb.shape)}")
    check("user table size == max_raw_user+1 (1-based, idx0 unused)",
          user_emb.shape[0] == max_u + 1 and user_emb.shape[0] == 944)
    check("item table size == max_raw_item+1",
          item_emb.shape[0] == max_i + 1 and item_emb.shape[0] == 1683)
    check("known mask matches table sizes", known.shape == (max_u + 1, max_i + 1))
    bad = [(u, i) for name in pairs for u, i in pairs[name]
           if not (1 <= u <= max_u and 1 <= i <= max_i)]
    check("all split pair ids inside raw ranges", not bad, f"first bad: {bad[:3]}")

    print("\n== 2. Split audit ==")
    n_tr, n_va, n_te = len(pairs["train"]), len(pairs["valid"]), len(pairs["test"])
    print(f"sizes: train={n_tr} valid={n_va} test={n_te} total={n_tr + n_va + n_te}")
    check("sizes == 80808/9596/9596", (n_tr, n_va, n_te) == (80808, 9596, 9596))
    check("total == 100000 raw pairs", n_tr + n_va + n_te == len(raw_pairs))

    seen = defaultdict(int)
    for name in ("train", "valid", "test"):
        for u, i in pairs[name]:
            seen[(u, i)] += 1
    dup = {k: v for k, v in seen.items() if v > 1}
    check("no pair in two splits", not dup, f"first dup: {list(dup)[:3]}")
    missing = raw_pairs - set(seen)
    extra = set(seen) - raw_pairs
    check("split pairs cover all raw pairs", not missing and not extra,
          f"missing={len(missing)} extra={len(extra)}")

    violations = 0
    per_u_ts = {}
    for u in {u for u, _ in raw_pairs}:
        ts_tr = [raw_ts[(u, i)] for u2, i in pairs["train"] if u2 == u]
        ts_va = [raw_ts[(u, i)] for u2, i in pairs["valid"] if u2 == u]
        ts_te = [raw_ts[(u, i)] for u2, i in pairs["test"] if u2 == u]
        per_u_ts[u] = (ts_tr, ts_va, ts_te)
        if (ts_tr and ts_va and max(ts_tr) > min(ts_va)) or \
           (ts_tr and ts_te and max(ts_tr) > min(ts_te)) or \
           (ts_va and ts_te and max(ts_va) > min(ts_te)):
            violations += 1
    check("chronology invariant: 0 violations", violations == 0, f"violations={violations}")

    size_bad = 0
    for u, (ts_tr, ts_va, ts_te) in per_u_ts.items():
        tot = len(ts_tr) + len(ts_va) + len(ts_te)
        exp_va, exp_te = int(0.1 * tot), int(0.1 * tot)
        if not (len(ts_va) == exp_va and len(ts_te) == exp_te
                and len(ts_tr) == tot - exp_va - exp_te):
            size_bad += 1
    check("per-user sizes follow 80/10/10 (floor rule)", size_bad == 0, f"bad_users={size_bad}")
    check("manifest reports 0 invariant violations", manifest["invariant_violations"] == 0)
    check("manifest users == split users == 943",
          len(manifest["users"]) == len(per_u_ts) == 943)

    print("\n== 3. Metric recomputation from saved embeddings ==")
    # RecBole full-sort semantics (trainer._full_sort_batch_eval +
    # FullSortEvalDataLoader + Sampler.get_used_ids):
    #   pool = all items - item 0 - user's history items, where
    #   history = used_ids[phase] - positive, and used_ids accumulates
    #   across phases: valid -> train items; test -> train+valid items.
    # embeddings are in REMAPPED space; pairs file is RAW - convert
    ue = user_emb[1:]  # drop unused row 0; remapped id r -> row r-1
    ie = item_emb[1:]
    hist_valid = defaultdict(set)  # remapped items to mask per user
    hist_test = defaultdict(set)
    for u, i in pairs["train"]:
        hist_valid[u].add(i2r[i])
        hist_test[u].add(i2r[i])
    for u, i in pairs["valid"]:
        hist_test[u].add(i2r[i])
    for part in ("valid", "test"):
        pos = defaultdict(list)
        for u, i in pairs[part]:
            pos[u].append(i)
        masked = hist_valid if part == "valid" else hist_test
        sums = [0.0, 0.0, 0.0]
        n = 0
        for u, items in pos.items():
            rem_scores = (ue[u2r[u] - 1] @ ie.T).tolist()
            # rank in REMAPPED space (item 0 excluded by ie), mask history
            order = sorted(
                ((-float("inf") if r in masked[u] else s, r)
                 for r, s in enumerate(rem_scores, start=1)),
                key=lambda x: (-x[0], x[1]))
            topk_rem = [r for s, r in order[:K] if s > float("-inf")]
            pos_rem = {i2r[i] for i in items}
            hits = [r for r in topk_rem if r in pos_rem]
            recall = len(hits) / len(pos_rem)
            dcg = sum(1.0 / math.log2(rk + 2) for rk, r in enumerate(topk_rem) if r in pos_rem)
            idcg = sum(1.0 / math.log2(rk + 2) for rk in range(min(len(pos_rem), K)))
            ndcg = dcg / idcg if idcg else 0.0
            mrr = 0.0
            for rank, (s, r) in enumerate(order, start=1):
                if rank > K:  # RecBole MRR@k: no positive in top-k -> 0
                    break
                if s > float("-inf") and r in pos_rem:
                    mrr = 1.0 / rank
                    break
            sums[0] += ndcg
            sums[1] += recall
            sums[2] += mrr
            n += 1
        rec = {
            "NDCG@10": round(sums[0] / n, 4),
            "Recall@10": round(sums[1] / n, 4),
            "MRR@10": round(sums[2] / n, 4),
        }
        rep = metrics[part]
        print(f"{part}: recomputed={rec} reported={rep}")
        check(f"{part} metrics match reported (<=1e-4)",
              all(abs(rec[k] - rep[k]) <= 1e-4 for k in rec),
              f"max diff {max(abs(rec[k] - rep[k]) for k in rec):.6f}")

    print("\n== 4. Serving mask ==")
    kt = known.tolist()
    # known mask is REMAPPED space; train pairs are RAW - convert to compare
    mask_rem = {(u, i) for u in range(1, len(kt)) for i in range(1, len(kt[0])) if kt[u][i]}
    train_rem = {(u2r[u], i2r[i]) for u, i in pairs["train"]}
    check("known mask == train split pairs (id-converted)", mask_rem == train_rem,
          f"mask={len(mask_rem)} train={len(train_rem)}")
    check("id map is a bijection on raw ranges",
          sorted(u2r.values()) == list(range(1, 944)) and sorted(i2r.values()) == list(range(1, 1683)))

    print("\n== 5. Serving API ==")
    try:
        with urllib.request.urlopen(f"{API}/recommend?user_id=196&k={K}", timeout=15) as r:
            api = json.loads(r.read())
        rec_items = [x["item_id"] for x in api["recommendations"]]
        user_known = {i for u, i in pairs["train"] if u == 196}
        leaked = [i for i in rec_items if i in user_known]
        check("API: no recommended item is in user 196's TRAIN interactions",
              not leaked, f"leaked={leaked[:5]}")
        # local masked recomputation (remapped space), converted back to raw
        rem_scores = (ue[u2r[196] - 1] @ ie.T).tolist()
        mask_rem_row = [kt[u2r[196]][i] for i in range(1, len(kt[0]))]
        order = sorted(
            ((s if not m else float("-inf"), r) for r, (s, m) in enumerate(zip(rem_scores, mask_rem_row), start=1)),
            key=lambda x: (-x[0], x[1]))
        local_top = [r2i[r] for s, r in order[:K] if s > float("-inf")]
        check("API: /recommend matches local masked recomputation", rec_items == local_top,
              f"api={rec_items[:5]}... local={local_top[:5]}...")
        no_title = [i for i in rec_items if not titles.get(str(i))]
        check("API: titles resolve for all recommended items", not no_title,
              f"missing={no_title[:5]}")
        sample = api["recommendations"][0]
        print(f"sample: user 196 -> item {sample['item_id']} "
              f"{sample.get('title', '?')!r} (score {sample['score']:.4f})")
    except Exception as e:  # noqa: BLE001
        check("API reachable", False, f"{type(e).__name__}: {e}")

    print()
    if FAILURES:
        print(f"RESULT: FAIL ({len(FAILURES)}): {FAILURES}")
        sys.exit(1)
    print("RESULT: PASS - all validation checks green")


if __name__ == "__main__":
    main()
