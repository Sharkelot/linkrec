#!/usr/bin/env python3
"""Popularity baseline for the ML-100k recommendation task.

Ranks items by their frequency in the TRAIN split only (no learning).
Evaluated with the EXACT same chronological split and the EXACT same
full-sort semantics as the RecBole LightGCN run (see validate_recsys.py):
  * pool: all items, user's history items masked (-inf)
      valid eval masks train items; test eval masks train+valid items
  * tie-break: item id ascending
  * NDCG@10 / Recall@10 / MRR@10 (MRR: first positive within top-10 else 0)
so the numbers are directly comparable with LightGCN.

Output: outputs/poprec_metrics.json
"""
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
K = 10


def topk_metrics(scores: dict, positives: set) -> tuple:
    order = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    topk = [i for i, _ in order[:K]]
    hits = [i for i in topk if i in positives]
    recall = len(hits) / len(positives) if positives else 0.0
    dcg = sum(1.0 / math.log2(r + 2) for r, i in enumerate(topk) if i in positives)
    idcg = sum(1.0 / math.log2(r + 2) for r in range(min(len(positives), K)))
    ndcg = dcg / idcg if idcg else 0.0
    mrr = 0.0
    for rank, (i, _) in enumerate(order, start=1):
        if rank > K:
            break
        if i in positives:
            mrr = 1.0 / rank
            break
    return ndcg, recall, mrr


def main():
    pairs = json.loads((OUT / "recsys_split_pairs.json").read_text())
    max_u = max(u for u, _ in pairs["train"])
    max_i = max(i for _, i in pairs["train"])

    freq = defaultdict(int)
    for _, i in pairs["train"]:
        freq[i] += 1
    n_items = max_i  # raw ids 1..max_i

    for part, mask_parts in (("valid", ("train",)), ("test", ("train", "valid"))):
        hist = defaultdict(set)
        for m in mask_parts:
            for u, i in pairs[m]:
                hist[u].add(i)
        pos = defaultdict(set)
        for u, i in pairs[part]:
            pos[u].add(i)
        s_ndcg = s_rec = s_mrr = 0.0
        for u in pos:
            scores = {i: freq.get(i, 0) for i in range(1, n_items + 1)}
            for h in hist[u]:
                scores[h] = float("-inf")
            ndcg, recall, mrr = topk_metrics(scores, pos[u])
            s_ndcg += ndcg
            s_rec += recall
            s_mrr += mrr
        n = len(pos)
        out = {
            "NDCG@10": round(s_ndcg / n, 4),
            "Recall@10": round(s_rec / n, 4),
            "MRR@10": round(s_mrr / n, 4),
        }
        print(f"[poprec] {part}: {out}")
        if part == "test":
            test_out = out
        else:
            valid_out = out

    (OUT / "poprec_metrics.json").write_text(json.dumps(
        {"model": "PopRec (item frequency in train split)",
         "valid": valid_out, "test": test_out}, indent=2))


if __name__ == "__main__":
    main()
