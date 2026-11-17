#!/usr/bin/env python3
"""Build the REAL ML-100k RecBole atomic dataset from the GroupLens raw zip.

Why: RecBole 1.2.0 hard-codes dataset name 'ml-100k' to its bundled
dataset_example dir (config/configurator.py::_set_default_parameters), which
contains a 100k-row SYNTHETIC placeholder (88,308 rows not present in the real
GroupLens data). We name our dataset 'ml100k' so the configured data_path is
respected, and build the atomic files from the authentic raw zip:

    data/ml-100k-raw.zip  (http://files.grouplens.org/datasets/movielens/ml-100k.zip)
      -> data/ml100k/inter   (user_id, item_id, rating, timestamp - raw values)

Idempotent: skips if the file already exists with the right row count.
"""
import io
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "ml-100k-raw.zip"
OUT = ROOT / "data" / "ml100k"


def main() -> None:
    out_file = OUT / "ml100k.inter"
    if out_file.exists():
        n = sum(1 for _ in out_file.open()) - 1
        if n == 100_000:
            print(f"[prep] {out_file} already present with {n} rows - skipping")
            return
    if not RAW.exists():
        import urllib.request

        print("[prep] downloading raw ml-100k zip ...")
        urllib.request.urlretrieve("http://files.grouplens.org/datasets/movielens/ml-100k.zip", RAW)

    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    users, items = set(), set()
    with zipfile.ZipFile(RAW) as zf:
        with zf.open("ml-100k/u.data") as f:
            lines = io.TextIOWrapper(f, encoding="latin-1").read().splitlines()
    # RecBole atomic files need a typed header (field:FeatureType)
    with out_file.open("w") as out:
        out.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\n")
        for line in lines:
            p = line.split("\t")
            uid, iid, rating, ts = int(p[0]), int(p[1]), int(p[2]), int(p[3])
            users.add(uid)
            items.add(iid)
            out.write(f"{uid}\t{iid}\t{rating}\t{ts}\n")
            n += 1
    print(f"[prep] wrote {out_file}: {n} rows, "
          f"users {min(users)}..{max(users)} ({len(users)}), "
          f"items {min(items)}..{max(items)} ({len(items)})")
    assert n == 100_000 and len(users) == 943 and len(items) == 1682, "unexpected ML-100k shape"


if __name__ == "__main__":
    main()
