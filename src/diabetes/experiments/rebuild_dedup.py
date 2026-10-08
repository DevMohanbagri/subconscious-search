#!/usr/bin/env python3
"""Rebuild the friend's data/dedup.csv WITHOUT UCI access (UCI is unreachable here).

Run from the friend's repo root AFTER `pip install numpy pandas scikit-learn`:
    python3 /home/user/subconscious-search/src/diabetes/experiments/rebuild_dedup.py

Recipe (reverse-engineered to reproduce the committed splits.npz data_hash
fc3774b0bdf6c0a16e80c7e518cd787d EXACTLY):
  source  our data/diabetes_012_health_indicators_BRFSS2015.csv (253,680 x 22,
          same BRFSS rows as UCI #891, only the target column differs)
  target  Diabetes_binary = (Diabetes_012 == 2)  [diabetes ONLY; the project
          doc's "diabetes or prediabetes" claim contradicts the bytes]
  dtypes  ALL 22 columns int64 (BMI included -- every value is whole; UCI's
          CSV stores them as ints and the md5 is dtype-sensitive)
  order   sorted(feature names) + [Diabetes_binary]  (== 00_fetch_data.py)
  dedup   drop_duplicates().reset_index(drop=True) -> 229,474 rows, prev 0.1529
The script asserts the hash against splits.npz before writing.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("/home/user/subconscious-search/data/diabetes_012_health_indicators_BRFSS2015.csv")
SPLITS = Path("splits.npz")
OUT = Path("data/dedup.csv")


def main():
    assert SPLITS.exists(), "run from the friend's repo root (splits.npz not found)"
    src = pd.read_csv(SRC)
    assert src.shape == (253680, 22), src.shape
    df = src.rename(columns={"Diabetes_012": "Diabetes_binary"})
    df["Diabetes_binary"] = (df["Diabetes_binary"] == 2).astype("int64")
    for c in df.columns:
        if c == "Diabetes_binary":
            continue
        v = df[c].to_numpy(float)
        assert np.all(v == v.astype("int64")), f"non-integer values in {c}"
        df[c] = v.astype("int64")
    feats = sorted(c for c in df.columns if c != "Diabetes_binary")
    df = df[feats + ["Diabetes_binary"]]
    df = df.drop_duplicates().reset_index(drop=True)
    assert len(df) == 229474, len(df)
    h = hashlib.md5(pd.util.hash_pandas_object(df, index=False).values).hexdigest()
    want = str(np.load(SPLITS, allow_pickle=True)["data_hash"])
    assert h == want, f"HASH-MISMATCH got {h} want {want}"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"OK: wrote {OUT} {df.shape} hash={h}")


if __name__ == "__main__":
    main()
