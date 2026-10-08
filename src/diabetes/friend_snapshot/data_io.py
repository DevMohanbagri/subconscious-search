import pandas as pd, numpy as np, hashlib, json
from pathlib import Path
import subprocess, datetime, csv, os

TARGET = "Diabetes_binary"
DEDUP = Path("data/dedup.csv")
SPLITS = Path("splits.npz")

LEAKAGE_SUSPECT = ["GenHlth", "DiffWalk", "PhysHlth"]


DATA_HASH = None


def load(allow_test=False):
    """Returns (df, train_idx, val_idx, test_idx or None). Asserts the data hash.
    Test indices are withheld unless allow_test=True (only 99_final_test.py passes it)."""
    df = pd.read_csv(DEDUP)
    s = np.load(SPLITS, allow_pickle=True)
    h = hashlib.md5(pd.util.hash_pandas_object(df, index=False).values).hexdigest()
    expected = str(s["data_hash"])
    assert h == expected, f"DATA HASH MISMATCH\n got {h}\n want {expected}"
    globals()["DATA_HASH"] = h
    return df, s["train"], s["val"], (s["test"] if allow_test else None)

def features(df, drop_leakage=False):
    cols = [c for c in df.columns if c != TARGET]
    if drop_leakage:
        cols = [c for c in cols if c not in LEAKAGE_SUSPECT]
    return cols
REGISTRY = Path("results/registry.csv")
FIELDS = ["timestamp", "git_commit", "data_hash", "machine", "script",
          "feature_set", "n_features", "model", "params", "seed", "split",
          "pr_auc", "roc_auc", "brier", "prevalence", "notes",
          # v2 columns, appended so old rows keep their meaning
          "accuracy", "rec_at_p30", "prec_at_r80", "ece", "threshold", "calibrated", "extra"]


def git_commit():
    try:
        h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "*.py"], text=True).strip()
        return h + ("-dirty" if dirty else "")   # -dirty = a .py file differs from that commit
    except Exception:
        return "nogit"


def log_result(**kw):
    """Append one row to the append-only results registry."""
    assert DATA_HASH, "call data_io.load() before logging, so every row carries a verified hash"
    unknown = set(kw) - set(FIELDS)
    assert not unknown, f"unknown registry fields: {unknown}"
    for k in ("params", "extra"):
        if isinstance(kw.get(k), dict):
            kw[k] = json.dumps(kw[k], sort_keys=True, default=str)
    row = {k: "" for k in FIELDS}
    row.update({"timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "git_commit": git_commit(), "data_hash": DATA_HASH,
                "machine": os.environ.get("COMPUTERNAME", "?")})
    row.update(kw)
    REGISTRY.parent.mkdir(exist_ok=True)
    new = not REGISTRY.exists()
    if not new:
        with REGISTRY.open(newline="") as f:
            header = next(csv.reader(f))
        assert header == FIELDS, "registry header != FIELDS: run tools/migrate_registry_v2.py once"
    with REGISTRY.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
