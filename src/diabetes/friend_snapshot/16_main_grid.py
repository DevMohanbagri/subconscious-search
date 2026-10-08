"""16 - The main results grid: feature sets x models x seeds, one strategy (chosen in Phase 5b)."""
import argparse
import config
import pandas as pd
from joblib import Parallel, delayed
from models import run_config

FEATURE_SETS = ["all_21", "leakage_free", "rst_heldout", "rst_heldout_noleak",
                "mi_k5", "l1_k5", "rfe_k5", "mi_k6_noleak", "l1_k6_noleak", "rfe_k6_noleak"]
ap = argparse.ArgumentParser()
ap.add_argument("--models", default="lr,lgbm,mlp"); ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--strategy", default="none"); ap.add_argument("--sets", default=",".join(FEATURE_SETS))
a = ap.parse_args()
cfgs = [(m, fs, seed) for m in a.models.split(",") for fs in a.sets.split(",")
        for seed in (range(a.seeds) if m != "lr" else [0])]          # logistic regression is deterministic
res = pd.DataFrame(Parallel(n_jobs=config.N_WORKERS)(
    delayed(run_config)(m, fs, a.strategy, seed, script="16_main_grid.py") for m, fs, seed in cfgs))
table = res.groupby(["feature_set", "model"]).agg(
    k=("feature_set", "size"), pr_auc=("pr_auc", "mean"), pr_sd=("pr_auc", "std"),
    roc_auc=("roc_auc", "mean"), rec_at_p30=("rec_at_p30", "mean"), brier=("brier", "mean")).round(4)
table["k"] = [len(__import__("fsets").get(fs)) for fs, _ in table.index]
table.to_csv("results/main_table_val.csv")
print(table.to_string())
