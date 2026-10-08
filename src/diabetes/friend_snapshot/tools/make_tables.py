"""Build the paper's main table from the registry: python tools/make_tables.py val   (or test)."""
import sys
import pandas as pd

split = sys.argv[1] if len(sys.argv) > 1 else "val"
r = pd.read_csv("results/registry.csv")
r = r[(r.split == split) & (r.script.isin(["16_main_grid.py", "99_final_test.py"]))]
if "--include-dirty" not in sys.argv:
    r = r[~r.git_commit.astype(str).str.endswith("-dirty")]            # only citable rows
t = r.groupby(["feature_set", "model"]).agg(k=("n_features", "first"), runs=("pr_auc", "size"),
                                            pr=("pr_auc", "mean"), sd=("pr_auc", "std"),
                                            roc=("roc_auc", "mean"), rec=("rec_at_p30", "mean"),
                                            brier=("brier", "mean")).reset_index()
t["PR-AUC"] = [f"{m:.4f} ± {0 if pd.isna(s) else s:.4f}" for m, s in zip(t.pr, t.sd)]
t = t[["feature_set", "model", "k", "runs", "PR-AUC", "roc", "rec", "brier"]].round(4)
t.to_csv(f"results/main_table_{split}.csv", index=False)
print(t.to_string(index=False))
