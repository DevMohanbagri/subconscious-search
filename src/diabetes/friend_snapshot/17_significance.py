"""17 - Is A really better than B? Paired bootstrap on PR-AUC, McNemar at a fixed 30% screening budget,
Holm correction across all comparisons. Reads saved validation predictions (seed 0)."""
import argparse
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from metrics import paired_bootstrap_delta

ap = argparse.ArgumentParser(); ap.add_argument("--model", default="mlp"); ap.add_argument("--strategy", default="none")
a = ap.parse_args()
PAIRS = [("rst_heldout", "all_21"), ("rst_heldout_noleak", "leakage_free"),
         ("rst_heldout", "mi_k5"), ("rst_heldout_noleak", "mi_k6_noleak"),
         ("rst_heldout_noleak", "l1_k6_noleak"), ("rst_heldout_noleak", "rfe_k6_noleak")]


def preds(fs, model=None):
    p = pd.read_csv(f"results/preds/val_{model or a.model}_{fs}_{a.strategy}_s0.csv")
    return p.y_true.to_numpy(), p.p_raw.to_numpy()


def mcnemar_budget(y, pa, pb, budget=0.30):
    """Flag the top 30% by each model; compare which true cases each one catches."""
    fa, fb = pa >= np.quantile(pa, 1 - budget), pb >= np.quantile(pb, 1 - budget)
    ca, cb = fa[y == 1], fb[y == 1]
    table = [[np.sum(ca & cb), np.sum(ca & ~cb)], [np.sum(~ca & cb), np.sum(~ca & ~cb)]]
    if table[0][1] + table[1][0] == 0:          # the two models flag exactly the same cases
        return 1.0, 0.0
    return mcnemar(table, exact=False, correction=True).pvalue, ca.mean() - cb.mean()


rows = []
for A, B in PAIRS:
    y, pa = preds(A); _, pb = preds(B)
    d, lo, hi, p_boot = paired_bootstrap_delta(average_precision_score, y, pa, pb, n_boot=2000)
    p_mc, d_recall = mcnemar_budget(y, pa, pb)
    rows.append({"A": A, "B": B, "delta_pr_auc": round(d, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                 "p_bootstrap": p_boot, "recall_diff_at_30pct": round(d_recall, 4), "p_mcnemar": p_mc})
res = pd.DataFrame(rows)
res["p_boot_holm"] = multipletests(res.p_bootstrap, method="holm")[1]
res["p_mcnemar_holm"] = multipletests(res.p_mcnemar, method="holm")[1]
res.round(4).to_csv(f"results/significance_{a.model}.csv", index=False)
print(res.round(4).to_string(index=False))
