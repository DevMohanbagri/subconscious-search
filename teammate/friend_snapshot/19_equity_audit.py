"""19 - Equity audit: does the system behave differently by sex, age, income and healthcare access?
    python 19_equity_audit.py                                         (validation, during development)
    python 19_equity_audit.py --preds results/preds/test_final.csv    (only after 99_final_test.py)"""
import argparse, json
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, average_precision_score
from data_io import load, features, TARGET
from metrics import wilson_ci
from models import load_predictor

ap = argparse.ArgumentParser(); ap.add_argument("--preds", default="results/preds/val_final.csv")
a = ap.parse_args()
v = pd.read_csv(a.preds)
y, p = v.y_true.to_numpy(), v.p_cal.to_numpy()
thr = json.load(open("models/deployed.json"))["threshold_30pct"]        # chosen on validation, never on test
flag = p >= thr
GROUPS = {"Sex": np.where(v.Sex == 1, "male", "female"),
          "Age": np.select([v.Age <= 5, v.Age <= 9], ["18-44", "45-64"], "65+"),
          "Income": np.select([v.Income <= 4, v.Income <= 6], ["under $25k", "$25-50k"], "$50k+"),
          "Coverage": np.where(v.AnyHealthcare == 1, "insured", "uninsured"),
          "Cost barrier": np.where(v.NoDocbcCost == 1, "skipped doctor", "none"),
          "Cholesterol check": np.where(v.CholCheck == 1, "within 5 years", "not checked"),
          "Access": np.where((v.AnyHealthcare == 0) | (v.NoDocbcCost == 1) | (v.CholCheck == 0), "low", "other")}

# 1. subgroup table
rows = []
for g, lev in GROUPS.items():
    for level in pd.unique(lev):
        m = lev == level; pos = m & (y == 1)
        rlo, rhi = wilson_ci(int(flag[pos].sum()), int(pos.sum()))
        olo, ohi = wilson_ci(int(y[m].sum()), int(m.sum()))
        rows.append({"group": g, "level": level, "n": int(m.sum()), "prevalence": y[m].mean(),
                     "mean_P": p[m].mean(), "O/E": y[m].mean() / p[m].mean(),
                     "O/E_ci": f"{olo / p[m].mean():.2f}-{ohi / p[m].mean():.2f}",
                     "roc_auc": roc_auc_score(y[m], p[m]), "pr_auc": average_precision_score(y[m], p[m]),
                     "flag_rate": flag[m].mean(), "recall": flag[pos].mean(), "recall_ci": f"{rlo:.3f}-{rhi:.3f}",
                     "fpr": flag[m & (y == 0)].mean()})
tab = pd.DataFrame(rows)
tab.round(4).to_csv(a.preds.replace(".csv", "_equity.csv").replace("preds/", ""), index=False)
print(tab.round(3).to_string(index=False))

# 2. mechanism: odds ratios of the access variables, adjusted for everything else (training rows)
df, tr, va, _ = load()
X = sm.add_constant(df.loc[tr, features(df)].astype(float))
fit = sm.Logit(df.loc[tr, TARGET], X).fit(disp=0)
orr = pd.DataFrame({"odds_ratio": np.exp(fit.params), "lo": np.exp(fit.conf_int()[0]), "hi": np.exp(fit.conf_int()[1])})
print(orr.loc[["AnyHealthcare", "NoDocbcCost", "CholCheck", "Income"]].round(3).to_string())

# 3. access flip: score each person as if they had access; who would newly be flagged?
predict = load_predictor()
for col, frm, to in [("CholCheck", 0, 1), ("AnyHealthcare", 0, 1), ("NoDocbcCost", 1, 0)]:
    m = (v[col] == frm).to_numpy()
    p2 = predict(v.loc[m].assign(**{col: to}))
    print(f"{col} {frm}->{to}: n={m.sum()}  mean P {p[m].mean():.3f} -> {p2.mean():.3f}  "
          f"flagged {flag[m].mean():.1%} -> {(p2 >= thr).mean():.1%}")

fig, ax = plt.subplots(figsize=(8, 5))
tab["label"] = tab.group + ": " + tab.level
lo = tab.recall - tab.recall_ci.str.split("-").str[0].astype(float)
hi = tab.recall_ci.str.split("-").str[1].astype(float) - tab.recall
ax.errorbar(tab.recall, range(len(tab)), xerr=[lo, hi], fmt="o", capsize=3)
ax.set_yticks(range(len(tab)), tab.label); ax.invert_yaxis()
ax.axvline(flag[y == 1].mean(), ls=":", color="0.5")
ax.set_xlabel("Recall at the 30% screening budget (share of diabetics flagged)")
fig.tight_layout(); fig.savefig("figures/fig_equity_recall.png", dpi=300)
