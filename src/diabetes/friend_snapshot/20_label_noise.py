"""20 - Label-noise sensitivity (validation only). Suppose a fraction pi of diabetes cases in the
low-access group were never diagnosed, so their label says 0. Reveal them and ask: how many does
the original model catch, and what changes if we retrain on the revealed labels?"""
import argparse
import config  # noqa: F401
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from data_io import load, features, TARGET
from models import fit_predict
import fsets

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mlp"); ap.add_argument("--feature-set", default="all_21")
ap.add_argument("--seeds", type=int, default=5)
a = ap.parse_args()
PIS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
ACCESS = ["AnyHealthcare", "NoDocbcCost", "CholCheck"]
df, tr, va, _ = load()
cols, y = fsets.get(a.feature_set), df[TARGET].to_numpy()
low = ((df.AnyHealthcare == 0) | (df.NoDocbcCost == 1) | (df.CholCheck == 0)).to_numpy()

# Who is plausibly an undiagnosed case? Score clinical profiles with a model fitted where access is good.
clin = [c for c in features(df) if c not in ACCESS]
good = tr[~low[tr]]
r = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(
    df.loc[good, clin], y[good]).predict_proba(df[clin])[:, 1]


def reveal(idx, pi, rng):
    """Labels for rows idx after revealing hidden cases in the low-access group: flip
    n = (low-access positives) * pi / (1 - pi) negatives to 1, chosen with probability proportional to r."""
    yy = y[idx].copy()
    cand = np.flatnonzero(low[idx] & (yy == 0))
    n = int(round(yy[low[idx]].sum() * pi / (1 - pi)))
    if n:
        w = r[idx][cand]
        yy[rng.choice(cand, n, replace=False, p=w / w.sum())] = 1
    return yy


Xtr, Xva = df.loc[tr, cols].to_numpy(np.float32), df.loc[va, cols].to_numpy(np.float32)
p_orig = fit_predict(a.model, Xtr, y[tr], Xva, cols, "none", seed=0)[0]
lv, rows = low[va], []
for pi in PIS:
    for seed in range(a.seeds if pi > 0 else 1):
        y_tr = reveal(tr, pi, np.random.default_rng(seed))
        y_va = reveal(va, pi, np.random.default_rng(100 + seed))
        p_new = p_orig if pi == 0 else fit_predict(a.model, Xtr, y_tr, Xva, cols, "none", seed=seed)[0]
        for name, p in [("original", p_orig), ("retrained", p_new)]:
            flag = p >= np.quantile(p, 0.70)                     # the same 30% screening budget
            for grp, m in [("low access", lv), ("other", ~lv)]:
                rows.append({"pi": pi, "seed": seed, "model": name, "group": grp,
                             "recall": flag[m & (y_va == 1)].mean(), "flag_rate": flag[m].mean()})
res = pd.DataFrame(rows)
res.to_csv(f"results/label_noise_{a.model}.csv", index=False)
summ = res.groupby(["group", "model", "pi"]).agg(recall=("recall", "mean"), sd=("recall", "std"),
                                                 flag_rate=("flag_rate", "mean")).round(3)
print(summ.unstack("pi")["recall"].to_string())
print(summ.unstack("pi")["flag_rate"].to_string())

fig, ax = plt.subplots(figsize=(7, 4.2))
for (grp, name), g in res.groupby(["group", "model"]):
    s = g.groupby("pi").recall.agg(["mean", "std"]).fillna(0)
    ax.errorbar(s.index, s["mean"], yerr=s["std"], marker="o", capsize=3,
                ls="-" if name == "original" else "--", label=f"{grp}, {name} model")
ax.set_xlabel("Assumed undiagnosed share of cases in the low-access group (pi)")
ax.set_ylabel("Recall on revealed labels\n(30% screening budget)"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("figures/fig_equity_label_noise.png", dpi=300)
