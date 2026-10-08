"""12 - Competing feature selectors (MI, L1, RFE) at matched k, and the PR-AUC-versus-k curve."""
import config  # noqa: F401
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.metrics import average_precision_score
from data_io import load, features, TARGET
import fsets

df, tr, va, _ = load()
ytr, yva = df.loc[tr, TARGET].to_numpy(), df.loc[va, TARGET].to_numpy()


def rankings(pool):
    """Three orderings of `pool`, each learned on training rows only."""
    X = StandardScaler().fit_transform(df.loc[tr, pool])
    mi = mutual_info_classif(df.loc[tr, pool], ytr, discrete_features=[c != "BMI" for c in pool], random_state=0)
    r_mi = [pool[i] for i in np.argsort(-mi)]
    entry = {}                                   # L1 path: the C at which each feature first turns on
    for C in np.logspace(-4, 0, 40):
        coef = LogisticRegression(l1_ratio=1.0, solver="liblinear", C=C).fit(X, ytr).coef_[0]
        for f, w in zip(pool, coef):
            if w != 0 and f not in entry:
                entry[f] = C
    r_l1 = sorted(pool, key=lambda f: entry.get(f, np.inf))
    rfe = RFE(LogisticRegression(max_iter=2000), n_features_to_select=1).fit(X, ytr)
    r_rfe = [pool[i] for i in np.argsort(rfe.ranking_)]
    return {"MI": r_mi, "L1": r_l1, "RFE": r_rfe}


def prauc(cols):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(df.loc[tr, cols], ytr)
    return average_precision_score(yva, m.predict_proba(df.loc[va, cols])[:, 1])


rows, ranks = [], {}
for pool_name, pool, rst_name in [("all_21", features(df), "rst_heldout"),
                                  ("leakage_free", features(df, drop_leakage=True), "rst_heldout_noleak")]:
    ranks[pool_name] = rankings(pool)
    k_rst = len(fsets.get(rst_name))
    suffix = "" if pool_name == "all_21" else "_noleak"
    for method, order in ranks[pool_name].items():
        fsets.save(f"{method.lower()}_k{k_rst}{suffix}", order[:k_rst], f"12_fs_comparators.py {method}", pool=pool_name)
        for k in range(1, len(pool) + 1):
            rows.append({"pool": pool_name, "method": method, "k": k, "pr_auc": round(prauc(order[:k]), 4)})
    print(pool_name, {m: o[:k_rst] for m, o in ranks[pool_name].items()})
curve = pd.DataFrame(rows)
curve.to_csv("results/fs_k_curve.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, pool_name, rst_names in [(axes[0], "all_21", ["rst_heldout"]),
                                 (axes[1], "leakage_free", ["rst_heldout_noleak"])]:
    for method, g in curve[curve.pool == pool_name].groupby("method"):
        ax.plot(g.k, g.pr_auc, "-", marker=".", label=method)
    for name in rst_names:
        cols = fsets.get(name)
        ax.scatter([len(cols)], [prauc(cols)], s=90, color="k", zorder=5, label=name)
    ax.axhline(yva.mean(), ls=":", color="0.5"); ax.set_title(pool_name); ax.set_xlabel("Number of features k")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True)); ax.legend(loc="lower right")
axes[0].set_ylabel("Validation PR-AUC (logistic regression)")
fig.tight_layout(); fig.savefig("figures/fig_fs_k_curve.png", dpi=300)
print(curve.pivot_table(index="k", columns=["pool", "method"], values="pr_auc").head(8).to_string())
