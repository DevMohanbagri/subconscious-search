"""06 - EDA figures for the paper. Training rows only; never touch validation or test here."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif
from data_io import load, features, TARGET
from metrics import wilson_ci

df, tr, _, _ = load()
d, feats = df.loc[tr], features(df)
PREV = d[TARGET].mean()
FIG = Path("figures"); FIG.mkdir(exist_ok=True)
AGE = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
       "55-59", "60-64", "65-69", "70-74", "75-79", "80+"]
BINARY = [c for c in feats if d[c].nunique() == 2]


def save(fig, name):
    fig.tight_layout(); fig.savefig(FIG / name, dpi=300); plt.close(fig)


def rate_ci(group):  # prevalence and 95% Wilson interval per level
    g = d.groupby(group)[TARGET].agg(["sum", "size"])
    lo, hi = zip(*(wilson_ci(k, n) for k, n in zip(g["sum"], g["size"])))
    return g["sum"] / g["size"], np.array(lo), np.array(hi)


# 1. class-conditional rates: P(D | x=0) vs P(D | x=1) for the 14 binary features
r = pd.DataFrame({c: [d.loc[d[c] == 0, TARGET].mean(), d.loc[d[c] == 1, TARGET].mean()]
                  for c in BINARY}, index=["x0", "x1"]).T
r = r.assign(gap=r.x1 - r.x0).sort_values("gap")
fig, ax = plt.subplots(figsize=(7, 6))
ax.hlines(r.index, r.x0, r.x1, color="0.7")
ax.scatter(r.x0, r.index, label="feature = 0", color="0.4")
ax.scatter(r.x1, r.index, label="feature = 1", color="C3")
ax.axvline(PREV, ls="--", color="0.5", lw=1)
ax.set_xlabel("Diabetes prevalence in group"); ax.legend(loc="lower right")
save(fig, "fig_eda_class_conditional.png")

# 2. BMI by class, WHO cut points marked
fig, ax = plt.subplots(figsize=(7, 4))
for v, lab in [(0, "no diabetes"), (1, "diabetes")]:
    ax.hist(d.loc[d[TARGET] == v, "BMI"], bins=np.arange(12, 61), density=True, alpha=0.5, label=lab)
for cut in (18.5, 25, 30, 35):
    ax.axvline(cut, color="0.3", lw=0.8, ls=":")
ax.set_xlabel("BMI (values above 60 not shown)"); ax.set_ylabel("Density"); ax.legend()
save(fig, "fig_eda_bmi_by_class.png")

# 3. prevalence by age band with 95% intervals
p, lo, hi = rate_ci("Age")
fig, ax = plt.subplots(figsize=(7, 4))
ax.errorbar(range(13), p, yerr=[p - lo, hi - p], fmt="o-", capsize=3)
ax.set_xticks(range(13), AGE, rotation=45); ax.set_ylabel("Diabetes prevalence")
save(fig, "fig_eda_prevalence_by_age.png")

# 4. zero-inflation and heaping in the health-day counts
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
for ax, c in zip(axes, ["MentHlth", "PhysHlth"]):
    share = d[c].value_counts(normalize=True).reindex(range(31), fill_value=0)
    ax.bar(share.index, share.values); ax.set_yscale("log")
    ax.set_title(c); ax.set_xlabel("Days in the past 30")
axes[0].set_ylabel("Share of respondents (log scale)")
save(fig, "fig_eda_health_days.png")

# 5. Spearman correlation (ordinal and binary data, so not Pearson)
rho = spearmanr(d[feats + [TARGET]]).statistic
fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(rho, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(22), feats + [TARGET], rotation=90, fontsize=7)
ax.set_yticks(range(22), feats + [TARGET], fontsize=7)
fig.colorbar(im, shrink=0.8)
save(fig, "fig_eda_spearman.png")

# 6. mutual information ranking (also the MI baseline selector in Phase 4)
mi = pd.Series(mutual_info_classif(d[feats], d[TARGET], discrete_features=[c != "BMI" for c in feats],
                                   random_state=0), index=feats).sort_values()
mi.to_csv("results/mi_train.csv")
fig, ax = plt.subplots(figsize=(6, 6))
ax.barh(mi.index, mi.values); ax.set_xlabel("Mutual information with diabetes (nats)")
save(fig, "fig_eda_mutual_information.png")

# 7. access and income: foundation of the equity audit
fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.3, 1]})
p, lo, hi = rate_ci("Income")
axes[0].errorbar(range(1, 9), p, yerr=[p - lo, hi - p], fmt="o-", capsize=3)
axes[0].set_xlabel("Income band (1 = under $10k, 8 = $75k+)"); axes[0].set_ylabel("Diabetes prevalence")
labels, vals, errs = [], [], []
for c in ["AnyHealthcare", "NoDocbcCost", "CholCheck"]:
    p, lo, hi = rate_ci(c)
    for lev in (0, 1):
        labels.append(f"{c}={lev}"); vals.append(p[lev]); errs.append([p[lev] - lo[lev], hi[lev] - p[lev]])
axes[1].barh(labels, vals, xerr=np.array(errs).T, capsize=3)
axes[1].axvline(PREV, ls="--", color="0.5", lw=1); axes[1].set_xlabel("Diabetes prevalence")
save(fig, "fig_eda_access.png")
print("saved 7 figures to", FIG.resolve())
