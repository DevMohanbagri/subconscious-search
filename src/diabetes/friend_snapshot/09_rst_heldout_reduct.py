"""09 - The fix: a reduct chosen by held-out conditional entropy of the granules (clinical bins)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data_io import load, features, TARGET
from discretise import fit_cuts, apply_cuts
from rst import encode, heldout_reduct
import fsets

df, tr, va, _ = load()
train = df.loc[tr].reset_index(drop=True)
d = apply_cuts(train, fit_cuts(train, "clinical"))
y = d[TARGET].to_numpy(float)
pools = {"all_21": features(df), "leakage_free": features(df, drop_leakage=True)}
enc = encode(d, pools["all_21"])

rows, primary = [], {}
for pool, attrs in pools.items():
    for a in [5, 20, 100]:                 # shrinkage strength (pseudo-counts)
        for seed in range(5):              # which rows land in which fold
            R, path, gh = heldout_reduct(enc, y, attrs, a=a, n_folds=5, seed=seed)
            rows.append({"pool": pool, "a": a, "seed": seed, "k": len(R),
                         "gamma_H": round(gh, 4), "reduct": " ".join(R)})
            if a == 20 and seed == 0:
                primary[pool] = (R, path)
res = pd.DataFrame(rows)
res.to_csv("results/rst_heldout_reduct.csv", index=False)
print(res.groupby(["pool", "a"]).agg(k=("k", "mean"), distinct_reducts=("reduct", "nunique"),
                                     gamma_H=("gamma_H", "mean")).round(4).to_string())

name = {"all_21": "rst_heldout", "leakage_free": "rst_heldout_noleak"}
for pool, (R, path) in primary.items():
    fsets.save(name[pool], R, "09_rst_heldout_reduct.py a=20 folds=5 seed=0 clinical bins", pool=pool)
    print(name[pool], R, path)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, (pool, (R, path)) in zip(axes, primary.items()):
    ax.bar(range(len(path)), [g for _, g in path])
    ax.set_xticks(range(len(path)), [a for a, _ in path], rotation=30, ha="right")
    ax.set_title(f"{pool}: search stops after {len(R)} attributes")
axes[0].set_ylabel("Label uncertainty removed\n(held-out, share of total)")
fig.tight_layout(); fig.savefig("figures/fig_rst_heldout_path.png", dpi=300)
