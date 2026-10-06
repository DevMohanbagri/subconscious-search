"""10 - Stability: re-run each selector on 50 random 80% subsamples of the training rows."""
import argparse
import config
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from data_io import load, features, TARGET
from discretise import fit_cuts, apply_cuts
from rst import encode, heldout_reduct, quickreduct, mean_pairwise_jaccard, nogueira_stability

ap = argparse.ArgumentParser(); ap.add_argument("--b", type=int, default=50)
B = ap.parse_args().b

df, tr, va, _ = load()
train = df.loc[tr].reset_index(drop=True)
d = apply_cuts(train, fit_cuts(train, "clinical"))   # bins fixed once; WHO cut points do not depend on data
ALL, NOLEAK = features(df), features(df, drop_leakage=True)
SELECTORS = {"heldout_all21": ("heldout", ALL), "heldout_noleak": ("heldout", NOLEAK),
             "vprs_b090": (0.90, ALL), "vprs_b094": (0.94, ALL)}


def one_run(b):
    idx = np.random.default_rng(1000 + b).choice(len(d), int(0.8 * len(d)), replace=False)
    sub = d.iloc[idx].reset_index(drop=True)            # subsample WITHOUT replacement
    y, enc = sub[TARGET].to_numpy(float), encode(sub, ALL)
    out = {}
    for name, (kind, attrs) in SELECTORS.items():
        if kind == "heldout":
            out[name] = heldout_reduct(enc, y, attrs, a=20, n_folds=5, seed=b)[0]
        else:
            out[name] = quickreduct(enc, y, attrs, beta=kind)[0]
    return out


runs = Parallel(n_jobs=config.N_WORKERS)(delayed(one_run)(b) for b in range(B))
freq, summary = {}, []
for name, (_, attrs) in SELECTORS.items():
    subsets = [r[name] for r in runs]
    freq[name] = pd.Series({a: np.mean([a in s for s in subsets]) for a in ALL})
    summary.append({"selector": name, "mean_size": np.mean([len(s) for s in subsets]),
                    "distinct_subsets": len({tuple(sorted(s)) for s in subsets}),
                    "mean_jaccard": round(mean_pairwise_jaccard(subsets), 3),
                    "nogueira": round(nogueira_stability(subsets, ALL), 3)})
summ = pd.DataFrame(summary)
summ.to_csv("results/rst_stability_summary.csv", index=False)
pd.DataFrame(freq).to_csv("results/rst_stability_freq.csv")
print(summ.to_string(index=False))
print(pd.DataFrame(freq).round(2).loc[lambda t: t.max(axis=1) > 0].to_string())

order = freq["heldout_all21"].sort_values().index
fig, ax = plt.subplots(figsize=(7, 6))
ypos = np.arange(len(order))
ax.barh(ypos - 0.2, freq["heldout_all21"][order], height=0.4, label="held-out entropy reduct")
ax.barh(ypos + 0.2, freq["vprs_b094"][order], height=0.4, label="VPRS QuickReduct, beta = 0.94")
ax.set_yticks(ypos, order); ax.set_xlabel(f"Selection frequency over {B} subsamples"); ax.legend()
fig.tight_layout(); fig.savefig("figures/fig_rst_stability.png", dpi=300)
