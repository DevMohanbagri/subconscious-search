"""05 - Evidence for rough-set failure modes A and B (training rows only)."""
import json
import numpy as np, pandas as pd
from data_io import load, features, TARGET
from discretise import fit_cuts, apply_cuts
from rst import encode, granules, class_stats, quickreduct, gamma_null

df, tr, va, _ = load()
feats = features(df)
train = df.loc[tr].reset_index(drop=True)
y = train[TARGET].to_numpy(float)

rows, paths = [], {}
for regime in ["raw", "clinical", "equal_width", "supervised"]:
    d = train if regime == "raw" else apply_cuts(train, fit_cuts(train, regime))
    enc = encode(d, feats)
    gid = granules(enc, feats, len(y))
    st = class_stats(gid, y, beta=1.0)
    R, g_R, g_C, path = quickreduct(enc, y, feats, beta=1.0)
    rows.append({"regime": regime,
                 "gamma_all21": round(st["gamma"], 4),
                 "gamma_shuffled": round(gamma_null(enc, y, feats, 1.0, n_perm=3).mean(), 4),
                 "n_granules": st["n_granules"],
                 "singleton_share": round(st["singleton_share"], 4),
                 "largest_granule": int(np.bincount(gid).max()),
                 "quickreduct_size": len(R), "quickreduct_gamma": round(g_R, 4)})
    paths[regime] = [(a, round(float(g), 4), int(t)) for a, g, t in path]

# Do rules that are 'certain' on raw training data generalise? Look them up for validation rows.
g = train.groupby(feats)[TARGET].agg(["size", "sum"])
certain = g[(g["sum"] == 0) | (g["sum"] == g["size"])]
rule = (certain["sum"] > 0).astype(int).rename("rule").reset_index()
v = df.loc[va].merge(rule, on=feats, how="left")
covered = v["rule"].notna()
acc = (v.loc[covered, "rule"] == v.loc[covered, TARGET]).mean()
print(f"raw certain rules cover {covered.sum()} of {len(v)} validation rows "
      f"({covered.mean():.2%}); accuracy on covered rows = {acc:.3f}")

out = pd.DataFrame(rows)
out.to_csv("results/rst_regimes.csv", index=False)
json.dump(paths, open("results/rst_quickreduct_paths.json", "w"), indent=1)
print(out.to_string(index=False))
