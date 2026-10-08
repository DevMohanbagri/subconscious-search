"""08 - VPRS beta sweep on clinical bins: reduct size, gamma, chance level, positives captured."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import average_precision_score
from data_io import load, features, TARGET
from discretise import fit_cuts, apply_cuts
from rst import encode, granules, class_stats, quickreduct, gamma_null

BETAS = [0.60, 0.70, 0.80, 0.82, 0.84, 0.85, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 0.99, 1.00]

df, tr, va, _ = load()
feats = features(df)
train = df.loc[tr].reset_index(drop=True)
d = apply_cuts(train, fit_cuts(train, "clinical"))
y = d[TARGET].to_numpy(float)
enc = encode(d, feats)


def lr_prauc(cols):  # downstream check uses the RAW columns of the reduct
    if not cols:
        return float(df.loc[va, TARGET].mean())
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    m.fit(df.loc[tr, cols], df.loc[tr, TARGET])
    return average_precision_score(df.loc[va, TARGET], m.predict_proba(df.loc[va, cols])[:, 1])


rows = []
for beta in BETAS:
    R, g_R, g_C, path = quickreduct(enc, y, feats, beta)
    st = class_stats(granules(enc, R, len(y)), y, beta) if R else {"pos_captured": 0.0}
    rows.append({"beta": beta, "reduct_size": len(R), "gamma_R": round(g_R, 4), "gamma_C": round(g_C, 4),
                 "gamma_R_shuffled": round(gamma_null(enc, y, R, beta, 3).mean(), 4) if R else 0.0,
                 "pos_captured": round(st["pos_captured"], 4),
                 "ties_first_step": path[0][2] if path else 0,
                 "lr_prauc_val": round(lr_prauc(R), 4), "reduct": " ".join(R)})
    print(rows[-1])
res = pd.DataFrame(rows)
res.to_csv("results/rst_beta_sweep.csv", index=False)

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
ax[0].plot(res.beta, res.reduct_size, "o-"); ax[0].set_ylabel("Attributes in reduct")
ax[1].plot(res.beta, res.gamma_R, "o-", label="real labels")
ax[1].plot(res.beta, res.gamma_R_shuffled, "s--", label="shuffled labels"); ax[1].legend()
ax[1].set_ylabel("Dependency degree of the reduct")
ax[2].plot(res.beta, res.pos_captured, "o-"); ax[2].set_ylabel("Share of positives captured")
for a in ax:
    a.set_xlabel("beta")
fig.tight_layout(); fig.savefig("figures/fig_rst_beta_sweep.png", dpi=300)
