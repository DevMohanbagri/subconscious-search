"""14 - Four imbalance strategies x 5 seeds on one feature set: PR-AUC and the calibration cost."""
import argparse
import config
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, brier_score_loss
from models import run_config, STRATEGIES
from calib import crossfit

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="mlp"); ap.add_argument("--feature-set", default="all_21")
ap.add_argument("--seeds", type=int, default=5)
a = ap.parse_args()
cfgs = [(s, seed) for s in STRATEGIES for seed in range(a.seeds)]
Parallel(n_jobs=config.N_WORKERS)(delayed(run_config)(a.model, a.feature_set, s, seed,
                                                      script="14_imbalance_ablation.py") for s, seed in cfgs)
rows = []
for s, seed in cfgs:
    p = pd.read_csv(f"results/preds/val_{a.model}_{a.feature_set}_{s}_s{seed}.csv")
    y, q = p.y_true.to_numpy(), p.p_raw.to_numpy()
    rows.append({"strategy": s, "seed": seed, "pr_auc": average_precision_score(y, q),
                 "mean_pred": q.mean(), "brier_raw": brier_score_loss(y, q),
                 "brier_after_platt": brier_score_loss(y, crossfit(q, y, "platt"))})
res = pd.DataFrame(rows)
res.to_csv(f"results/imbalance_{a.model}_{a.feature_set}.csv", index=False)
summ = res.groupby("strategy", sort=False).agg(["mean", "std"]).drop(columns="seed").round(4)
print(summ.to_string())

m = res.groupby("strategy", sort=False).mean()
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].bar(m.index, m.pr_auc); ax[0].set_ylabel("Validation PR-AUC"); ax[0].set_ylim(m.pr_auc.min() - 0.02, m.pr_auc.max() + 0.01)
x = range(len(m))
ax[1].bar([i - 0.2 for i in x], m.brier_raw, width=0.4, label="as trained")
ax[1].bar([i + 0.2 for i in x], m.brier_after_platt, width=0.4, label="after Platt scaling")
ax[1].set_xticks(list(x), m.index); ax[1].set_ylabel("Brier score (lower is better)"); ax[1].legend()
fig.tight_layout(); fig.savefig(f"figures/fig_mlp_imbalance_{a.model}.png", dpi=300)
