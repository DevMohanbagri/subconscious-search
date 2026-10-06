"""15 - Calibrate the chosen model on validation; write the contract file Stage 4 reads.
    python 15_calibrate.py --preds results/preds/val_mlp_rst_heldout_noleak_none_s0.csv"""
import argparse
from pathlib import Path
import json, joblib, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, brier_score_loss
from data_io import load, features, log_result
from metrics import ece, wilson_ci
from calib import Calibrator, crossfit

ap = argparse.ArgumentParser(); ap.add_argument("--preds", required=True)
a = ap.parse_args()
df, tr, va, _ = load()
p = pd.read_csv(a.preds)
assert (p.row_id.to_numpy() == va).all(), "prediction rows must be the validation rows, in order"
y, raw = p.y_true.to_numpy(), p.p_raw.to_numpy()

versions = {"raw": raw, "platt": crossfit(raw, y, "platt"), "isotonic": crossfit(raw, y, "isotonic")}
table = pd.DataFrame({k: {"brier": brier_score_loss(y, q), "ece": ece(y, q), "pr_auc": average_precision_score(y, q),
                          "mean_pred": q.mean()} for k, q in versions.items()}).T.round(4)
print(table.to_string(), f"\nobserved prevalence {y.mean():.4f}")
# Prefer Platt: it is smooth and keeps the ranking. Isotonic must earn its place: a clearly better Brier
# (by 0.001+) without losing more than 0.002 PR-AUC to the ties its step function creates.
iso_better = (table.loc["platt", "brier"] - table.loc["isotonic", "brier"] >= 0.001 and
              table.loc["platt", "pr_auc"] - table.loc["isotonic", "pr_auc"] <= 0.002)
best = "isotonic" if iso_better else "platt"

Path("models").mkdir(exist_ok=True)
cal = Calibrator(best).fit(raw, y)                                     # deployed: fitted on ALL validation
joblib.dump(cal, "models/calibrator.joblib")
stem = Path(a.preds).stem[4:]                                          # <model>_<feature set>_<strategy>_s<seed>
rest = stem.split("_", 1)[1]
json.dump({"preds": a.preds, "model_file": f"models/{stem}.joblib", "calibrator_file": "models/calibrator.joblib",
           "calibrator": best, "model": stem.split("_", 1)[0], "feature_set": rest.rsplit("_", 2)[0],
           "threshold_30pct": float(np.quantile(cal.predict(raw), 0.70))},   # flags 30% of validation
          open("models/deployed.json", "w"), indent=2)
out = df.loc[va, features(df)].reset_index(drop=True)
out.insert(0, "p_cal", versions[best]); out.insert(0, "p_raw", raw)
out.insert(0, "y_true", y); out.insert(0, "row_id", va)
out.to_csv("results/preds/val_final.csv", index=False)                 # THE Stage 4 contract file
log_result(script="15_calibrate.py", feature_set=Path(a.preds).stem, model="calibrated", seed=0, split="val",
           calibrated=best, pr_auc=table.loc[best, "pr_auc"], brier=table.loc[best, "brier"],
           ece=table.loc[best, "ece"], prevalence=round(y.mean(), 4), notes="cross-fitted, 2 folds")

fig, ax = plt.subplots(figsize=(5, 5))
for name, q in [("as trained", raw), (f"after {best}", versions[best])]:
    bins = np.quantile(q, np.linspace(0, 1, 11)); idx = np.clip(np.searchsorted(bins, q, "right") - 1, 0, 9)
    mp = [q[idx == b].mean() for b in range(10)]; ob = [y[idx == b].mean() for b in range(10)]
    ci = [wilson_ci(y[idx == b].sum(), (idx == b).sum()) for b in range(10)]
    ax.errorbar(mp, ob, yerr=np.array([[o - l, h - o] for o, (l, h) in zip(ob, ci)]).T, fmt="o-", capsize=2, label=name)
ax.plot([0, 1], [0, 1], ":", color="0.5"); ax.set_xlabel("Mean predicted probability (10 equal-size bins)")
ax.set_ylabel("Observed diabetes rate"); ax.legend(); fig.tight_layout()
fig.savefig("figures/fig_mlp_reliability.png", dpi=300)
print("chosen:", best, "-> models/calibrator.joblib and results/preds/val_final.csv")
