"""99 - THE ONE-TIME TEST EVALUATION. Refuses to run without the freeze tag, or a second time.
    python 99_final_test.py --check      (prerequisites only; never opens the test split)
    python 99_final_test.py              (the single real run)"""
import argparse, datetime, json, subprocess, sys
from pathlib import Path

LOCK = Path("results/TEST_TOUCHED.lock")
git = lambda *args: subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()
ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true")
a = ap.parse_args()
problems = []
if LOCK.exists():
    problems.append(f"test already evaluated ({LOCK.read_text().strip()})")
if "freeze" not in git("tag", "--points-at", "HEAD").split():
    problems.append("HEAD is not tagged 'freeze' (run: git tag freeze)")
if git("status", "--porcelain", "--", "*.py", "*.json"):
    problems.append("uncommitted changes to .py or .json files")
for f in ["models/deployed.json", "models/calibrator.joblib", "fuzzy_rules_v2.json", "feature_sets.json", "DECISIONS.md"]:
    if not Path(f).exists():
        problems.append(f"missing {f}")
if problems or a.check:
    print("\n".join(problems) or "all prerequisites met - ready for the single test run")
    sys.exit(1 if problems else 0)

# ---------------- from here on the test split is used, exactly once ----------------
import glob, joblib, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from data_io import load, log_result, TARGET
from metrics import summarize, bootstrap_ci, at_threshold, wilson_ci
from models import raw_predict
from pipeline import RiskPipeline

LOCK.write_text(f"{datetime.datetime.now().isoformat(timespec='seconds')} at commit {git('rev-parse', '--short', 'HEAD')}\n")
df, tr, va, te = load(allow_test=True)
test, y = df.loc[te].reset_index(drop=True), df.loc[te, TARGET].to_numpy()
spec = json.load(open("models/deployed.json"))

# 1. every saved main-grid model: ranking metrics on test (no calibration needed for these)
rows = []
for f in sorted(glob.glob("models/*_s[0-9].joblib")):
    info, name = joblib.load(f), Path(f).stem
    p = raw_predict(info, test)
    rows.append({"config": name, "pr_auc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p)})
    log_result(script="99_final_test.py", feature_set=name, model=name.split("_")[0], seed=int(name[-1]),
               split="test", calibrated="no", pr_auc=round(rows[-1]["pr_auc"], 4),
               roc_auc=round(rows[-1]["roc_auc"], 4), prevalence=round(y.mean(), 4))
pd.DataFrame(rows).to_csv("results/final_test_models.csv", index=False)

# 2. the deployed system end to end, at the operating point chosen on validation
res = RiskPipeline().assess(test)
p = res.p_diabetes.to_numpy()
pr, lo, hi = bootstrap_ci(average_precision_score, y, p)
op = at_threshold(y, p, spec["threshold_30pct"])
tiers = {t: {"n": int((res.tier == t).sum()), "prevalence": float(y[res.tier == t].mean()),
             "ci": wilson_ci(int(y[res.tier == t].sum()), int((res.tier == t).sum()))}
         for t in ["Minimal", "Watch", "Elevated", "Priority"]}
log_result(script="99_final_test.py", feature_set=spec["feature_set"], model=f"deployed {spec['model']}",
           seed=0, split="test", calibrated=spec["calibrator"], threshold=spec["threshold_30pct"],
           **summarize(y, p), extra={"pr_auc_ci": [lo, hi], **op})
out = test.assign(y_true=y, p_cal=p, tier=res.tier)
out.insert(0, "row_id", te)
out.to_csv("results/preds/test_final.csv", index=False)       # 19_equity_audit.py --preds reads this
json.dump({"date": str(datetime.date.today()), "pr_auc": [pr, lo, hi], "operating_point": op, "tiers": tiers},
          open(f"results/final_test_{datetime.date.today()}.json", "w"), indent=2, default=float)
print(json.dumps({"pr_auc": [round(pr, 4), round(lo, 4), round(hi, 4)], **op}, indent=2))
