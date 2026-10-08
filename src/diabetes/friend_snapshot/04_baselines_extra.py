"""04 - majority class, unweighted logistic regression, LightGBM x 5 seeds (validation only)."""
import config  # noqa: F401  thread settings must load before numerical libraries
import numpy as np, lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from data_io import load, features, log_result, TARGET
from metrics import summarize

df, tr, va, _ = load()
LGB = dict(n_estimators=3000, learning_rate=0.03, num_leaves=31, min_child_samples=100,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)

for fs, drop in [("all_21", False), ("leakage_free", True)]:
    cols = features(df, drop_leakage=drop)
    Xtr, ytr = df.loc[tr, cols].to_numpy(np.float32), df.loc[tr, TARGET].to_numpy()
    Xva, yva = df.loc[va, cols].to_numpy(np.float32), df.loc[va, TARGET].to_numpy()
    base = dict(script="04_baselines_extra.py", feature_set=fs, n_features=len(cols), split="val")

    p = np.full(len(yva), ytr.mean())                      # 1. the floor
    log_result(**base, model="majority class", seed=0, calibrated="n/a", **summarize(yva, p),
               notes="constant score = train prevalence")

    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr)
    p = lr.predict_proba(Xva)[:, 1]                         # 2. calibration reference
    log_result(**base, model="logistic regression (unweighted)", seed=0, calibrated="native",
               **summarize(yva, p), notes="no class weight")

    for seed in range(5):                                   # 3. strong tabular baseline
        fit, es = train_test_split(np.arange(len(ytr)), test_size=0.10, stratify=ytr, random_state=seed)
        m = lgb.LGBMClassifier(**LGB, n_jobs=config.THREADS_PER_WORKER, random_state=seed)
        m.fit(Xtr[fit], ytr[fit], eval_X=(Xtr[es],), eval_y=(ytr[es],), eval_metric="average_precision",
              callbacks=[lgb.early_stopping(150, verbose=False)])
        p = m.predict_proba(Xva)[:, 1]
        log_result(**base, model="lightgbm", seed=seed, calibrated="native",
                   params={**LGB, "best_iteration": m.best_iteration_}, **summarize(yva, p),
                   notes="early stopping on inner 10% of train")
        print(fs, "lightgbm seed", seed, summarize(yva, p)["pr_auc"], "iters", m.best_iteration_)
