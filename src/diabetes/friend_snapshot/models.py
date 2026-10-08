"""models.py - one fit_predict() for every model and imbalance strategy, so comparisons share code."""
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from data_io import load, log_result, TARGET
from metrics import summarize
import fsets

STRATEGIES = ["none", "weighted", "smotenc", "undersample"]
CONTINUOUS = ("BMI", "MentHlth", "PhysHlth")


def resample(X, y, strategy, cols, seed):
    """Oversample or undersample TRAINING rows only (never validation or test)."""
    if strategy == "smotenc":
        from imblearn.over_sampling import SMOTE, SMOTEN, SMOTENC
        cont = [i for i, c in enumerate(cols) if c in CONTINUOUS]
        cat = [i for i in range(len(cols)) if i not in cont]
        if not cont:
            sm = SMOTEN(sampling_strategy=0.5, random_state=seed)       # all columns categorical
        elif not cat:
            sm = SMOTE(sampling_strategy=0.5, random_state=seed)        # all columns continuous
        else:
            sm = SMOTENC(categorical_features=cat, sampling_strategy=0.5, random_state=seed)
        Xr, yr = sm.fit_resample(X, y)
        if cont:
            Xr[:, cont] = np.round(Xr[:, cont])   # synthetic BMI and day counts back to whole numbers
        return Xr, yr
    if strategy == "undersample":
        from imblearn.under_sampling import RandomUnderSampler
        return RandomUnderSampler(sampling_strategy=0.5, random_state=seed).fit_resample(X, y)
    return X, y


def fit_predict(model, Xtr, ytr, Xva, cols, strategy="none", seed=0, params=None):
    """Train 'lr', 'lgbm' or 'mlp' with an imbalance strategy; return (validation P(diabetes), info)."""
    params = params or {}
    Xtr, ytr = resample(Xtr, ytr, strategy, cols, seed)
    pos_weight = float((ytr == 0).sum() / (ytr == 1).sum()) if strategy == "weighted" else 1.0
    if model == "lr":
        sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(max_iter=2000, class_weight={0: 1.0, 1: pos_weight}).fit(sc.transform(Xtr), ytr)
        return m.predict_proba(sc.transform(Xva))[:, 1], {"model": m, "scaler": sc}
    if model == "lgbm":
        import config, lightgbm as lgb
        fit, es = train_test_split(np.arange(len(ytr)), test_size=0.10, stratify=ytr, random_state=seed)
        hp = dict(n_estimators=3000, learning_rate=0.03, num_leaves=31, min_child_samples=100,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0)
        hp.update(params)
        m = lgb.LGBMClassifier(**hp, scale_pos_weight=pos_weight, n_jobs=config.THREADS_PER_WORKER,
                               random_state=seed, verbose=-1)
        m.fit(Xtr[fit], ytr[fit], eval_X=(Xtr[es],), eval_y=(ytr[es],), eval_metric="average_precision",
              callbacks=[lgb.early_stopping(150, verbose=False)])
        return m.predict_proba(Xva)[:, 1], {"model": m, "best_iteration": m.best_iteration_}
    if model == "mlp":
        from mlp import train_mlp          # imported lazily so lr/lgbm runs do not need torch
        return train_mlp(Xtr, ytr, Xva, pos_weight=pos_weight, seed=seed, **params)
    raise ValueError(model)


def run_config(model, feature_set, strategy="none", seed=0, params=None, script="?"):
    """Train one configuration, log it to the registry, save its validation predictions."""
    df, tr, va, _ = load()
    cols = fsets.get(feature_set)
    Xtr, ytr = df.loc[tr, cols].to_numpy(np.float32), df.loc[tr, TARGET].to_numpy()
    Xva, yva = df.loc[va, cols].to_numpy(np.float32), df.loc[va, TARGET].to_numpy()
    p, info = fit_predict(model, Xtr, ytr, Xva, cols, strategy, seed, params)
    res = summarize(yva, p)
    log_result(script=script, feature_set=feature_set, n_features=len(cols), model=model, seed=seed,
               split="val", calibrated="no", params={"strategy": strategy, **(params or {})}, **res)
    Path("models").mkdir(exist_ok=True)                       # keep the fitted model for the pipeline
    joblib.dump({"cols": cols, **info}, f"models/{model}_{feature_set}_{strategy}_s{seed}.joblib")
    out = Path("results/preds"); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row_id": va, "y_true": yva, "p_raw": p}).to_csv(
        out / f"val_{model}_{feature_set}_{strategy}_s{seed}.csv", index=False)
    return {"model": model, "feature_set": feature_set, "strategy": strategy, "seed": seed, **res}


def raw_predict(info, frame):
    """Uncalibrated P(diabetes) from a model saved by run_config (lr, lgbm or mlp)."""
    X = frame[info["cols"]].to_numpy(np.float32)
    model = info["model"]
    if hasattr(model, "forward"):                              # PyTorch MLP
        import torch
        with torch.no_grad():
            z = model(torch.tensor(info["scaler"].transform(X), dtype=torch.float32)).squeeze(1)
            return torch.sigmoid(z).numpy()
    if "scaler" in info:                                       # logistic regression
        return model.predict_proba(info["scaler"].transform(X))[:, 1]
    return model.predict_proba(X)[:, 1]                        # LightGBM


def load_predictor(spec_path="models/deployed.json"):
    """Callable: DataFrame of answers -> calibrated P(diabetes) from the deployed model."""
    import json
    spec = json.load(open(spec_path))
    info, cal = joblib.load(spec["model_file"]), joblib.load(spec["calibrator_file"])
    return lambda frame: cal.predict(raw_predict(info, frame))
