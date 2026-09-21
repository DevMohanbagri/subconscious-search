#!/usr/bin/env python3
"""SMO on real data: BRFSS 2015 Diabetes Health Indicators (binary 50/50).

Compares test accuracy of:
  Baselines  - Dummy, LogisticRegression, RandomForest,
                 HistGradientBoosting, MLP (all sklearn, fixed seeds)
  SMO-direct - SMO-Pop directly optimizes regularized logistic-regression
               weights (22-D) on validation log-loss  [can SMO *train*?]
  SMO-HPO    - SMO-Pop tunes HistGradientBoosting hyperparams (4-D),
               vs same-budget RandomSearch and defaults  [can SMO *tune*?]

Test set is touched exactly once per method, at the end.
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from smo_pop import SMOPop

SEED = 42
DATA = "data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"


def load_data():
    df = pd.read_csv(DATA)
    y = df.iloc[:, 0].to_numpy(dtype=int)
    X = df.iloc[:, 1:].to_numpy(dtype=float)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED)
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
    print(f"train/val/test: {len(X_tr)}/{len(X_va)}/{len(X_te)} "
          f"({X.shape[1]} features, standardized for linear/MLP/SMO-direct)")
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te)


def report(name, y_true, y_pred, y_score=None, extra=""):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score) if y_score is not None else float("nan")
    print(f"  {name:28s} acc={acc*100:6.2f}%  f1={f1:.4f}  auc={auc:.4f}  {extra}")
    return {"acc": acc, "f1": f1, "auc": auc}


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def run_baselines(splits):
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = splits
    sc = StandardScaler().fit(X_tr)
    Xtr_s, Xva_s, Xte_s = sc.transform(X_tr), sc.transform(X_va), sc.transform(X_te)
    Xtr_full_s = np.vstack([Xtr_s, Xva_s])
    ytr_full = np.concatenate([y_tr, y_va])
    Xtr_full = np.vstack([X_tr, X_va])
    out = {}
    print("\n[1] sklearn baselines (train+val fit, test eval)")
    t0 = time.time()
    m = DummyClassifier(strategy="most_frequent").fit(Xtr_full, ytr_full)
    out["dummy"] = report("dummy (majority)", y_te, m.predict(X_te), extra=f"{time.time()-t0:.1f}s")
    t0 = time.time()
    m = LogisticRegression(max_iter=2000).fit(Xtr_full_s, ytr_full)
    out["logreg"] = report("logreg (LBFGS)", y_te, m.predict(Xte_s),
                           m.predict_proba(Xte_s)[:, 1], extra=f"{time.time()-t0:.1f}s")
    t0 = time.time()
    m = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    m.fit(Xtr_full, ytr_full)
    out["randforest"] = report("randomforest (300)", y_te, m.predict(X_te),
                               m.predict_proba(X_te)[:, 1], extra=f"{time.time()-t0:.1f}s")
    t0 = time.time()
    m = HistGradientBoostingClassifier(random_state=SEED).fit(Xtr_full, ytr_full)
    out["hgb_default"] = report("hgb (defaults)", y_te, m.predict(X_te),
                                m.predict_proba(X_te)[:, 1], extra=f"{time.time()-t0:.1f}s")
    t0 = time.time()
    m = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                      random_state=SEED).fit(Xtr_full_s, ytr_full)
    out["mlp"] = report("mlp (64x32, adam)", y_te, m.predict(Xte_s),
                        m.predict_proba(Xte_s)[:, 1], extra=f"{time.time()-t0:.1f}s")
    return out, (Xtr_s, Xva_s, Xte_s, Xtr_full, ytr_full)


def run_smo_direct(splits_scaled, y_tr, y_va, y_te, max_evals=2000):
    Xtr_s, Xva_s, Xte_s = splits_scaled
    d = Xtr_s.shape[1]
    Xva1 = np.hstack([Xva_s, np.ones((len(Xva_s), 1))])
    Xte1 = np.hstack([Xte_s, np.ones((len(Xte_s), 1))])

    def val_logloss(w):
        p = sigmoid(Xva1 @ w)
        eps = 1e-12
        return float(-(y_va * np.log(p + eps)
                         + (1 - y_va) * np.log(1 - p + eps)).mean()
                     + 1e-4 * float(w @ w))

    print(f"\n[2] SMO-direct: {d+1}-D logistic weights on val log-loss "
          f"({max_evals} evals)")
    t0 = time.time()
    opt = SMOPop(d + 1, -5.0, 5.0, seed=SEED)
    w_best, best_val, info = opt.optimize(val_logloss, max_evals)
    dt = time.time() - t0
    p_te = sigmoid(Xte1 @ w_best)
    print(f"    val logloss={best_val:.5f} gens={info['gens']} "
          f"restarts={info['restarts']} bh={info['bh_phase']} time={dt:.1f}s")
    out = report("SMO-direct (logreg w)", y_te, (p_te >= 0.5).astype(int), p_te,
                 extra=f"{dt:.1f}s")
    return out


def hgb_from_x(x):
    lr = 10.0 ** float(np.clip(x[0], -3.0, -0.3))
    depth = int(np.clip(round(x[1]), 2, 12))
    leaf = int(np.clip(round(10.0 ** float(np.clip(x[2], 0.0, 2.5))), 1, 300))
    l2 = float(np.clip(x[3], 0.0, 20.0))
    return (HistGradientBoostingClassifier(
        learning_rate=lr, max_depth=depth, min_samples_leaf=leaf,
        l2_regularization=l2, max_iter=200, random_state=SEED),
        {"lr": lr, "depth": depth, "leaf": leaf, "l2": l2})


def run_smo_hpo(X_tr, y_tr, X_va, y_va, X_te, y_te, budget=108):
    print(f"\n[3] HPO on HistGradientBoosting: SMO vs RandomSearch "
          f"({budget} evals each, val log-loss objective)")

    def val_loss(x):
        m, _ = hgb_from_x(x)
        m.fit(X_tr, y_tr)
        p = m.predict_proba(X_va)[:, 1]
        eps = 1e-12
        return float(-(y_va * np.log(p + eps)
                         + (1 - y_va) * np.log(1 - p + eps)).mean())

    lo = np.array([-3.0, 2.0, 0.0, 0.0])
    hi = np.array([-0.3, 12.0, 2.5, 20.0])

    # SMO (local search OFF: int-rounded HPO landscape is stepwise,
    #  gradient polish would waste evals)
    t0 = time.time()
    opt = SMOPop(4, lo, hi, seed=SEED)
    x_smo, v_smo, info = opt.optimize(val_loss, budget, local_search=False)
    t_smo = time.time() - t0
    m_smo, p_smo = hgb_from_x(x_smo)
    m_smo.fit(np.vstack([X_tr, X_va]), np.concatenate([y_tr, y_va]))
    print(f"    SMO best val={v_smo:.5f} params={p_smo} time={t_smo:.1f}s")
    out_smo = report("SMO-HPO (hgb)", y_te, m_smo.predict(X_te),
                     m_smo.predict_proba(X_te)[:, 1], extra=f"{t_smo:.1f}s")

    # Random search, identical budget
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    best, best_x = np.inf, None
    for _ in range(budget):
        x = rng.uniform(lo, hi)
        v = val_loss(x)
        if v < best:
            best, best_x = v, x.copy()
    t_rs = time.time() - t0
    m_rs, p_rs = hgb_from_x(best_x)
    m_rs.fit(np.vstack([X_tr, X_va]), np.concatenate([y_tr, y_va]))
    print(f"    RS  best val={best:.5f} params={p_rs} time={t_rs:.1f}s")
    out_rs = report("randsearch-HPO (hgb)", y_te, m_rs.predict(X_te),
                    m_rs.predict_proba(X_te)[:, 1], extra=f"{t_rs:.1f}s")
    return {"smo_hpo": out_smo, "rs_hpo": out_rs}


def main():
    print("BRFSS2015 diabetes_binary 50/50 — SMO vs sklearn (seed 42)")
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = load_data()
    base, scaled = run_baselines(((X_tr, y_tr), (X_va, y_va), (X_te, y_te)))
    Xtr_s, Xva_s, Xte_s, Xtr_full, ytr_full = scaled
    smo_direct = run_smo_direct((Xtr_s, Xva_s, Xte_s), y_tr, y_va, y_te)
    hpo = run_smo_hpo(X_tr, y_tr, X_va, y_va, X_te, y_te)

    print("\n================ TEST ACCURACY % (higher better) ================")
    rows = [("dummy", base["dummy"]), ("logreg", base["logreg"]),
            ("SMO-direct", smo_direct), ("randforest", base["randforest"]),
            ("hgb default", base["hgb_default"]), ("mlp", base["mlp"]),
            ("randsearch-HPO", hpo["rs_hpo"]), ("SMO-HPO", hpo["smo_hpo"])]
    for name, m in rows:
        print(f"  {name:14s} acc={m['acc']*100:6.2f}%  f1={m['f1']:.4f}  auc={m['auc']:.4f}")
    print("=================================================================")


if __name__ == "__main__":
    main()
