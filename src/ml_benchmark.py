#!/usr/bin/env python3
"""Bicameral on real data: BRFSS 2015 Diabetes Health Indicators (binary 50/50).

Compares test accuracy of:
  Baselines  - Dummy, LogisticRegression, RandomForest,
                 HistGradientBoosting, MLP (all sklearn, fixed seeds)
  Bicameral-direct - Bicameral directly optimizes regularized logistic-regression
               weights (22-D) on validation log-loss  [can Bicameral *train*?]
  Bicameral-HPO    - Bicameral tunes HistGradientBoosting hyperparams (4-D),
               vs same-budget RandomSearch and defaults  [can Bicameral *tune*?]

Test set is touched exactly once per method, at the end.

Task 4 upgrades (stolen from the diabetes papers, honestly cited):
  [4] paired stats for the headline comparisons: DeLong test on AUC
      (Sun & Su 2008; the Paper-2 method), exact McNemar on accuracy,
      and bootstrap 95% CIs - the old section was point estimates only.
  [5] SHAP explainability (the Paper-1 method): global mean|SHAP|
      ranking plus per-instance SHAP for one TP/TN/FP/FN.
"""

import argparse
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

from bicameral import Bicameral

SEED = 42
DATA = "data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"

# test-set predictions/scores per method, filled by report(key=...)
TEST = {}


def load_data(path=DATA):
    df = pd.read_csv(path)
    if isinstance(df.columns[0], str):
        names = [str(c) for c in df.columns[1:]]
    else:
        names = [f"x{j}" for j in range(df.shape[1] - 1)]
    y = df.iloc[:, 0].to_numpy(dtype=int)
    X = df.iloc[:, 1:].to_numpy(dtype=float)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED)
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
    print(f"train/val/test: {len(X_tr)}/{len(X_va)}/{len(X_te)} "
          f"({X.shape[1]} features, standardized for linear/MLP/Bicameral-direct)")
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te), names


def report(name, y_true, y_pred, y_score=None, extra="", key=None):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score) if y_score is not None else float("nan")
    print(f"  {name:28s} acc={acc*100:6.2f}%  f1={f1:.4f}  auc={auc:.4f}  {extra}")
    if key is not None:
        TEST[key] = (np.asarray(y_pred), None if y_score is None
                     else np.asarray(y_score, dtype=float))
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
                           m.predict_proba(Xte_s)[:, 1], extra=f"{time.time()-t0:.1f}s",
                           key="logreg")
    t0 = time.time()
    m = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    m.fit(Xtr_full, ytr_full)
    out["randforest"] = report("randomforest (300)", y_te, m.predict(X_te),
                               m.predict_proba(X_te)[:, 1], extra=f"{time.time()-t0:.1f}s",
                               key="randforest")
    t0 = time.time()
    m = HistGradientBoostingClassifier(random_state=SEED).fit(Xtr_full, ytr_full)
    out["hgb_default"] = report("hgb (defaults)", y_te, m.predict(X_te),
                                m.predict_proba(X_te)[:, 1], extra=f"{time.time()-t0:.1f}s",
                                key="hgb_default")
    t0 = time.time()
    m = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                      random_state=SEED).fit(Xtr_full_s, ytr_full)
    out["mlp"] = report("mlp (64x32, adam)", y_te, m.predict(Xte_s),
                        m.predict_proba(Xte_s)[:, 1], extra=f"{time.time()-t0:.1f}s",
                        key="mlp")
    return out, (Xtr_s, Xva_s, Xte_s, Xtr_full, ytr_full)


def run_bicameral_direct(splits_scaled, y_tr, y_va, y_te, max_evals=2000):
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

    print(f"\n[2] Bicameral-direct: {d+1}-D logistic weights on val log-loss "
          f"({max_evals} evals)")
    t0 = time.time()
    opt = Bicameral(d + 1, -5.0, 5.0, seed=SEED)
    w_best, best_val, info = opt.optimize(val_logloss, max_evals)
    dt = time.time() - t0
    p_te = sigmoid(Xte1 @ w_best)
    print(f"    val logloss={best_val:.5f} gens={info['gens']} "
          f"restarts={info['restarts']} bh={info['bh_phase']} time={dt:.1f}s")
    out = report("Bicameral-direct (logreg w)", y_te, (p_te >= 0.5).astype(int), p_te,
                 extra=f"{dt:.1f}s", key="bicameral_direct")
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


def run_bicameral_hpo(X_tr, y_tr, X_va, y_va, X_te, y_te, budget=108):
    print(f"\n[3] HPO on HistGradientBoosting: Bicameral vs RandomSearch "
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

    # Bicameral (local search OFF: int-rounded HPO landscape is stepwise,
    #  gradient polish would waste evals)
    t0 = time.time()
    opt = Bicameral(4, lo, hi, seed=SEED)
    x_bic, v_bic, info = opt.optimize(val_loss, budget, local_search=False)
    t_bic = time.time() - t0
    m_bic, p_bic = hgb_from_x(x_bic)
    m_bic.fit(np.vstack([X_tr, X_va]), np.concatenate([y_tr, y_va]))
    print(f"    Bicameral best val={v_bic:.5f} params={p_bic} time={t_bic:.1f}s")
    out_bic = report("Bicameral-HPO (hgb)", y_te, m_bic.predict(X_te),
                     m_bic.predict_proba(X_te)[:, 1], extra=f"{t_bic:.1f}s",
                     key="bicameral_hpo")

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
                    m_rs.predict_proba(X_te)[:, 1], extra=f"{t_rs:.1f}s",
                    key="rs_hpo")
    return {"bicameral_hpo": out_bic, "rs_hpo": out_rs}


# ---------------------------------------------------------------- paired stats
def _delong_auc_and_cov(y_true, s1, s2):
    """Exact DeLong AUCs + 2x2 covariance (Sun & Su 2008), chunked."""
    y = np.asarray(y_true).astype(int)
    P1 = np.asarray(s1, float)[y == 1]
    N1 = np.asarray(s1, float)[y == 0]
    P2 = np.asarray(s2, float)[y == 1]
    N2 = np.asarray(s2, float)[y == 0]
    m, n = len(P1), len(N1)
    V10 = np.zeros((2, m))
    V01 = np.zeros((2, n))
    for a in range(0, m, 1000):
        b = min(a + 1000, m)
        d1 = P1[a:b, None] - N1[None, :]
        d2 = P2[a:b, None] - N2[None, :]
        V10[0, a:b] = ((d1 > 0) + 0.5 * (d1 == 0)).mean(axis=1)
        V10[1, a:b] = ((d2 > 0) + 0.5 * (d2 == 0)).mean(axis=1)
    for a in range(0, n, 1000):
        b = min(a + 1000, n)
        d1 = P1[:, None] - N1[a:b][None, :]
        d2 = P2[:, None] - N2[a:b][None, :]
        V01[0, a:b] = ((d1 > 0) + 0.5 * (d1 == 0)).mean(axis=0)
        V01[1, a:b] = ((d2 > 0) + 0.5 * (d2 == 0)).mean(axis=0)
    aucs = V10.mean(axis=1)
    S = np.cov(V10, bias=True) / m + np.cov(V01, bias=True) / n
    return aucs, S


def delong_test(y_true, s1, s2):
    """Two-sided DeLong test for AUC(s1) vs AUC(s2). Returns (z, p)."""
    from scipy.stats import norm
    aucs, S = _delong_auc_and_cov(y_true, s1, s2)
    var = S[0, 0] + S[1, 1] - 2.0 * S[0, 1]
    if var <= 1e-15:
        return 0.0, 1.0
    z = float((aucs[0] - aucs[1]) / np.sqrt(var))
    return z, float(2.0 * norm.sf(abs(z)))


def mcnemar_test(y_true, p1, p2):
    """Exact two-sided McNemar on paired accuracies. Returns (b, c, p)."""
    from scipy.stats import binomtest
    y = np.asarray(y_true).astype(int)
    r1 = (np.asarray(p1) == y)
    r2 = (np.asarray(p2) == y)
    b = int((r1 & ~r2).sum())  # 1 right, 2 wrong
    c = int((~r1 & r2).sum())  # 1 wrong, 2 right
    if b + c == 0:
        return b, c, 1.0
    return b, c, float(binomtest(min(b, c), b + c, 0.5).pvalue)


def bootstrap_ci(y_true, y_pred, y_score, n_boot=1000, seed=SEED):
    """Percentile 95% CI for (acc, auc). Returns ((acc_lo, acc_hi), (auc_lo, auc_hi))."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true)
    n = len(y)
    accs, aucs = np.zeros(n_boot), np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        accs[i] = accuracy_score(y[idx], np.asarray(y_pred)[idx])
        try:
            aucs[i] = roc_auc_score(y[idx], np.asarray(y_score)[idx])
        except ValueError:
            aucs[i] = np.nan
    aucs = aucs[~np.isnan(aucs)]
    return ((float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))),
            (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))))


def run_stats(y_te):
    print("\n[4] paired stats on TEST (DeLong for AUC, exact McNemar for acc, "
          "bootstrap 95% CI)")
    pairs = [("bicameral_direct", "logreg", "Bicameral-direct vs logreg (can Bicameral train?)"),
             ("bicameral_hpo", "hgb_default", "Bicameral-HPO vs hgb-defaults (can Bicameral tune?)"),
             ("bicameral_hpo", "rs_hpo", "Bicameral-HPO vs randsearch-HPO (same budget)"),
             ("rs_hpo", "hgb_default", "RS-HPO vs hgb-defaults (tuning value?)")]
    for k1, k2, title in pairs:
        p1, s1 = TEST[k1]
        p2, s2 = TEST[k2]
        z, p_d = delong_test(y_te, s1, s2)
        b, c, p_m = mcnemar_test(y_te, p1, p2)
        (a1, u1), (a2, u2) = bootstrap_ci(y_te, p1, s1), bootstrap_ci(y_te, p2, s2)
        d1 = "SIG" if p_d < 0.05 else "n.s."
        d2 = "SIG" if p_m < 0.05 else "n.s."
        print(f"  {title}\n"
              f"    DeLong AUC: z={z:+.2f} p={p_d:.4f} {d1} | "
              f"McNemar acc: b={b} c={c} p={p_m:.4f} {d2}\n"
              f"    95% CI {k1}: acc [{a1[0]*100:.2f},{a1[1]*100:.2f}] "
              f"auc [{u1[0]:.4f},{u1[1]:.4f}] | {k2}: acc "
              f"[{a2[0]*100:.2f},{a2[1]*100:.2f}] auc [{u2[0]:.4f},{u2[1]:.4f}]")


# --------------------------------------------------------------------- SHAP
def run_shap(X_full, y_full, X_te, y_te, names, n_sample=2000):
    print("\n[5] SHAP explainability (HGB defaults, TreeExplainer)")
    try:
        import shap
    except ImportError:
        print("  shap not installed - skipped")
        return
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(X_te), size=min(n_sample, len(X_te)), replace=False)
    Xs, ys = X_te[idx], y_te[idx]
    m = HistGradientBoostingClassifier(random_state=SEED).fit(X_full, y_full)
    try:
        sv = np.asarray(shap.TreeExplainer(m)(Xs).values, dtype=float)
    except Exception as e:
        print(f"  TreeExplainer failed ({e}) - skipped")
        return
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(-mean_abs)
    print("  top-10 global mean|SHAP|:")
    for j in order[:10]:
        print(f"    {names[j]:28s} {mean_abs[j]:.4f}")
    pred = m.predict(Xs)
    tp = np.where((pred == 1) & (ys == 1))[0]
    tn = np.where((pred == 0) & (ys == 0))[0]
    fp = np.where((pred == 1) & (ys == 0))[0]
    fn = np.where((pred == 0) & (ys == 1))[0]
    print("  individual cases (top-3 |SHAP| features each):")
    for tag, arr in [("TP", tp), ("TN", tn), ("FP", fp), ("FN", fn)]:
        if len(arr) == 0:
            print(f"    {tag}: none in sample")
            continue
        i = arr[0]
        top = np.argsort(-np.abs(sv[i]))[:3]
        feats = ", ".join(f"{names[j]}={Xs[i, j]:.2f}(shap={sv[i, j]:+.3f})"
                          for j in top)
        print(f"    {tag} sample#{idx[i]} pred={pred[i]} true={ys[i]}: {feats}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--bicameral-evals", type=int, default=2000)
    ap.add_argument("--hpo-budget", type=int, default=108)
    ap.add_argument("--shap-n", type=int, default=2000)
    ap.add_argument("--no-shap", action="store_true")
    a = ap.parse_args()
    print(f"BRFSS2015 diabetes_binary 50/50 - Bicameral vs sklearn (seed {SEED})")
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te), names = load_data(a.data)
    base, scaled = run_baselines(((X_tr, y_tr), (X_va, y_va), (X_te, y_te)))
    Xtr_s, Xva_s, Xte_s, Xtr_full, ytr_full = scaled
    bicameral_direct = run_bicameral_direct((Xtr_s, Xva_s, Xte_s), y_tr, y_va, y_te,
                                max_evals=a.bicameral_evals)
    hpo = run_bicameral_hpo(X_tr, y_tr, X_va, y_va, X_te, y_te, budget=a.hpo_budget)

    print("\n================ TEST ACCURACY % (higher better) ================")
    rows = [("dummy", base["dummy"]), ("logreg", base["logreg"]),
            ("Bicameral-direct", bicameral_direct), ("randforest", base["randforest"]),
            ("hgb default", base["hgb_default"]), ("mlp", base["mlp"]),
            ("randsearch-HPO", hpo["rs_hpo"]), ("Bicameral-HPO", hpo["bicameral_hpo"])]
    for name, m in rows:
        print(f"  {name:14s} acc={m['acc']*100:6.2f}%  f1={m['f1']:.4f}  auc={m['auc']:.4f}")
    print("=================================================================")

    run_stats(y_te)
    if not a.no_shap:
        run_shap(Xtr_full, ytr_full, X_te, y_te, names, n_sample=a.shap_n)


if __name__ == "__main__":
    main()
