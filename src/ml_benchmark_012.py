#!/usr/bin/env python3
"""Bicameral on real data: BRFSS 2015 Diabetes Health Indicators (012 3-class).

Same protocol as ml_benchmark.py (binary 50/50), ported to the 3-class
target Diabetes_012 (0 = no diabetes 84.2%, 1 = prediabetes 1.8%,
2 = diabetes 13.9%; 253,680 rows, 21 features). Stratified 70/15/15,
seed 42. Test set touched exactly once per method, at the end.

Multiclass adaptations (binary pieces reused where valid):
  metrics - accuracy + macro-F1 + macro-AUC (one-vs-rest) + per-class F1
  Bicameral-direct - 66-D multinomial softmax weights on val cross-entropy
  HPO objective    - multiclass val log-loss (same 4-D HGB space, 108 evals)
  paired stats     - per-class OvR DeLong (reuses ml_benchmark.delong_test),
                     Stuart-Maxwell test for accuracy (the k-class McNemar:
                     Stuart 1955, Maxwell 1970), bootstrap 95% CIs
  SHAP             - per-class global rankings + correct/confused cases
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
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix)

from bicameral import Bicameral
from ml_benchmark import delong_test, hgb_from_x

SEED = 42
DATA = "data/diabetes_012_health_indicators_BRFSS2015.csv"
CLASSES = [0, 1, 2]

# test-set predictions/probas per method, filled by report(key=...)
TEST = {}


def load_data(path=DATA):
    df = pd.read_csv(path)
    names = [str(c) for c in df.columns[1:]]
    y = df.iloc[:, 0].to_numpy(dtype=int)
    X = df.iloc[:, 1:].to_numpy(dtype=float)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED)
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
    dist = {c: int((y_te == c).sum()) for c in CLASSES}
    print(f"train/val/test: {len(X_tr)}/{len(X_va)}/{len(X_te)} "
          f"({X.shape[1]} features, test class counts {dist})")
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te), names


def mc_auc(y_true, proba):
    """Macro one-vs-rest AUC (nan if a class is missing)."""
    try:
        return float(roc_auc_score(y_true, proba, multi_class="ovr"))
    except ValueError:
        return float("nan")


def report(name, y_true, y_pred, proba=None, extra="", key=None):
    y_pred = np.asarray(y_pred)
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    f1c = f1_score(y_true, y_pred, average=None, labels=CLASSES)
    auc = mc_auc(y_true, proba) if proba is not None else float("nan")
    print(f"  {name:28s} acc={acc*100:6.2f}%  macroF1={f1m:.4f}  "
          f"auc={auc:.4f}  F1={np.array2string(f1c, precision=3, separator='/')}  {extra}")
    if key is not None:
        TEST[key] = (y_pred, None if proba is None
                     else np.asarray(proba, dtype=float))
    return {"acc": acc, "f1m": f1m, "f1c": f1c, "auc": auc}


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(np.clip(z, -30, 30))
    return e / e.sum(axis=1, keepdims=True)


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
    out["dummy"] = report("dummy (majority)", y_te, m.predict(X_te),
                          extra=f"{time.time()-t0:.1f}s")
    t0 = time.time()
    m = LogisticRegression(max_iter=2000).fit(Xtr_full_s, ytr_full)
    out["logreg"] = report("logreg (LBFGS)", y_te, m.predict(Xte_s),
                           m.predict_proba(Xte_s), extra=f"{time.time()-t0:.1f}s",
                           key="logreg")
    t0 = time.time()
    m = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    m.fit(Xtr_full, ytr_full)
    out["randforest"] = report("randomforest (300)", y_te, m.predict(X_te),
                               m.predict_proba(X_te), extra=f"{time.time()-t0:.1f}s",
                               key="randforest")
    t0 = time.time()
    m = HistGradientBoostingClassifier(random_state=SEED).fit(Xtr_full, ytr_full)
    out["hgb_default"] = report("hgb (defaults)", y_te, m.predict(X_te),
                                m.predict_proba(X_te), extra=f"{time.time()-t0:.1f}s",
                                key="hgb_default")
    t0 = time.time()
    m = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                      random_state=SEED).fit(Xtr_full_s, ytr_full)
    out["mlp"] = report("mlp (64x32, adam)", y_te, m.predict(Xte_s),
                        m.predict_proba(Xte_s), extra=f"{time.time()-t0:.1f}s",
                        key="mlp")
    return out, (Xtr_s, Xva_s, Xte_s, Xtr_full, ytr_full)


def run_bicameral_direct(splits_scaled, y_tr, y_va, y_te, max_evals=2000):
    Xtr_s, Xva_s, Xte_s = splits_scaled
    d = Xtr_s.shape[1]
    Xva1 = np.hstack([Xva_s, np.ones((len(Xva_s), 1))])
    Xte1 = np.hstack([Xte_s, np.ones((len(Xte_s), 1))])
    Yva = np.eye(3)[y_va]

    def val_xent(w):
        P = softmax(Xva1 @ w.reshape(d + 1, 3))
        eps = 1e-12
        return float(-(Yva * np.log(P + eps)).sum(axis=1).mean()
                     + 1e-4 * float(w @ w))

    print(f"\n[2] Bicameral-direct: {(d+1)*3}-D softmax weights on val xent "
          f"({max_evals} evals)")
    t0 = time.time()
    opt = Bicameral((d + 1) * 3, -5.0, 5.0, seed=SEED)
    w_best, best_val, info = opt.optimize(val_xent, max_evals)
    dt = time.time() - t0
    P_te = softmax(Xte1 @ w_best.reshape(d + 1, 3))
    print(f"    val xent={best_val:.5f} gens={info['gens']} "
          f"restarts={info['restarts']} bh={info['bh_phase']} time={dt:.1f}s")
    out = report("Bicameral-direct (softmax w)", y_te, P_te.argmax(axis=1), P_te,
                 extra=f"{dt:.1f}s", key="bicameral_direct")
    return out


def run_bicameral_hpo(X_tr, y_tr, X_va, y_va, X_te, y_te, budget=108):
    print(f"\n[3] HPO on HistGradientBoosting: Bicameral vs RandomSearch "
          f"({budget} evals each, multiclass val log-loss)")

    def val_loss(x):
        m, _ = hgb_from_x(x)
        m.fit(X_tr, y_tr)
        P = m.predict_proba(X_va)
        eps = 1e-12
        return float(-np.log(P[np.arange(len(y_va)), y_va] + eps).mean())

    lo = np.array([-3.0, 2.0, 0.0, 0.0])
    hi = np.array([-0.3, 12.0, 2.5, 20.0])

    # Bicameral (local search OFF: int-rounded HPO landscape is stepwise)
    t0 = time.time()
    opt = Bicameral(4, lo, hi, seed=SEED)
    x_bic, v_bic, info = opt.optimize(val_loss, budget, local_search=False)
    t_bic = time.time() - t0
    m_bic, p_bic = hgb_from_x(x_bic)
    m_bic.fit(np.vstack([X_tr, X_va]), np.concatenate([y_tr, y_va]))
    print(f"    Bicameral best val={v_bic:.5f} params={p_bic} time={t_bic:.1f}s")
    out_bic = report("Bicameral-HPO (hgb)", y_te, m_bic.predict(X_te),
                     m_bic.predict_proba(X_te), extra=f"{t_bic:.1f}s",
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
                    m_rs.predict_proba(X_te), extra=f"{t_rs:.1f}s",
                    key="rs_hpo")
    return {"bicameral_hpo": out_bic, "rs_hpo": out_rs}


# ---------------------------------------------------------------- paired stats
def stuart_maxwell_test(p1, p2, labels=CLASSES):
    """Stuart-Maxwell test for paired k-class accuracies (the k-class
    McNemar; Stuart 1955, Maxwell 1970). Returns (chi2, p); df = k-1."""
    from scipy.stats import chi2 as chi2_dist
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    k = len(labels)
    N = np.zeros((k, k))
    for a in range(k):
        for b in range(k):
            N[a, b] = ((p1 == labels[a]) & (p2 == labels[b])).sum()
    d = N.sum(axis=1) - N.sum(axis=0)  # row - col marginals
    V = np.zeros((k, k))
    for i in range(k):
        V[i, i] = N[i, :].sum() + N[:, i].sum() - 2.0 * N[i, i]
        for j in range(k):
            if j != i:
                V[i, j] = -(N[i, j] + N[j, i])
    d, V = d[:k-1], V[:k-1, :k-1]
    stat = float(d @ np.linalg.pinv(V) @ d)
    return stat, float(chi2_dist.sf(stat, k - 1))


def bootstrap_ci(y_true, y_pred, proba, n_boot=1000, seed=SEED):
    """Percentile 95% CIs for (acc, macro-F1, macro-AUC)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true)
    n = len(y)
    accs = np.zeros(n_boot)
    f1s = np.zeros(n_boot)
    aucs = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        accs[i] = accuracy_score(y[idx], np.asarray(y_pred)[idx])
        f1s[i] = f1_score(y[idx], np.asarray(y_pred)[idx], average="macro")
        aucs[i] = mc_auc(y[idx], np.asarray(proba)[idx])
    aucs = aucs[~np.isnan(aucs)]
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return q(accs), q(f1s), q(aucs)


def run_stats(y_te):
    print("\n[4] paired stats on TEST (per-class OvR DeLong for AUC, "
          "Stuart-Maxwell for acc, bootstrap 95% CIs)")
    pairs = [("bicameral_direct", "logreg", "Bicameral-direct vs logreg (can Bicameral train?)"),
             ("bicameral_hpo", "hgb_default", "Bicameral-HPO vs hgb-defaults (can Bicameral tune?)"),
             ("bicameral_hpo", "rs_hpo", "Bicameral-HPO vs randsearch-HPO (same budget)"),
             ("rs_hpo", "hgb_default", "RS-HPO vs hgb-defaults (tuning value?)")]
    for k1, k2, title in pairs:
        p1, P1 = TEST[k1]
        p2, P2 = TEST[k2]
        dl = []
        for c in CLASSES:
            z, p = delong_test((y_te == c).astype(int), P1[:, c], P2[:, c])
            dl.append(f"c{c}: z={z:+.2f} p={p:.4f}{' SIG' if p < 0.05 else ''}")
        sm, p_sm = stuart_maxwell_test(p1, p2)
        ci1 = bootstrap_ci(y_te, p1, P1)
        ci2 = bootstrap_ci(y_te, p2, P2)
        print(f"  {title}\n"
              f"    DeLong OvR AUC: {' | '.join(dl)}\n"
              f"    Stuart-Maxwell acc: chi2={sm:.2f} p={p_sm:.4f} "
              f"{'SIG' if p_sm < 0.05 else 'n.s.'}\n"
              f"    95% CI {k1}: acc [{ci1[0][0]*100:.2f},{ci1[0][1]*100:.2f}] "
              f"macroF1 [{ci1[1][0]:.4f},{ci1[1][1]:.4f}] auc [{ci1[2][0]:.4f},{ci1[2][1]:.4f}]\n"
              f"    95% CI {k2}: acc [{ci2[0][0]*100:.2f},{ci2[0][1]*100:.2f}] "
              f"macroF1 [{ci2[1][0]:.4f},{ci2[1][1]:.4f}] auc [{ci2[2][0]:.4f},{ci2[2][1]:.4f}]")


# --------------------------------------------------------------------- SHAP
def run_shap(X_full, y_full, X_te, y_te, names, n_sample=2000):
    print("\n[5] SHAP explainability (HGB defaults, TreeExplainer, per-class)")
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
    if sv.ndim != 3 or sv.shape[2] != 3:  # expect (n, p, 3)
        print(f"  unexpected SHAP shape {sv.shape} - skipped")
        return
    mean_abs = np.abs(sv).mean(axis=(0, 2))
    order = np.argsort(-mean_abs)
    print("  top-10 global mean|SHAP| (mean over classes):")
    for j in order[:10]:
        print(f"    {names[j]:28s} {mean_abs[j]:.4f}")
    for c in CLASSES:
        mc = np.abs(sv[:, :, c]).mean(axis=0)
        top = np.argsort(-mc)[:5]
        feats = ", ".join(f"{names[j]}={mc[j]:.3f}" for j in top)
        print(f"  class-{c} top-5: {feats}")
    pred = m.predict(Xs)
    ok0 = np.where((pred == 0) & (ys == 0))[0]
    okm = np.where((pred == ys) & (ys != 0))[0]
    err = np.where(pred != ys)[0]
    print("  individual cases (top-3 |SHAP| wrt PREDICTED class):")
    cases = [("OK-0", ok0[:1]), ("OK-minority", okm[:1]), ("ERR", err[:2])]
    for tag, arr in cases:
        for i in arr:
            top = np.argsort(-np.abs(sv[i, :, pred[i]]))[:3]
            feats = ", ".join(f"{names[j]}={Xs[i, j]:.2f}(shap={sv[i, j, pred[i]]:+.3f})"
                              for j in top)
            print(f"    {tag} sample#{idx[i]} pred={pred[i]} true={ys[i]}: {feats}")
        if len(arr) == 0:
            print(f"    {tag}: none in sample")


def run_confusions(y_te):
    print("\n[6] confusion matrices (test; rows=true, cols=pred)")
    for key in ["logreg", "bicameral_direct", "randforest", "hgb_default",
                "mlp", "rs_hpo", "bicameral_hpo"]:
        p, _ = TEST[key]
        C = confusion_matrix(y_te, p, labels=CLASSES)
        rows = " / ".join("[" + " ".join(f"{v:6d}" for v in row) + "]" for row in C)
        print(f"  {key:16s} {rows}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--bicameral-evals", type=int, default=2000)
    ap.add_argument("--hpo-budget", type=int, default=108)
    ap.add_argument("--shap-n", type=int, default=2000)
    ap.add_argument("--no-shap", action="store_true")
    a = ap.parse_args()
    print(f"BRFSS2015 diabetes_012 3-class (84/14/2) - Bicameral vs sklearn (seed {SEED})")
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te), names = load_data(a.data)
    base, scaled = run_baselines(((X_tr, y_tr), (X_va, y_va), (X_te, y_te)))
    Xtr_s, Xva_s, Xte_s, Xtr_full, ytr_full = scaled
    bicameral_direct = run_bicameral_direct((Xtr_s, Xva_s, Xte_s), y_tr, y_va, y_te,
                                max_evals=a.bicameral_evals)
    hpo = run_bicameral_hpo(X_tr, y_tr, X_va, y_va, X_te, y_te, budget=a.hpo_budget)

    print("\n================ TEST (acc / macro-F1 / macro-AUC) ================")
    rows = [("dummy", base["dummy"]), ("logreg", base["logreg"]),
            ("Bicameral-direct", bicameral_direct), ("randforest", base["randforest"]),
            ("hgb default", base["hgb_default"]), ("mlp", base["mlp"]),
            ("randsearch-HPO", hpo["rs_hpo"]), ("Bicameral-HPO", hpo["bicameral_hpo"])]
    for name, m in rows:
        print(f"  {name:14s} acc={m['acc']*100:6.2f}%  macroF1={m['f1m']:.4f}  "
              f"auc={m['auc']:.4f}  F1={np.array2string(m['f1c'], precision=3, separator='/')}")
    print("=================================================================")

    run_stats(y_te)
    run_confusions(y_te)
    if not a.no_shap:
        run_shap(Xtr_full, ytr_full, X_te, y_te, names, n_sample=a.shap_n)


if __name__ == "__main__":
    main()
