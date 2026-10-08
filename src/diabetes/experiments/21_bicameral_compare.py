#!/usr/bin/env python3
"""Bicameral (our optimizer) vs sklearn on the frozen BRFSS data.

M4 optimizer-validation contribution: no changes to any existing
project file; this script only READS dedup.csv/splits.npz and APPENDS
rows to results/registry.csv (v2 schema).

Respects project rules: train/val ONLY (test split is structurally
withheld by data_io.load), frozen splits.npz untouched, hash-verified.

  [A] optimizer shootout x {all_21, leakage_free}: Bicameral-direct
      optimizes the EXACT sklearn-LR objective (balanced log-loss +
      C=1.0-equivalent L2, standardized features) with 2000 evals,
      vs refit sklearn LR (LBFGS). Any gap = pure optimizer difference.
  [B] tuner shootout (all_21): Bicameral-HPO vs same-budget
      RandomSearch on HGB hyperparams (4-D, 108 evals, val log-loss
      objective), vs the project's fixed-RF row as anchor.
      NOTE: final models fit on TRAIN only (no train+val refit: test
      is sealed and val is the eval split, so refit would contaminate).

Metrics: PR-AUC (project metric) + ROC-AUC + Brier + accuracy +
rec@P30/prec@R80/ECE (registry v2, via metrics.summarize).
Stats: DeLong (ROC-AUC), McNemar (accuracy), bootstrap CIs, paired
bootstrap CI on dPR-AUC. Bicameral imported from the optimizer repo.
"""

import sys
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, "/home/user/subconscious-search/src")
from bicameral import Bicameral
from ml_benchmark import delong_test, mcnemar_test, hgb_from_x

from data_io import load, features, log_result, TARGET
from metrics import summarize
import data_io

SEED = 42
EVALS_DIRECT = 2000
HPO_BUDGET = 108
N_BOOT = 1000

PRED = {}  # key -> (y_pred, y_proba) on val, for paired stats


def pr_auc(y, p):
    return float(average_precision_score(y, p))


def full_metrics(y, pred, proba):
    m = summarize(y, proba)
    return m


def metrics_line(name, y, pred, proba, extra=""):
    m = full_metrics(y, pred, proba)
    print(f"  {name:30s} PR-AUC {m['pr_auc']:.4f}  ROC-AUC {m['roc_auc']:.4f}  "
          f"Brier {m['brier']:.4f}  acc {m['accuracy']*100:6.2f}%  {extra}")
    return m


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_logreg(Xtr, ytr, Xva):
    m = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=2000, class_weight="balanced"))
    m.fit(Xtr, ytr)
    return m.predict(Xva), m.predict_proba(Xva)[:, 1]


def run_direct(Xtr, ytr, Xva, yva, tag, n_feat):
    """Bicameral-direct on the exact sklearn-LR objective (train subset)."""
    sc = StandardScaler().fit(Xtr)
    Xtrs, Xvas = sc.transform(Xtr), sc.transform(Xva)
    Xtr1 = np.hstack([Xtrs, np.ones((len(Xtrs), 1))])
    Xva1 = np.hstack([Xvas, np.ones((len(Xvas), 1))])
    n = len(Xtr1)
    n1 = int(ytr.sum())
    w1 = n / (2.0 * n1)  # class_weight='balanced', 2 classes
    w0 = n / (2.0 * (n - n1))
    lam = 0.5 / (1.0 * n)  # C=1.0 in mean-loss form

    def train_loss(w):
        p = sigmoid(Xtr1 @ w)
        eps = 1e-12
        wv = np.where(ytr == 1, w1, w0)
        return float((wv * (-(ytr * np.log(p + eps)
                              + (1 - ytr) * np.log(1 - p + eps)))).mean()
                     + lam * float(w @ w))

    print(f"\n[A-{tag}] Bicameral-direct: {Xtr1.shape[1]}-D LR weights on "
          f"train balanced log-loss ({EVALS_DIRECT} evals)")
    t0 = time.time()
    opt = Bicameral(Xtr1.shape[1], -5.0, 5.0, seed=SEED)
    w_best, best_tr, info = opt.optimize(train_loss, EVALS_DIRECT)
    dt = time.time() - t0
    p_va = sigmoid(Xva1 @ w_best)
    print(f"    train loss={best_tr:.5f} gens={info['gens']} "
          f"restarts={info['restarts']} bh={info['bh_phase']} time={dt:.1f}s")
    out = metrics_line("Bicameral-direct (LR w)", yva,
                       (p_va >= 0.5).astype(int), p_va, extra=f"{dt:.1f}s")
    PRED[f"direct_{tag}"] = ((p_va >= 0.5).astype(int), p_va)
    log_result(data_hash=data_io.DATA_HASH, script="21_bicameral_compare.py",
               feature_set=tag, n_features=n_feat,
               model="bicameral-direct (LR weights, 2000 evals)",
               params=f"balanced log-loss + L2 lam={lam:.3g}, bounds=[-5,5], seed={SEED}",
               seed=SEED, split="val", **out,
               notes="M4 optimizer shootout vs LR-LBFGS; test sealed")
    return out


def run_hpo(Xtr, ytr, Xva, yva, tag, n_feat):
    print(f"\n[B-{tag}] HPO on HGB: Bicameral vs RandomSearch "
          f"({HPO_BUDGET} evals each, val log-loss; fit on train)")

    def val_loss(x):
        m, _ = hgb_from_x(x)
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xva)[:, 1]
        eps = 1e-12
        return float(-(yva * np.log(p + eps)
                       + (1 - yva) * np.log(1 - p + eps)).mean())

    lo = np.array([-3.0, 2.0, 0.0, 0.0])
    hi = np.array([-0.3, 12.0, 2.5, 20.0])

    t0 = time.time()
    opt = Bicameral(4, lo, hi, seed=SEED)
    x_bic, v_bic, _ = opt.optimize(val_loss, HPO_BUDGET, local_search=False)
    t_bic = time.time() - t0
    m_bic, p_bic = hgb_from_x(x_bic)
    m_bic.fit(Xtr, ytr)
    print(f"    Bicameral best val={v_bic:.5f} params={p_bic} time={t_bic:.1f}s")
    out_bic = metrics_line("Bicameral-HPO (hgb)", yva, m_bic.predict(Xva),
                           m_bic.predict_proba(Xva)[:, 1], extra=f"{t_bic:.1f}s")
    PRED["hpo_bic"] = (m_bic.predict(Xva), m_bic.predict_proba(Xva)[:, 1])

    t0 = time.time()
    rng = np.random.default_rng(SEED)
    best, best_x = np.inf, None
    for _ in range(HPO_BUDGET):
        x = rng.uniform(lo, hi)
        v = val_loss(x)
        if v < best:
            best, best_x = v, x.copy()
    t_rs = time.time() - t0
    m_rs, p_rs = hgb_from_x(best_x)
    m_rs.fit(Xtr, ytr)          # train-only fit (val stays a pure eval split)
    print(f"    RS  best val={best:.5f} params={p_rs} time={t_rs:.1f}s")
    out_rs = metrics_line("randsearch-HPO (hgb)", yva, m_rs.predict(Xva),
                          m_rs.predict_proba(Xva)[:, 1], extra=f"{t_rs:.1f}s")
    PRED["hpo_rs"] = (m_rs.predict(Xva), m_rs.predict_proba(Xva)[:, 1])

    for key, out, prm, tt in [("hpo_bic", out_bic, p_bic, t_bic),
                              ("hpo_rs", out_rs, p_rs, t_rs)]:
        log_result(data_hash=data_io.DATA_HASH, script="21_bicameral_compare.py",
                   feature_set=tag, n_features=n_feat,
                   model=f"{'bicameral-HPO' if key == 'hpo_bic' else 'randsearch-HPO'} (hgb, 108 evals)",
                   params=str(prm), seed=SEED, split="val", **out,
                   notes=f"M4 tuner shootout, val log-loss objective, fit-train-only, time={tt:.0f}s; test sealed")
    return out_bic, out_rs


def bootstrap_cis(y, pred, proba):
    from sklearn.metrics import accuracy_score
    rng = np.random.default_rng(SEED)
    n = len(y)
    P, R, A = np.zeros(N_BOOT), np.zeros(N_BOOT), np.zeros(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        P[i] = pr_auc(y[idx], proba[idx])
        R[i] = roc_auc_score(y[idx], proba[idx])
        A[i] = accuracy_score(y[idx], pred[idx])
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return q(P), q(R), q(A)


def paired_dpr_ci(y, p1, p2):
    """Paired bootstrap 95% CI on dPR-AUC (no DeLong analogue for PR)."""
    rng = np.random.default_rng(SEED)
    n = len(y)
    d = np.zeros(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        d[i] = pr_auc(y[idx], p1[idx]) - pr_auc(y[idx], p2[idx])
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def run_stats(yva, pairs):
    print("\n[C] paired stats on VAL (DeLong ROC-AUC, McNemar acc, "
          "paired-bootstrap dPR-AUC, bootstrap CIs)")
    for k1, k2, title in pairs:
        pr1, pb1 = PRED[k1]
        pr2, pb2 = PRED[k2]
        z, p_d = delong_test(yva, pb1, pb2)
        b, c, p_m = mcnemar_test(yva, pr1, pr2)
        lo, hi = paired_dpr_ci(yva, pb1, pb2)
        ci1, ci2 = bootstrap_cis(yva, pr1, pb1), bootstrap_cis(yva, pr2, pb2)
        print(f"  {title}\n"
              f"    DeLong ROC: z={z:+.2f} p={p_d:.4f} {'SIG' if p_d < 0.05 else 'n.s.'} | "
              f"McNemar acc: b={b} c={c} p={p_m:.4f} {'SIG' if p_m < 0.05 else 'n.s.'}\n"
              f"    dPR-AUC 95% CI: [{lo:+.4f},{hi:+.4f}] "
              f"({'excludes 0' if lo > 0 or hi < 0 else 'includes 0'})\n"
              f"    CI {k1}: pr [{ci1[0][0]:.4f},{ci1[0][1]:.4f}] "
              f"roc [{ci1[1][0]:.4f},{ci1[1][1]:.4f}] acc [{ci1[2][0]*100:.2f},{ci1[2][1]*100:.2f}]\n"
              f"    CI {k2}: pr [{ci2[0][0]:.4f},{ci2[0][1]:.4f}] "
              f"roc [{ci2[1][0]:.4f},{ci2[1][1]:.4f}] acc [{ci2[2][0]*100:.2f},{ci2[2][1]*100:.2f}]")


def main():
    df, tr, va, te = load()
    print(f"frozen data: hash {data_io.DATA_HASH[:12]}... "
          f"train/val = {len(tr)}/{len(va)} (test structurally withheld: {te is None})")
    results = {}
    for drop, tag in [(False, "all_21"), (True, "leakage_free")]:
        cols = features(df, drop_leakage=drop)
        Xtr, ytr = df.loc[tr, cols].to_numpy(float), df.loc[tr, TARGET].to_numpy(int)
        Xva, yva = df.loc[va, cols].to_numpy(float), df.loc[va, TARGET].to_numpy(int)
        prev = float(yva.mean())
        print(f"\n=== {tag} ({len(cols)} features) ===")
        print(f"PR-AUC baseline (prevalence): {prev:.4f}")
        print(f"majority-class accuracy:      {1 - prev:.4f}   <- the accuracy trap")
        print(f"\n[A-{tag}] sklearn LR refit (anchor; registry says "
              f"{'0.4191' if tag == 'all_21' else '0.3906'})")
        t0 = time.time()
        pred_lr, prob_lr = fit_logreg(Xtr, ytr, Xva)
        results[f"lr_{tag}"] = metrics_line("logreg refit (LBFGS)", yva, pred_lr, prob_lr,
                                            extra=f"{time.time()-t0:.1f}s")
        PRED[f"lr_{tag}"] = (pred_lr, prob_lr)
        results[f"direct_{tag}"] = run_direct(Xtr, ytr, Xva, yva, tag, len(cols))

    cols = features(df, drop_leakage=False)
    Xtr = df.loc[tr, cols].to_numpy(float)
    ytr = df.loc[tr, TARGET].to_numpy(int)
    Xva = df.loc[va, cols].to_numpy(float)
    yva = df.loc[va, TARGET].to_numpy(int)
    print("\n[B-all_21] sklearn RF refit (anchor; registry says 0.4426)")
    t0 = time.time()
    m = RandomForestClassifier(n_estimators=300, min_samples_leaf=20,
                               class_weight="balanced", n_jobs=2, random_state=42)
    m.fit(Xtr, ytr)
    pred_rf, prob_rf = m.predict(Xva), m.predict_proba(Xva)[:, 1]
    results["rf_all_21"] = metrics_line("randforest refit (300)", yva, pred_rf, prob_rf,
                                        extra=f"{time.time()-t0:.1f}s")
    PRED["rf_all_21"] = (pred_rf, prob_rf)
    results["hpo_bic"], results["hpo_rs"] = run_hpo(Xtr, ytr, Xva, yva, "all_21", len(cols))

    print("\n================ VAL (PR-AUC / ROC-AUC / Brier / acc) ================")
    for name in ["lr_all_21", "direct_all_21", "rf_all_21", "hpo_rs", "hpo_bic",
                 "lr_leakage_free", "direct_leakage_free"]:
        m_ = results[name]
        print(f"  {name:18s} PR-AUC {m_['pr_auc']:.4f}  ROC-AUC {m_['roc_auc']:.4f}  "
              f"Brier {m_['brier']:.4f}  acc {m_['accuracy']*100:6.2f}%")
    print("======================================================================")

    run_stats(yva, [
        ("direct_all_21", "lr_all_21", "Bicameral-direct vs LR (all_21: can Bicameral train?)"),
        ("direct_leakage_free", "lr_leakage_free", "Bicameral-direct vs LR (leakage_free)"),
        ("hpo_bic", "hpo_rs", "Bicameral-HPO vs randsearch-HPO (same budget)"),
        ("hpo_bic", "rf_all_21", "Bicameral-HPO vs project RF (tuning value?)"),
    ])
    print("\nDone. Test split was never touched. Registry appended with 4 new rows.")


if __name__ == "__main__":
    main()
