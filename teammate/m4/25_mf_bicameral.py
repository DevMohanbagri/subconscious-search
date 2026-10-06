#!/usr/bin/env python3
"""25 - Stage 5 Target 2 (PILOT): Bicameral tuning of the fuzzy P membership functions.

M4 delivers the doc's Target 2 (tune MF parameters for cases-detected within
a fixed screening budget). The FULL run needs M2's results/preds/val_final.csv
(calibrated P) -- not yet produced -- so this PILOT proves the tuning
machinery end-to-end against HGB P on the same rows, and re-runs on M2's P
later by swapping one loader line (see P_SOURCE below).

Why the pilot transfers: 18's fold A/B split is StratifiedKFold(y) with seed
0, which depends ONLY on the val targets -- identical rows here and in 18's
future run. Only P's scale differs, which is exactly what MF tuning absorbs.

Search: Bicameral 6-D [0,1] -> sorted P cut points (c1<d1<c2<d2<c3<d3) with
min-gap repair; chained trapezoids Low/Moderate/High/VeryHigh (BMI/Age/M/BP
sets FIXED; rules FIXED at fuzzy_rules.json v1 -- rule refinement is M3's).
Objective on fold A: max catch@20% - 0.25*(monotonicity violations + empty
tiers), with 18's EXACT metric definitions (lexsort top-k, strict monotone).
Baselines: default MFs + 30 random MF draws (best-on-A). Confirm on fold B.

Outputs: results/mf_tuning_pilot.csv, results/mf_tuned_p.json, figure
(figures/fig_mf_p.png). NO registry rows (consistent with 18_fuzzy_eval.py:
fuzzy tiers are not P(diabetes) models). NO fuzzy_rules_v2.json (M3's).

Protocol: train fit (HGB) / val folds A-B only, test sealed, deterministic.
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, "/home/user/subconscious-search")
from bicameral import Bicameral

from data_io import load, features, TARGET
from fuzzy import MamdaniFIS, load_rules, modifiable_score, VARIABLES, TIERS

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--budget", type=int, default=80)
ap.add_argument("--random-n", type=int, default=30)
A = ap.parse_args()

P_SOURCE = "pilot:HGB-raw"   # FULL run: "m2:results/preds/val_final.csv p_cal"
MIN_GAP = 0.02
N_CUTS = 6                   # c1 d1 c2 d2 c3 d3


def decode_cuts(x):
    """[0,1]^6 -> ordered P cut points with min-gap repair (soft)."""
    s = np.sort(np.asarray(x, float))
    s[0] = float(np.clip(s[0], 0.005, 0.9))
    for i in range(1, N_CUTS):
        s[i] = max(s[i], s[i - 1] + MIN_GAP)
    if s[-1] > 0.995:        # rescale span into bounds (preserves order)
        s = 0.005 + (s - s[0]) * (0.99 / (s[-1] - s[0]))
    return s


def p_sets(cuts):
    c1, d1, c2, d2, c3, d3 = (float(v) for v in cuts)
    return {"Low": (0, 0, c1, d1), "Moderate": (c1, d1, c2, d2),
            "High": (c2, d2, c3, d3), "VeryHigh": (c3, d3, 1, 1)}


DEFAULT_CUTS = np.array([0.06, 0.10, 0.20, 0.30, 0.40, 0.50])


def fis_for(cuts, rules):
    variables = dict(VARIABLES)
    variables["P"] = p_sets(cuts)
    return MamdaniFIS(rules, variables=variables)


def metrics_18(fis, X, y, P, idx):
    """18_fuzzy_eval.evaluate() replicated EXACTLY (same formulas)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score, _ = fis.infer({k: a[idx] for k, a in X.items()})
    if not np.all(np.isfinite(score)):
        return None
    t, yy, pp = fis.tier(score), y[idx], P[idx]
    obs = np.array([yy[t == i].mean() for i in range(4)])
    k = int(0.2 * len(yy))
    cap_f = yy[np.lexsort((-pp, -score))[:k]].sum() / yy.sum()
    cap_p = yy[np.argsort(-pp)[:k]].sum() / yy.sum()
    viol = sum(1 for i in range(3) if not (obs[i + 1] > obs[i]))
    empty = sum(1 for i in range(4) if not (t == i).any())
    return {"catch_f": float(cap_f), "catch_p": float(cap_p),
            "viol": viol, "empty": empty,
            "pr_f": float(average_precision_score(yy, score + 1e-6 * pp)),
            "pr_p": float(average_precision_score(yy, pp)),
            "obs": [None if np.isnan(v) else round(float(v), 4) for v in obs],
            "objective": float(cap_f - 0.25 * (viol + empty))}


def line(label, m):
    print(f"{label}: catch fuzzy {m['catch_f']:.3f} vs P {m['catch_p']:.3f} | "
          f"tiers {m['obs']} viol={m['viol']} empty={m['empty']} | "
          f"PR-AUC fuzzy {m['pr_f']:.4f} vs P {m['pr_p']:.4f}", flush=True)


def main():
    t_all = time.time()
    df, tr, va, te = load()
    assert te is None, "test must stay sealed"
    print(f"P_SOURCE={P_SOURCE} seed={A.seed} budget={A.budget}", flush=True)

    # ---- pilot P: HGB fit on TRAIN, predicted on VAL (raw, uncalibrated) ----
    cols = features(df)
    t0 = time.time()
    hgb = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=50, l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=30, random_state=A.seed)
    hgb.fit(df.loc[tr, cols].to_numpy(np.float32), df.loc[tr, TARGET].to_numpy())
    v = df.loc[va].reset_index(drop=True)
    y = v[TARGET].to_numpy(int)
    P = hgb.predict_proba(v[cols].to_numpy(np.float32))[:, 1]
    print(f"HGB fit+predict: {time.time()-t0:.0f}s  "
          f"val PR-AUC(P)={average_precision_score(y, P):.4f}", flush=True)

    X = {"P": P, "BMI": v.BMI.to_numpy(float), "Age": v.Age.to_numpy(float),
         "M": modifiable_score(v.BMI, v.PhysActivity, v.Fruits, v.Veggies),
         "HighBP": v.HighBP.to_numpy(), "HighChol": v.HighChol.to_numpy(),
         "HeartDiseaseorAttack": v.HeartDiseaseorAttack.to_numpy()}
    fold_a, fold_b = next(StratifiedKFold(2, shuffle=True, random_state=0)
                          .split(P[:, None], y))
    print(f"folds A/B: {len(fold_a)}/{len(fold_b)} "
          f"(== 18's future rows: StratifiedKFold depends on y only)", flush=True)
    rules = load_rules("fuzzy_rules.json")
    print(f"rules: {len(rules)} (v1, FIXED -- refinement is M3's)", flush=True)

    # ---- baseline: default MFs ----
    fis0 = fis_for(DEFAULT_CUTS, rules)
    m0a = metrics_18(fis0, X, y, P, fold_a)
    line("default fold A", m0a)

    # ---- Bicameral tuning on fold A ----
    t0 = time.time()

    def obj(x):
        m = metrics_18(fis_for(decode_cuts(x), rules), X, y, P, fold_a)
        return -(m["objective"] if m else -1.0)

    opt = Bicameral(N_CUTS, 0.0, 1.0, seed=A.seed)
    xb, fb, info = opt.optimize(obj, A.budget, local_search=False)
    cuts_b = decode_cuts(xb)
    m_ba = metrics_18(fis_for(cuts_b, rules), X, y, P, fold_a)
    print(f"[bicameral] evals={A.budget} time={time.time()-t0:.0f}s "
          f"objective(A)={m_ba['objective']:.4f} gens={info['gens']}", flush=True)
    line("tuned   fold A", m_ba)

    # ---- random-MF reference: best-on-A of random_n draws ----
    # NOTE: seed+1, NOT seed: Bicameral(42) and default_rng(42) share their first
    # uniform draws, which once made both methods "find" the same point.
    rng = np.random.default_rng(A.seed + 1)
    best = None
    for _ in range(A.random_n):
        c = decode_cuts(rng.uniform(0, 1, N_CUTS))
        m = metrics_18(fis_for(c, rules), X, y, P, fold_a)
        if m and (best is None or m["objective"] > best[1]["objective"]):
            best = (c, m)
    cuts_r, m_ra = best if best is not None else (DEFAULT_CUTS.copy(), m0a)
    if best is None:
        print("[random] all draws degenerate -> fell back to default cuts", flush=True)
    line("random  fold A", m_ra)

    # ---- confirm on fold B ----
    print("--- fold B (confirmation) ---", flush=True)
    m0b = metrics_18(fis0, X, y, P, fold_b)
    m_bb = metrics_18(fis_for(cuts_b, rules), X, y, P, fold_b)
    m_rb = metrics_18(fis_for(cuts_r, rules), X, y, P, fold_b)
    line("default fold B", m0b)
    line("tuned   fold B", m_bb)
    line("random  fold B", m_rb)

    print(f"\ndefault cuts: {DEFAULT_CUTS.round(3).tolist()}", flush=True)
    print(f"tuned   cuts: {cuts_b.round(3).tolist()}", flush=True)
    print(f"B verdict: dCatch(tuned-default)={m_bb['catch_f']-m0b['catch_f']:+.3f} "
          f"dCatch(random-default)={m_rb['catch_f']-m0b['catch_f']:+.3f}", flush=True)

    rows = []
    for tag, cuts, ma, mb in [("default", DEFAULT_CUTS, m0a, m0b),
                              ("bicameral", cuts_b, m_ba, m_bb),
                              ("random_bestA", cuts_r, m_ra, m_rb)]:
        rows.append({"set": tag, "cuts": "|".join(f"{v:.4f}" for v in cuts),
                     "catch_A": round(ma["catch_f"], 4), "viol_A": ma["viol"],
                     "prf_A": round(ma["pr_f"], 4),
                     "catch_B": round(mb["catch_f"], 4), "viol_B": mb["viol"],
                     "prf_B": round(mb["pr_f"], 4)})
    pd.DataFrame(rows).to_csv("results/mf_tuning_pilot.csv", index=False)
    json.dump({"P_source": P_SOURCE, "cuts": [float(v) for v in cuts_b],
               "P_sets": {k: list(v) for k, v in p_sets(cuts_b).items()},
               "foldB": {"catch_f": m_bb["catch_f"], "viol": m_bb["viol"],
                         "pr_f": m_bb["pr_f"]}},
              open("results/mf_tuned_p.json", "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from fuzzy import trapmf
    z = np.linspace(0, 1, 401)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, cuts, title in [(axes[0], DEFAULT_CUTS, "default P sets"),
                            (axes[1], cuts_b, "tuned P sets (pilot HGB-P)")]:
        for term, prm in p_sets(cuts).items():
            ax.plot(z, trapmf(z, *prm), label=term)
        ax.set_title(title)
        ax.set_xlabel("P")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("membership")
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/fig_mf_p.png", dpi=300)
    print(f"\nDone in {(time.time()-t_all)/60:.1f} min. Test sealed.", flush=True)


if __name__ == "__main__":
    main()
