#!/usr/bin/env python3
"""Multi-seed HPO: Bicameral-HPO vs RandomSearch-HPO x {42,101,202,303,404}.

M4 optimizer-validation companion to 21_bicameral_compare.py. Reads only
dedup.csv/splits.npz; appends 12 rows to results/registry.csv (v2 schema).

Design (matches pre-registered rules in results/00_questions.md):
  - same budget 108 evals each, per seed; LEG seeds {s1}=101,{s2}=202,
    {s3}=303,{s4}=404 (RS leg uses the SAME seed each run);
  - Bicameral objective = val log-loss; RS identical objective;
  - fit-train-only (no train+val refit; test sealed);
  - all_21 for all 5 seeds + leakage_free at 42 (LF = seed-42 only,
    pre-registered as secondary).

RESUME: a pair counts as done only if BOTH legs are in the registry (same
script name, same feature_set, same seed). Re-running skips finished pairs
and reads their PR-AUCs back from the registry, so an interrupted run
resumes EXACTLY (deterministic seeds: identical numbers). Rows are keyed
off the registry itself -- no separate state file to lose.

Decision rule (pre-registered): mean dPR-AUC over 5 seeds with Welch
t-test: |t| with p<0.05 and |mean_d|>0.002 -> win/loss, else tie.
Discrimination rule: measured only via val->test generalization AFTER
99_final_test.py opens test; until then marked UNKNOWN.
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

sys.path.insert(0, "/home/user/subconscious-search/src")
from bicameral import Bicameral
from ml_benchmark import hgb_from_x

from data_io import load, features, log_result, TARGET
from metrics import summarize
import data_io

BUDGET = 108
SEEDS = [42, 101, 202, 303, 404]
SCRIPT = "22_bicameral_hpo_seeds.py"


def completed_pairs():
    """{(feature_set, seed): {'bic': pr_auc, 'rs': pr_auc}} for finished pairs."""
    done = {}
    reg = Path("results/registry.csv")
    if not reg.exists():
        return done
    with reg.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("script") != SCRIPT:
                continue
            m = row.get("model", "")
            if "(hgb, 108 evals)" not in m:
                continue
            if m.startswith("bicameral-HPO"):
                leg = "bic"
            elif m.startswith("randsearch-HPO"):
                leg = "rs"
            else:
                continue
            try:
                key = (row["feature_set"], int(row["seed"]))
                pr = float(row["pr_auc"])
            except (KeyError, ValueError):
                continue
            done.setdefault(key, {})[leg] = pr
    return {k: v for k, v in done.items() if set(v) == {"bic", "rs"}}


def run_pair(Xtr, ytr, Xva, yva, seed, tag, n_feat):
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
    opt = Bicameral(4, lo, hi, seed=seed)
    x_bic, _, _ = opt.optimize(val_loss, BUDGET, local_search=False)
    t_bic = time.time() - t0
    m_bic, p_bic = hgb_from_x(x_bic)
    m_bic.fit(Xtr, ytr)
    pb = m_bic.predict_proba(Xva)[:, 1]
    pr_b = float(average_precision_score(yva, pb))

    t0 = time.time()
    rng = np.random.default_rng(seed)
    best, best_x = np.inf, None
    for _ in range(BUDGET):
        x = rng.uniform(lo, hi)
        v = val_loss(x)
        if v < best:
            best, best_x = v, x.copy()
    t_rs = time.time() - t0
    m_rs, p_rs = hgb_from_x(best_x)
    m_rs.fit(Xtr, ytr)
    pr = m_rs.predict_proba(Xva)[:, 1]
    pr_r = float(average_precision_score(yva, pr))

    print(f"  seed {seed}: Bicameral {pr_b:.4f} ({t_bic:5.0f}s, {p_bic}) | "
          f"RS {pr_r:.4f} ({t_rs:5.0f}s, {p_rs}) | d={pr_b - pr_r:+.4f}",
          flush=True)
    mb, mr = summarize(yva, pb), summarize(yva, pr)
    log_result(data_hash=data_io.DATA_HASH, script=SCRIPT,
               feature_set=tag, n_features=n_feat,
               model="bicameral-HPO (hgb, 108 evals)", params=str(p_bic), seed=seed,
               split="val", **mb,
               notes=f"M4 multi-seed HPO; fit-train-only; time={t_bic:.0f}s; test sealed")
    log_result(data_hash=data_io.DATA_HASH, script=SCRIPT,
               feature_set=tag, n_features=n_feat,
               model="randsearch-HPO (hgb, 108 evals)", params=str(p_rs), seed=seed,
               split="val", **mr,
               notes=f"M4 multi-seed HPO; fit-train-only; time={t_rs:.0f}s; test sealed")
    return pr_b, pr_r, t_bic, t_rs


def main():
    df, tr, va, te = load()
    print(f"frozen data: hash {data_io.DATA_HASH[:12]}... "
          f"(test structurally withheld: {te is None})", flush=True)
    done = completed_pairs()
    if done:
        print(f"resuming: {len(done)} finished pair(s) found in registry, skipping",
              flush=True)
    n_run = 0

    print("\n=== all_21 x 5 seeds ===", flush=True)
    cols = features(df, drop_leakage=False)
    Xtr = df.loc[tr, cols].to_numpy(float)
    ytr = df.loc[tr, TARGET].to_numpy(int)
    Xva = df.loc[va, cols].to_numpy(float)
    yva = df.loc[va, TARGET].to_numpy(int)
    bics, rss, tbs, trs = [], [], [], []
    for s in SEEDS:
        if ("all_21", s) in done:
            b, r = done[("all_21", s)]["bic"], done[("all_21", s)]["rs"]
            print(f"  seed {s}: RESUMED Bicameral {b:.4f} | RS {r:.4f} | "
                  f"d={b - r:+.4f}", flush=True)
            tb, tr_ = float("nan"), float("nan")
        else:
            b, r, tb, tr_ = run_pair(Xtr, ytr, Xva, yva, s, "all_21", len(cols))
            n_run += 1
        bics.append(b)
        rss.append(r)
        tbs.append(tb)
        trs.append(tr_)

    d = np.array(bics) - np.array(rss)
    from scipy import stats
    t, p = stats.ttest_ind(bics, rss, equal_var=False)
    print(f"\nBicameral mean {np.mean(bics):.4f}+-{np.std(bics, ddof=1):.4f} | "
          f"RS mean {np.mean(rss):.4f}+-{np.std(rss, ddof=1):.4f} | "
          f"mean_d={d.mean():+.4f} Welch t={t:+.2f} p={p:.4f}")
    verdict = ("bicameral" if (p < 0.05 and d.mean() > 0.002)
               else ("randsearch" if (p < 0.05 and d.mean() < -0.002) else "TIE"))
    print(f"VERDICT (pre-registered rule): {verdict}")
    print(f"time: Bicameral {np.nanmean(tbs):.0f}s/seed, RS {np.nanmean(trs):.0f}s/seed")

    print("\n=== leakage_free x seed 42 (secondary) ===", flush=True)
    cols = features(df, drop_leakage=True)
    Xtr = df.loc[tr, cols].to_numpy(float)
    ytr = df.loc[tr, TARGET].to_numpy(int)
    Xva = df.loc[va, cols].to_numpy(float)
    yva = df.loc[va, TARGET].to_numpy(int)
    if ("leakage_free", 42) in done:
        b, r = done[("leakage_free", 42)]["bic"], done[("leakage_free", 42)]["rs"]
        print(f"  seed 42: RESUMED Bicameral {b:.4f} | RS {r:.4f} | d={b - r:+.4f}",
              flush=True)
    else:
        run_pair(Xtr, ytr, Xva, yva, 42, "leakage_free", len(cols))
        n_run += 1
    print(f"\nDone. Test split was never touched. Registry: {2 * n_run} new rows "
          f"({len(done)} pair(s) resumed).")


if __name__ == "__main__":
    main()
