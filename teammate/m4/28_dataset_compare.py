#!/usr/bin/env python3
"""28 - 50/50 vs imbalanced-full: does dataset balance change the optimizer verdict?

OUR-REPO analysis (not a friend deliverable: runs from here, writes nothing
to the friend's registry). Motivation: on the friend's imbalanced data every
optimizer comparison ties (Bicameral~RS, NSGA-II>scalarized, greedy wins).
Is the tie an artifact of the imbalanced task, or dataset-invariant?

Forensics (verified): the 50/50 file is a balanced subsample of the SAME
binary task (100% of sampled positives match 012==2 feature-patterns, 100%
of sampled negatives match 012==0; small overlap = contradictory patterns
shared across classes). Isolated variable: prevalence (50% vs 15.3%) +
sample size (70,692 vs 229,474, noted as secondary).

Design (same duel, two datasets, paired seeds):
  friend-full  frozen splits via the friend's data_io (test sealed); duel
               pairs for seeds 42/101/202 REUSED from 22's banked registry
               rows (same code+seeds+data = rerun-identical by determinism,
               proven bit-exact at seed 42); plain-LR anchor refit fresh (~1s).
  50/50        own stratified 70/15/15 split (random_state=42, same two-step
               code shape as 01_stage0_split.py); test held out, never used;
               duel pairs run FRESH (Bicameral-HPO vs RS-HPO, 108 evals,
               hgb_from_x, val log-loss, fit-train-only -- 22's run_pair
               verbatim); plain-LR anchor fresh.
Metrics reported on both: accuracy + PR-AUC + ROC-AUC + Brier (accuracy is
meaningful at 50%, a trap at 15.3% -- shown, not just claimed).
Resume: completed 50/50 pairs are read back from the output CSV and skipped.
Outputs: teammate/m4_results/analysis/dataset_compare.csv (appended per pair).
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/home/user/subconscious-search")          # bicameral, ml_benchmark
sys.path.insert(0, "/home/user/diabetes-project")             # friend data_io (frozen splits)
from bicameral import Bicameral
from ml_benchmark import hgb_from_x

OUT = HERE.parent / "m4_results" / "analysis" / "dataset_compare.csv"
SEEDS = [42, 101, 202]
BUDGET = 108
REGISTRY_MIRROR = HERE.parent / "m4_results" / "tables" / "registry.csv"


def metrics(y, p):
    return {"pr_auc": round(float(average_precision_score(y, p)), 4),
            "roc_auc": round(float(roc_auc_score(y, p)), 4),
            "brier": round(float(brier_score_loss(y, p)), 4),
            "acc": round(float(accuracy_score(y, (p >= 0.5).astype(int))), 4)}


def plain_lr(Xtr, ytr, Xva):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    m.fit(Xtr, ytr)
    return m.predict_proba(Xva)[:, 1]


def run_pair(Xtr, ytr, Xva, yva, seed):
    """22's run_pair verbatim (returns pr_b, pr_r + full metric dicts)."""
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
    return (metrics(yva, pb), metrics(yva, pr),
            t_bic, t_rs, str(p_bic), str(p_rs))


def load_5050():
    f = pd.read_csv("/home/user/subconscious-search/data/"
                    "diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
    t = [c for c in f.columns if "iab" in c.lower()][0]
    assert f.shape == (70692, 22) and set(f[t].unique()) == {0.0, 1.0}
    for c in f.columns:                      # all whole-valued -> int64 (as in dedup)
        v = f[c].to_numpy(float)
        assert np.all(v == v.astype("int64")), c
        f[c] = v.astype("int64")
    f = f.rename(columns={t: "Diabetes_binary"})
    print(f"50/50: {f.shape} dupes={int(f.duplicated().sum())} "
          f"(friend dedup: 0 by construction -- noted secondary difference)", flush=True)
    return f


def split_5050(f):
    idx = np.arange(len(f))
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=f["Diabetes_binary"],
                               random_state=42)
    va, te = train_test_split(tmp, test_size=0.50,
                              stratify=f["Diabetes_binary"].iloc[tmp], random_state=42)
    return tr, va, te                         # te held out, never indexed


def banked_friend_pairs():
    """{(seed): (bic_row, rs_row)} from 22's banked registry rows (all_21)."""
    out = {}
    with REGISTRY_MIRROR.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("script") != "22_bicameral_hpo_seeds.py":
                continue
            if row.get("feature_set") != "all_21":
                continue
            m = row.get("model", "")
            leg = ("bic" if m.startswith("bicameral-HPO")
                   else ("rs" if m.startswith("randsearch-HPO") else None))
            if leg is None:
                continue
            out.setdefault(int(row["seed"]), {})[leg] = row
    return out


def completed():
    done = set()
    if OUT.exists():
        with OUT.open(newline="") as fh:
            for row in csv.DictReader(fh):
                done.add((row["dataset"], int(row["seed"]), row["method"]))
    return done


def append(row):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    new = not OUT.exists()
    with OUT.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    t_all = time.time()
    done = completed()
    if done:
        print(f"resuming: {len(done)} method-rows already in {OUT.name}, skipping",
              flush=True)

    # ---- friend-full: banked duel pairs + fresh plain-LR anchor ----
    from data_io import load as fload, features as ffeatures
    df, tr, va, te = fload()
    assert te is None
    cols = ffeatures(df)
    Xtr = df.loc[tr, cols].to_numpy(float)
    ytr = df.loc[tr, "Diabetes_binary"].to_numpy(int)
    Xva = df.loc[va, cols].to_numpy(float)
    yva = df.loc[va, "Diabetes_binary"].to_numpy(int)
    print(f"\n[friend-full] n=229474 prev_tr={ytr.mean():.4f} prev_va={yva.mean():.4f} "
          f"acc-trap(majority)={1 - yva.mean():.4f}", flush=True)
    p_lr = plain_lr(Xtr, ytr, Xva)
    ml = metrics(yva, p_lr)
    print(f"  plain-LR anchor: acc={ml['acc']:.4f} pr={ml['pr_auc']:.4f} "
          f"roc={ml['roc_auc']:.4f}", flush=True)
    bank = banked_friend_pairs()
    friend_d = {}
    for s in SEEDS:
        b, r = bank[s]["bic"], bank[s]["rs"]
        friend_d[s] = (float(b["pr_auc"]), float(r["pr_auc"]),
                       float(b["accuracy"]), float(r["accuracy"]))
        print(f"  seed {s} (BANKED): bic pr={b['pr_auc']} acc={b['accuracy']} | "
              f"rs pr={r['pr_auc']} acc={r['accuracy']}", flush=True)

    # ---- 50/50: fresh split + fresh anchor + fresh duel pairs ----
    f = load_5050()
    tr5, va5, te5 = split_5050(f)
    assert len(te5) == 10604, len(te5)     # 15% of 70692; never indexed below
    cols5 = [c for c in f.columns if c != "Diabetes_binary"]
    Xtr5 = f.loc[tr5, cols5].to_numpy(float)
    ytr5 = f.loc[tr5, "Diabetes_binary"].to_numpy(int)
    Xva5 = f.loc[va5, cols5].to_numpy(float)
    yva5 = f.loc[va5, "Diabetes_binary"].to_numpy(int)
    print(f"\n[50/50] n=70692 train/val/test={len(tr5)}/{len(va5)}/{len(te5)} "
          f"prev_tr={ytr5.mean():.4f} prev_va={yva5.mean():.4f} (test sealed)",
          flush=True)
    p_lr5 = plain_lr(Xtr5, ytr5, Xva5)
    ml5 = metrics(yva5, p_lr5)
    print(f"  plain-LR anchor: acc={ml5['acc']:.4f} pr={ml5['pr_auc']:.4f} "
          f"roc={ml5['roc_auc']:.4f}", flush=True)
    bics, rss = {}, {}
    for s in SEEDS:
        if ("5050", s, "bicameral-HPO") in done and ("5050", s, "randsearch-HPO") in done:
            print(f"  seed {s}: RESUMED from CSV", flush=True)
            with OUT.open(newline="") as fh:
                for row in csv.DictReader(fh):
                    if row["dataset"] == "5050" and int(row["seed"]) == s:
                        (bics if row["method"] == "bicameral-HPO" else rss)[s] = row
            continue
        mb, mr, tb, tr_, pb_, pr_ = run_pair(Xtr5, ytr5, Xva5, yva5, s)
        print(f"  seed {s}: bic pr={mb['pr_auc']:.4f} acc={mb['acc']:.4f} ({tb:.0f}s) | "
              f"rs pr={mr['pr_auc']:.4f} acc={mr['acc']:.4f} ({tr_:.0f}s) | "
              f"d_pr={mb['pr_auc'] - mr['pr_auc']:+.4f}", flush=True)
        for method, m_, tt, pp in [("bicameral-HPO", mb, tb, pb_),
                                   ("randsearch-HPO", mr, tr_, pr_)]:
            append({"dataset": "5050", "seed": s, "method": method,
                    "pr_auc": m_["pr_auc"], "roc_auc": m_["roc_auc"],
                    "brier": m_["brier"], "acc": m_["acc"],
                    "time_s": round(tt), "params": pp})
        bics[s], rss[s] = mb, mr

    # ---- comparison ----
    print("\n================ 50/50 vs friend-full (paired seeds 42/101/202) ================",
          flush=True)
    for s in SEEDS:
        fb, fr, fa_b, fa_r = friend_d[s]
        b5, r5 = bics[s], rss[s]
        b5pr, b5ac = float(b5["pr_auc"]), float(b5["acc"])
        r5pr, r5ac = float(r5["pr_auc"]), float(r5["acc"])
        print(f"  seed {s}: friend d_pr={fb - fr:+.4f} d_acc={fa_b - fa_r:+.4f} | "
              f"5050 d_pr={b5pr - r5pr:+.4f} d_acc={b5ac - r5ac:+.4f}",
              flush=True)
    print("==============================================================================",
          flush=True)
    print(f"Done in {(time.time() - t_all) / 60:.1f} min. Friend test sealed; "
          f"50/50 test never indexed.", flush=True)


if __name__ == "__main__":
    main()
