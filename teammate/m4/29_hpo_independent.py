#!/usr/bin/env python3
"""29 - HPO duel REDO with independent random streams (both datasets).

Replaces 22/28's headline. Those duels shared RNG streams: Bicameral(seed)
uses default_rng(seed) and its 20-member initial population IS random
search's first 20 draws (proven: RS draw #12 = the reported seed-202 best
for BOTH methods). Shared evals inflate P(exact tie). The PR-AUC values were
genuine, but independence was violated -- this rerun fixes it.

Design: Bicameral keeps the SAME seeds (42/101/...) so its legs must
REPRODUCE 22/28's Bicameral values bit-exactly (determinism proof, checked
per seed); ONLY the RS leg moves to seed+1000 (independent stream).
  friend-full  frozen splits (test sealed), all_21 x {42,101,202,303,404} +
               leakage_free x 42; rows to results/registry.csv, script=29.
  50/50        28's identical split (asserted sizes); x {42,101,202}; rows to
               teammate/m4_results/analysis/dataset_compare_ind.csv.
Same budgets (108), objective (val log-loss), fit-train-only as 22.
Order: 50/50 first (~15 min: validates the script end-to-end), then friend.
Resume: friend pairs from live registry (script=29, both legs); 50/50 pairs
from the ind-CSV. Verdicts per dataset with the pre-registered rule.
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, roc_auc_score)
from sklearn.model_selection import train_test_split

sys.path.insert(0, "/home/user/subconscious-search")
from bicameral import Bicameral
from ml_benchmark import hgb_from_x

from data_io import load, features, log_result, TARGET
from metrics import summarize
import data_io

BUDGET = 108
SEEDS_FULL = [42, 101, 202, 303, 404]
SEEDS_5050 = [42, 101, 202]
RS_SHIFT = 1000
SCRIPT = "29_hpo_independent.py"
IND_CSV = Path("/home/user/subconscious-search/teammate/m4_results/analysis/"
               "dataset_compare_ind.csv")
CSV28 = Path("/home/user/subconscious-search/teammate/m4_results/analysis/"
             "dataset_compare.csv")


def metrics4(y, p):
    return {"pr_auc": round(float(average_precision_score(y, p)), 4),
            "roc_auc": round(float(roc_auc_score(y, p)), 4),
            "brier": round(float(brier_score_loss(y, p)), 4),
            "acc": round(float(accuracy_score(y, (p >= 0.5).astype(int))), 4)}


def run_duel(Xtr, ytr, Xva, yva, seed):
    """One Bicameral(seed) vs RS(seed+1000) pair. Returns (bic, rs) dicts."""

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
    rng = np.random.default_rng(seed + RS_SHIFT)
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
    return ({"m": metrics4(yva, pb), "t": t_bic, "p": str(p_bic)},
            {"m": metrics4(yva, pr), "t": t_rs, "p": str(p_rs)})


def completed_friend():
    done = {}
    reg = Path("results/registry.csv")
    with reg.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("script") != SCRIPT:
                continue
            m = row.get("model", "")
            if "(hgb, 108 evals)" not in m:
                continue
            leg = ("bic" if m.startswith("bicameral-HPO")
                   else ("rs" if m.startswith("randsearch-HPO") else None))
            if leg is None:
                continue
            try:
                done.setdefault((row["feature_set"], int(row["seed"])), {})[leg] = row
            except (KeyError, ValueError):
                continue
    return {k: v for k, v in done.items() if set(v) == {"bic", "rs"}}


def completed_ind():
    done = {}
    if IND_CSV.exists():
        with IND_CSV.open(newline="") as fh:
            for row in csv.DictReader(fh):
                done.setdefault(int(row["seed"]), {})[row["method"]] = row
    return {k: v for k, v in done.items()
            if set(v) == {"bicameral-HPO", "randsearch-HPO"}}


def append_ind(seed, method, m, t, p):
    IND_CSV.parent.mkdir(parents=True, exist_ok=True)
    new = not IND_CSV.exists()
    with IND_CSV.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "seed", "method", "pr_auc",
                                           "roc_auc", "brier", "acc", "time_s",
                                           "params", "rs_stream"])
        if new:
            w.writeheader()
        w.writerow({"dataset": "5050", "seed": seed, "method": method,
                    "pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"],
                    "brier": m["brier"], "acc": m["acc"], "time_s": round(t),
                    "params": p, "rs_stream": "independent+1000"})


def verdict(name, bics, rss):
    from scipy import stats
    bics = np.array(bics)
    rss = np.array(rss)
    d = bics - rss
    t, p = stats.ttest_ind(bics, rss, equal_var=False)
    v = ("bicameral" if (p < 0.05 and d.mean() > 0.002)
         else ("randsearch" if (p < 0.05 and d.mean() < -0.002) else "TIE"))
    print(f"[{name}] bic {bics.mean():.4f}+-{bics.std(ddof=1):.4f} | "
          f"rs {rss.mean():.4f}+-{rss.std(ddof=1):.4f} | mean_d={d.mean():+.4f} "
          f"Welch t={t:+.2f} p={p:.4f} -> {v}", flush=True)
    return v


def expected_bic():
    """Banked Bicameral PR-AUCs that 29's Bicameral legs must reproduce."""
    exp = {}
    with Path("results/registry.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            if (row.get("script") == "22_bicameral_hpo_seeds.py"
                    and row.get("model", "").startswith("bicameral-HPO")):
                exp[("full", row["feature_set"], int(row["seed"]))] = float(row["pr_auc"])
    with CSV28.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row["method"] == "bicameral-HPO":
                exp[("5050", "all_21", int(row["seed"]))] = float(row["pr_auc"])
    return exp


def main():
    t_all = time.time()
    exp = expected_bic()
    n_repro = n_tot = 0

    def check(which, tag, seed, pr):
        nonlocal n_repro, n_tot
        n_tot += 1
        want = exp[(which, tag, seed)]
        ok = abs(pr - want) < 1e-9
        n_repro += ok
        print(f"    Bicameral determinism: got {pr:.4f} want {want:.4f} "
              f"{'REPRODUCED' if ok else 'MISMATCH!'}", flush=True)

    # ================= 50/50 first (fast validation) =================
    print("================ 50/50 (independent RS) ================", flush=True)
    f = pd.read_csv("/home/user/subconscious-search/data/"
                    "diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
    f = f.rename(columns={c: "Diabetes_binary" for c in f.columns if "iab" in c.lower()})
    for c in f.columns:
        f[c] = f[c].to_numpy(float).astype("int64")
    idx = np.arange(len(f))
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=f["Diabetes_binary"],
                               random_state=42)
    va, te = train_test_split(tmp, test_size=0.50,
                              stratify=f["Diabetes_binary"].iloc[tmp], random_state=42)
    assert (len(tr), len(va), len(te)) == (49484, 10604, 10604)
    assert abs(f.loc[va, "Diabetes_binary"].mean() - 0.5) < 1e-9
    cols5 = [c for c in f.columns if c != "Diabetes_binary"]
    Xtr5 = f.loc[tr, cols5].to_numpy(float)
    ytr5 = f.loc[tr, "Diabetes_binary"].to_numpy(int)
    Xva5 = f.loc[va, cols5].to_numpy(float)
    yva5 = f.loc[va, "Diabetes_binary"].to_numpy(int)
    b5, r5 = [], []
    done_ind = completed_ind()
    for s in SEEDS_5050:
        if s in done_ind:
            b = float(done_ind[s]["bicameral-HPO"]["pr_auc"])
            r = float(done_ind[s]["randsearch-HPO"]["pr_auc"])
            print(f"  seed {s}: RESUMED bic={b:.4f} rs={r:.4f}", flush=True)
        else:
            bic, rs = run_duel(Xtr5, ytr5, Xva5, yva5, s)
            b, r = bic["m"]["pr_auc"], rs["m"]["pr_auc"]
            print(f"  seed {s}: bic pr={b:.4f} acc={bic['m']['acc']:.4f} ({bic['t']:.0f}s) | "
                  f"rs-ind pr={r:.4f} acc={rs['m']['acc']:.4f} ({rs['t']:.0f}s) | "
                  f"d={b - r:+.4f}", flush=True)
            check("5050", "all_21", s, b)
            append_ind(s, "bicameral-HPO", bic["m"], bic["t"], bic["p"])
            append_ind(s, "randsearch-HPO", rs["m"], rs["t"], rs["p"])
        b5.append(b)
        r5.append(r)
    verdict("5050 (n=3)", b5, r5)

    # ================= friend-full =================
    print("\n================ friend-full (independent RS) ================", flush=True)
    df, trf, vaf, tef = load()
    assert tef is None
    bF, rF = [], []
    done_fr = completed_friend()
    for drop, tag in [(False, "all_21"), (True, "leakage_free")]:
        seeds = SEEDS_FULL if tag == "all_21" else [42]
        cols = features(df, drop_leakage=drop)
        Xtr = df.loc[trf, cols].to_numpy(float)
        ytr = df.loc[trf, TARGET].to_numpy(int)
        Xva = df.loc[vaf, cols].to_numpy(float)
        yva = df.loc[vaf, TARGET].to_numpy(int)
        print(f"--- {tag} ---", flush=True)
        for s in seeds:
            if (tag, s) in done_fr:
                b = float(done_fr[(tag, s)]["bic"]["pr_auc"])
                r = float(done_fr[(tag, s)]["rs"]["pr_auc"])
                print(f"  seed {s}: RESUMED bic={b:.4f} rs={r:.4f}", flush=True)
            else:
                bic, rs = run_duel(Xtr, ytr, Xva, yva, s)
                b, r = bic["m"]["pr_auc"], rs["m"]["pr_auc"]
                print(f"  seed {s}: bic pr={b:.4f} ({bic['t']:.0f}s) | "
                      f"rs-ind pr={r:.4f} ({rs['t']:.0f}s) | d={b - r:+.4f}",
                      flush=True)
                check("full", tag, s, b)
                # registry needs summarize()'s v2 fields: refit best params
                # (deterministic -> identical P) to recover prediction vectors
                p_bic = predict_from_params(Xtr, ytr, Xva, bic["p"])
                p_rs = predict_from_params(Xtr, ytr, Xva, rs["p"])
                for leg, dd, pp in [("bicameral-HPO", bic, p_bic),
                                    ("randsearch-HPO", rs, p_rs)]:
                    row = summarize(yva, pp)
                    log_result(script=SCRIPT, feature_set=tag, n_features=len(cols),
                               model=f"{leg} (hgb, 108 evals)", params=dd["p"],
                               seed=s, split="val", **row,
                               notes=f"29 independent-RS rerun (+1000); "
                                     f"fit-train-only; time={dd['t']:.0f}s; test sealed")
            if tag == "all_21":
                bF.append(b)
                rF.append(r)
    verdict("friend-full (n=5)", bF, rF)
    print(f"\nBicameral determinism: {n_repro}/{n_tot} reproduced bit-exactly",
          flush=True)
    print(f"Done in {(time.time() - t_all) / 60:.1f} min. Test sealed.", flush=True)


def predict_from_params(Xtr, ytr, Xva, params_str):
    """Refit recorded best HGB params (deterministic) to recover P for summarize()."""
    import ast
    import ml_benchmark
    from sklearn.ensemble import HistGradientBoostingClassifier
    d = ast.literal_eval(params_str)
    m = HistGradientBoostingClassifier(
        learning_rate=d["lr"], max_depth=d["depth"],
        min_samples_leaf=d["leaf"], l2_regularization=d["l2"],
        max_iter=200, random_state=ml_benchmark.SEED)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xva)[:, 1]


if __name__ == "__main__":
    main()
