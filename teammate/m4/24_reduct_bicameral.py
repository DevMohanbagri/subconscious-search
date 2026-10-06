#!/usr/bin/env python3
"""24 - Stage 5 Target 3: Bicameral reduct search vs 09's greedy heldout-reduct.

M4 delivers the doc's Target 3 (find SMALLER reducts than greedy forward
selection) with Bicameral. The comparison is exact: same binned frame
(clinical cuts fit on train), same encoded attributes, same folds (seed 0),
same shrinkage (a=20), same criterion (held-out gamma_H = share of label
uncertainty removed out of sample). Only the SEARCH differs.

  greedy   rst.heldout_reduct inline (a=20, 5 folds, seed 0) -- the same call
           09 makes for its primary reduct. Its path gives gamma_H at every
           prefix k for free: the pointwise matched-k baseline. (M4 does NOT
           save M1's rst_heldout names; the regen exists only inside this run.)
  bicameral per-K scalarized runs: max gamma_H - 0.01*|k-K|, K = 1..kmax,
           plus ONE matched-budget run (budget = greedy's candidate-eval
           count) on max gamma_H - 0.01*k as the fair single cell.
Budgets are reported as a ratio (Bicameral EVALS / greedy EVALS); the claim is
per-k search quality, not compute parity.

Front members get a downstream LR eval (12's exact evaluator, raw features),
fsets saves (bicreduct_k{K}{suffix}), registry v2 rows, and a figure
(figures/fig_reduct.png). Nominations print last.

Protocol: train/val only (binned frame from TRAIN rows; LR eval on val),
test sealed, deterministic seeds.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sys.path.insert(0, "/home/user/subconscious-search")
from bicameral import Bicameral

from data_io import load, features, log_result, TARGET
from discretise import fit_cuts, apply_cuts
from rst import encode, granules, heldout_logloss, heldout_reduct
from metrics import summarize
import fsets

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--budget", type=int, default=40, help="Bicameral evals per K-target")
ap.add_argument("--kmax", type=int, default=8)
ap.add_argument("--pools", default="all_21,leakage_free")
A = ap.parse_args()

RHO = 0.01
SHRINK_A = 20.0
N_FOLDS = 5
GREEDY_SEED = 0


def decode(x):
    """Shared genotype->attribute-subset rule (indices into the pool's attrs)."""
    return tuple(sorted(int(i) for i in np.where(np.asarray(x) > 0.5)[0]))


class GammaH:
    """Held-out gamma_H with a subset cache. Folds/prior match heldout_reduct(seed=0)."""

    def __init__(self, enc, y, attrs):
        self.enc, self.y, self.attrs = enc, y, attrs
        self.n = len(y)
        self.folds = np.random.default_rng(GREEDY_SEED).permutation(self.n) % N_FOLDS
        self.prior = heldout_logloss(np.zeros(self.n, dtype=np.int64), y,
                                     self.folds, SHRINK_A)
        self.cache = {}               # mask -> gamma_H
        self.evals = 0                # uncached criterion evals

    def __call__(self, sub):
        m = 0
        for i in sub:
            m |= 1 << i
        if m not in self.cache:
            B = [self.attrs[i] for i in sub]
            gid = granules(self.enc, B, self.n)
            v = heldout_logloss(gid, self.y, self.folds, SHRINK_A)
            self.cache[m] = float(1 - v / self.prior)
            self.evals += 1
        return self.cache[m]


def lr_prauc(Xtr, ytr, Xva, yva):
    """12_fs_comparators.py's exact evaluator (plain LR, raw features)."""
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    m.fit(Xtr, ytr)
    return m.predict_proba(Xva)[:, 1]


def run_pool(pool, cols_raw, enc, y, Xtr_raw, ytr, Xva_raw, yva, seed, budget, kmax):
    attrs = cols_raw
    D = len(attrs)
    suffix = "" if pool == "all_21" else "_noleak"
    print(f"\n=== {pool} (D={D}) ===", flush=True)

    # ---- greedy baseline: the exact call 09 makes ----
    t0 = time.time()
    R_g, path_g, gh_g = heldout_reduct(enc, y, attrs, a=SHRINK_A,
                                       n_folds=N_FOLDS, seed=GREEDY_SEED)
    k_g = len(R_g)
    greedy_evals = sum(D - j for j in range(k_g + 1))   # k successes + final failed round
    print(f"[greedy] k={k_g} gamma_H={gh_g:.4f} evals={greedy_evals} "
          f"time={time.time()-t0:.1f}s\n  path={path_g}", flush=True)

    gh = GammaH(enc, y, attrs)
    assert abs(gh(tuple(attrs.index(a) for a in R_g)) - gh_g) < 1e-9, \
        "criterion mismatch vs heldout_reduct!"

    # ---- Bicameral per-K front ----
    t0 = time.time()
    front = {}                        # K -> (subset_idx, gamma_H)
    for K in range(1, max(kmax, k_g) + 1):
        def obj(x, _K=K):
            s = decode(x)
            return -(gh(s) - RHO * abs(len(s) - _K))

        opt = Bicameral(D, 0.0, 1.0, seed=seed + K)
        xb, fb, _ = opt.optimize(obj, budget, local_search=False)
        s = decode(xb)
        front[K] = (s, gh(s))
        print(f"  K={K:2d}: got k={len(s):2d} gamma_H={gh(s):.4f}", flush=True)

    # ---- matched-budget single run (the fair cell) ----
    def obj_m(x):
        s = decode(x)
        return -(gh(s) - RHO * len(s))

    opt = Bicameral(D, 0.0, 1.0, seed=seed)
    xb, fb, _ = opt.optimize(obj_m, greedy_evals, local_search=False)
    s_m = decode(xb)
    print(f"[matched-budget] evals={greedy_evals}: k={len(s_m)} "
          f"gamma_H={gh(s_m):.4f} time={time.time()-t0:.0f}s", flush=True)
    print(f"budget ratio Bicameral/greedy: per-K {(max(kmax,k_g)*budget)}/{greedy_evals} = "
          f"{(max(kmax,k_g)*budget)/greedy_evals:.1f}x (+ 1x matched run)", flush=True)

    # ---- pointwise matched-k table: greedy path prefixes vs Bicameral ----
    print(f"\n[matched-k] k | greedy-prefix gh | bicameral gh | winner", flush=True)
    g_prefix_gh = {0: 0.0}
    for j, (_, g) in enumerate(path_g, start=1):
        g_prefix_gh[j] = g
    rows = []
    cand = dict(front)
    if len(s_m) not in cand or gh(s_m) > cand[len(s_m)][1]:
        cand[len(s_m)] = (s_m, gh(s_m))
    for K in sorted(cand):
        s, g = cand[K]
        gp = g_prefix_gh.get(len(s), None)
        tag = ("bicameral" if gp is None or g > gp + 1e-9
               else ("tie" if abs(g - gp) <= 1e-9 else "greedy"))
        print(f"  k={len(s):2d} | {gp if gp is not None else float('nan'):>14} | "
              f"{g:.4f} | {tag}", flush=True)
        rows.append({"pool": pool, "K": K, "k": len(s), "gamma_h": round(g, 4),
                     "greedy_prefix_gh": gp,
                     "features": "|".join(attrs[i] for i in s)})

    # ---- downstream LR eval + saves (Bicameral front only; greedy is M1's) ----
    for r in rows:
        sub = tuple(attrs.index(a) for a in r["features"].split("|")) if r["features"] else ()
        p = lr_prauc(Xtr_raw[:, list(sub)] if sub else np.zeros((len(ytr), 0)),
                     ytr, Xva_raw[:, list(sub)] if sub else np.zeros((len(yva), 0)), yva) \
            if sub else np.full(len(yva), ytr.mean())
        ml = summarize(yva, p)
        r["lr_prauc"] = round(ml["pr_auc"], 4)
        fs_name = f"bicreduct_k{r['k']}{suffix}"
        if r["features"]:
            fsets.save(fs_name, r["features"].split("|"),
                       f"24_reduct_bicameral.py gamma_H={r['gamma_h']}", pool=pool)
            log_result(script="24_reduct_bicameral.py", feature_set=fs_name,
                       n_features=r["k"], model="lr", seed=seed, split="val",
                       calibrated="no",
                       params={"strategy": "none", "eval": "12-matched",
                               "gamma_h": r["gamma_h"]}, **ml,
                       notes="M4 Bicameral reduct, downstream LR; test sealed")

    # greedy full-set LR (context only: computed inline, NOT saved -- M1's name)
    if R_g:
        cols = [attrs.index(a) for a in R_g]
        from sklearn.metrics import average_precision_score
        pg = lr_prauc(Xtr_raw[:, cols], ytr, Xva_raw[:, cols], yva)
        print(f"[greedy downstream] k={k_g} lr_prauc={average_precision_score(yva, pg):.4f} "
              f"(not saved: M1's rst_heldout name)", flush=True)
    return rows, {"k_greedy": k_g, "gh_greedy": round(gh_g, 4),
                  "greedy_evals": greedy_evals, "bic_evals": gh.evals,
                  "greedy_reduct": " ".join(R_g)}


def main():
    t_all = time.time()
    df, tr, va, te = load()
    assert te is None, "test must stay sealed"
    print(f"pools={A.pools} seed={A.seed} budget={A.budget}/K kmax={A.kmax}",
          flush=True)

    # 09's exact setup, once: clinical bins fit on TRAIN rows
    train = df.loc[tr].reset_index(drop=True)
    d = apply_cuts(train, fit_cuts(train, "clinical"))
    y = d[TARGET].to_numpy(float)
    enc = encode(d, features(df))
    ytr = df.loc[tr, TARGET].to_numpy(int)
    yva = df.loc[va, TARGET].to_numpy(int)

    all_rows, summary = [], {}
    for pool in A.pools.split(","):
        cols = features(df, drop_leakage=(pool == "leakage_free"))
        Xtr = df.loc[tr, cols].to_numpy(np.float32)
        Xva = df.loc[va, cols].to_numpy(np.float32)
        rows, summ = run_pool(pool, cols, enc, y, Xtr, ytr, Xva, yva,
                              A.seed, A.budget, A.kmax)
        all_rows += rows
        summary[pool] = summ

    import pandas as pd
    pd.DataFrame(all_rows).to_csv("results/bicameral_reduct.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pools = A.pools.split(",")
    fig, axes = plt.subplots(1, len(pools), figsize=(6 * len(pools), 4),
                             sharey=True, squeeze=False)
    rdf = pd.DataFrame(all_rows)
    for ax, pool in zip(axes[0], pools):
        g = rdf[rdf.pool == pool].sort_values("k")
        ax.plot(g.k, g.gamma_h, "o-", label="bicameral front")
        gp = g.dropna(subset=["greedy_prefix_gh"]).sort_values("k")
        ax.plot(gp.k, gp.greedy_prefix_gh, "s--", label="greedy path")
        ax.set_title(f"{pool} (greedy: k={summary[pool]['k_greedy']}, "
                     f"gh={summary[pool]['gh_greedy']})")
        ax.set_xlabel("reduct size k")
        ax.legend(loc="lower right")
    axes[0][0].set_ylabel("Held-out gamma_H (share of uncertainty removed)")
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/fig_reduct.png", dpi=300)

    print("\nNominations (smallest Bicameral k at >= greedy gamma_H):", flush=True)
    for pool in pools:
        g = rdf[rdf.pool == pool]
        gh_g = summary[pool]["gh_greedy"]
        beat = g[g.gamma_h >= gh_g - 1e-9].sort_values("k")
        if len(beat):
            b = beat.iloc[0]
            print(f"  {pool}: bicreduct_k{int(b.k)} gh={b.gamma_h:.4f} "
                  f"(greedy k={summary[pool]['k_greedy']} gh={gh_g:.4f})", flush=True)
        else:
            print(f"  {pool}: none beat greedy gh={gh_g:.4f}", flush=True)
    print(f"\nDone in {(time.time()-t_all)/60:.1f} min. Test sealed.", flush=True)


if __name__ == "__main__":
    main()
