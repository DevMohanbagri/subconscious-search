#!/usr/bin/env python3
"""23 - Stage 5 Target 1: Pareto feature-subset front, Bicameral instead of NSGA-II.

M4 delivers the doc's Target 1 (minimize #features, maximize PR-AUC) with
Bicameral INSTEAD of the prescribed NSGA-II -- plus the prescribed NSGA-II at
a MATCHED total budget as the honest comparator, plus a random-subset front
as the sanity baseline. Reads only dedup.csv/splits.npz; appends to
results/registry.csv (v2); saves front members via fsets.

Design (all three methods share everything except the search rule):
  genotype  21-D (all_21) / 19-D (leakage_free) x in [0,1]^D, bit = x_i > 0.5.
            Empty decodes to the argmax singleton (deterministic, shared rule).
  fitness   LGBM-small on a stratified 50k TRAIN subsample, PR-AUC on FULL val.
            Each method gets a SEPARATE subset cache, so budgets count true
            (uncached) LightGBM fits fairly.
  bicameral per-k scalarized runs: max PR - 0.01*|k - target|, `budget` evals
            per target, 11 targets per pool.
  nsga2     pymoo NSGA-II, 2 objectives (-PR, k), ONE run with total evals =
            11 * budget (exactly Bicameral's per-pool total).
  random    `random_n` uniform random subsets (uniform k, then uniform set).
Front members are re-evaluated PROPERLY (LGBM-medium on full train + plain LR
matching 12_fs_comparators.py's evaluator for overlay), logged, and plotted
(figures/fig_pareto.png). Nominations for 16_main_grid FEATURE_SETS print last.

Protocol: train/val only, test sealed, frozen splits, deterministic seeds.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, "/home/user/subconscious-search")
from bicameral import Bicameral
from ml_benchmark import delong_test

from data_io import load, features, log_result, TARGET
from metrics import summarize
import fsets

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--budget", type=int, default=80,
                help="Bicameral evals per k-target (NSGA-II gets 11 x budget total)")
ap.add_argument("--random-n", type=int, default=400)
ap.add_argument("--pools", default="all_21,leakage_free")
A = ap.parse_args()

TARGETS = {"all_21": [1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 21],
           # leakage_free has 18 features (LEAKAGE_SUSPECT = GenHlth/DiffWalk/PhysHlth)
           "leakage_free": [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18]}
RHO = 0.01            # per-feature penalty in the scalarized objective
SUB_N = 50_000        # stratified train subsample for the cheap fitness
CACHE_FILE = Path("results/pareto_cache.json")


def decode(x):
    """Shared genotype->subset rule. Returns sorted tuple of indices."""
    idx = tuple(sorted(int(i) for i in np.where(np.asarray(x) > 0.5)[0]))
    if not idx:                       # empty -> argmax singleton (deterministic)
        idx = (int(np.argmax(np.asarray(x))),)
    return idx


def mask_of(sub):
    m = 0
    for i in sub:
        m |= 1 << i
    return m


def sub_of(mask, D):
    return tuple(sorted(i for i in range(D) if mask >> i & 1))


class CheapFitness:
    """LGBM-small on 50k train subsample; PR-AUC on full val. Own cache."""

    def __init__(self, cols, Xtr, ytr, Xva, yva, seed):
        rng = np.random.default_rng(1000 + seed)
        i1 = np.where(ytr == 1)[0]
        i0 = np.where(ytr == 0)[0]
        n1 = int(SUB_N * ytr.mean())
        sub = np.concatenate([rng.choice(i1, n1, replace=False),
                              rng.choice(i0, SUB_N - n1, replace=False)])
        rng.shuffle(sub)
        self.Xs, self.ys = Xtr[sub], ytr[sub]
        self.Xva, self.yva = Xva, yva
        self.spw = float((self.ys == 0).sum() / (self.ys == 1).sum())
        self.seed = seed
        self.cache = {}               # mask -> pr_auc
        self.evaled = set()           # every subset truly evaluated (masks)
        self.fits = 0                 # uncached LightGBM fits (= true budget use)

    def __call__(self, sub):
        m = mask_of(sub)
        if m not in self.cache:
            import lightgbm as lgb
            cols = list(sub)
            clf = lgb.LGBMClassifier(
                n_estimators=150, learning_rate=0.08, num_leaves=15,
                min_child_samples=80, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, reg_lambda=1.0,
                scale_pos_weight=self.spw, random_state=self.seed,
                n_jobs=1, verbose=-1)
            clf.fit(self.Xs[:, cols], self.ys)
            p = clf.predict_proba(self.Xva[:, cols])[:, 1]
            self.cache[m] = float(average_precision_score(self.yva, p))
            self.evaled.add(m)
            self.fits += 1
            if self.fits % 250 == 0:
                dump_cache()
        return self.cache[m]


CACHES = {}   # (method, pool) -> CheapFitness, for periodic dumps


def dump_cache():
    try:
        out = {f"{me}|{po}": {str(m): v for m, v in cf.cache.items()}
               for (me, po), cf in CACHES.items()}
        CACHE_FILE.write_text(json.dumps(out))
    except OSError:
        pass


def pareto_front(items):
    """items: list of (mask, k, pr). Nondominated for (max pr, min k)."""
    items = sorted(items, key=lambda t: (t[1], -t[2]))
    front, best = [], -1.0
    for mask, k, pr in items:
        if pr > best + 1e-12:
            front.append((mask, k, pr))
            best = pr
    return front


def hypervolume(front, prevalence, D):
    """HV of a nondominated front, ref point (PR=prevalence, k=D+1)."""
    hv, f = 0.0, sorted(front, key=lambda t: t[1])
    for j, (_, k, pr) in enumerate(f):
        k_next = f[j + 1][1] if j + 1 < len(f) else D + 1
        hv += max(0.0, pr - prevalence) * (k_next - k)
    return hv


def run_bicameral(pool, cols, D, seed, budget):
    print(f"[bicameral|{pool}] per-k runs, budget={budget}/target", flush=True)
    t0 = time.time()
    best = {}                 # target -> (mask, obj)
    for target in TARGETS[pool]:
        cf = CACHES[("bicameral", pool)]

        def obj(x, _t=target):
            s = decode(x)
            return -(cf(s) - RHO * abs(len(s) - _t))

        opt = Bicameral(D, 0.0, 1.0, seed=seed + target)
        xb, fb, _ = opt.optimize(obj, budget, local_search=False)
        s = decode(xb)
        best[target] = (mask_of(s), -fb, cf(s))
        print(f"  target k={target:2d}: got k={len(s):2d} pr_cheap={cf(s):.4f}",
              flush=True)
    print(f"  true fits: {cf.fits}  time: {time.time()-t0:.0f}s", flush=True)
    return best


def run_nsga2(pool, cols, D, seed, total_budget):
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.optimize import minimize
    from pymoo.termination import get_termination

    print(f"[nsga2|{pool}] one run, total evals={total_budget}", flush=True)
    t0 = time.time()
    cf = CACHES[("nsga2", pool)]

    class SubsetProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=D, n_obj=2, n_ieq_constr=0, xl=0.0, xu=1.0)

        def _evaluate(self, x, out):
            s = decode(x)
            out["F"] = [-cf(s), len(s)]

    minimize(SubsetProblem(), NSGA2(pop_size=40),
             termination=get_termination("n_eval", total_budget),
             seed=seed, verbose=False)
    print(f"  true fits: {cf.fits}  time: {time.time()-t0:.0f}s", flush=True)


def run_random(pool, cols, D, seed, n):
    print(f"[random|{pool}] {n} uniform subsets", flush=True)
    t0 = time.time()
    cf = CACHES[("random", pool)]
    rng = np.random.default_rng(seed)
    for _ in range(n):
        k = int(rng.integers(1, D + 1))
        s = tuple(sorted(int(i) for i in rng.choice(D, size=k, replace=False)))
        cf(s)
    print(f"  true fits: {cf.fits}  time: {time.time()-t0:.0f}s", flush=True)


def proper_eval(cols, Xtr, ytr, Xva, yva, seed):
    """LGBM-medium on full train + plain LR (12's evaluator). Returns (p_lgbm, p_lr)."""
    import lightgbm as lgb
    fit, es = train_test_split(np.arange(len(ytr)), test_size=0.10,
                               stratify=ytr, random_state=seed)
    spw = float((ytr == 0).sum() / (ytr == 1).sum())
    m = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.03, num_leaves=31,
                           min_child_samples=100, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, reg_lambda=1.0,
                           scale_pos_weight=spw, n_jobs=2, random_state=seed,
                           verbose=-1)
    m.fit(Xtr[fit], ytr[fit], eval_X=(Xtr[es],), eval_y=(ytr[es],),
          eval_metric="average_precision",
          callbacks=[lgb.early_stopping(50, verbose=False)])
    p_lgbm = m.predict_proba(Xva)[:, 1]
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    lr.fit(Xtr, ytr)
    return p_lgbm, lr.predict_proba(Xva)[:, 1]


def paired_dpr_ci(y, p1, p2, seed=42, n_boot=1000):
    rng = np.random.default_rng(seed)
    n = len(y)
    d = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        d[i] = (average_precision_score(y[idx], p1[idx])
                - average_precision_score(y[idx], p2[idx]))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    t_all = time.time()
    df, tr, va, te = load()
    assert te is None, "test must stay sealed"
    ytr = df.loc[tr, TARGET].to_numpy(int)
    yva = df.loc[va, TARGET].to_numpy(int)
    prev = float(yva.mean())
    print(f"pools={A.pools} seed={A.seed} budget={A.budget} "
          f"train/val={len(tr)}/{len(va)} prevalence={prev:.4f}", flush=True)

    all_rows, overlay_rows = [], []
    store = {}                # (pool, method, k) -> (mask, pr_lgbm, p_lgbm, pr_lr)

    for pool in A.pools.split(","):
        cols = features(df, drop_leakage=(pool == "leakage_free"))
        D = len(cols)
        Xtr = df.loc[tr, cols].to_numpy(np.float32)
        Xva = df.loc[va, cols].to_numpy(np.float32)
        suffix = "" if pool == "all_21" else "_noleak"

        for method in ["bicameral", "nsga2", "random"]:
            CACHES[(method, pool)] = CheapFitness(cols, Xtr, ytr, Xva, yva, A.seed)
        run_bicameral(pool, cols, D, A.seed, A.budget)
        run_nsga2(pool, cols, D, A.seed, len(TARGETS[pool]) * A.budget)
        run_random(pool, cols, D, A.seed, A.random_n)
        dump_cache()

        fronts = {}
        for method in ["bicameral", "nsga2", "random"]:
            cf = CACHES[(method, pool)]
            items = [(m, bin(m).count("1"), cf.cache[m]) for m in cf.evaled]
            fronts[method] = pareto_front(items)
            hv = hypervolume(fronts[method], prev, D)
            print(f"[{pool}|{method}] front size={len(fronts[method])} "
                  f"hv_cheap={hv:.4f} true_fits={cf.fits}", flush=True)

        # ---- proper re-evaluation of the union of fronts ----
        union = {}            # mask -> (method, k, pr_cheap); first method wins ties
        for method in ["bicameral", "nsga2", "random"]:
            for mask, k, pr in fronts[method]:
                union.setdefault(mask, (method, k, pr))
        print(f"[{pool}] proper re-eval of {len(union)} unique front subsets",
              flush=True)
        for mask, (method, k, pr_cheap) in sorted(union.items(), key=lambda t: t[1][1]):
            sub = sub_of(mask, D)
            names = [cols[i] for i in sub]
            p_lgbm, p_lr = proper_eval(Xtr[:, list(sub)], ytr, Xva[:, list(sub)],
                                       yva, A.seed)
            ml, mr = summarize(yva, p_lgbm), summarize(yva, p_lr)
            tag = {"bicameral": "bicpareto", "nsga2": "nsga2pareto",
                   "random": "randpareto"}[method]
            fs_name = f"{tag}_k{k}{suffix}"
            fsets.save(fs_name, names, f"23_pareto_bicameral.py {method}",
                       pool=pool)
            log_result(script="23_pareto_bicameral.py", feature_set=fs_name,
                       n_features=k, model="lgbm", seed=A.seed, split="val",
                       calibrated="no",
                       params={"strategy": "none", "front": method,
                               "pr_cheap": round(pr_cheap, 4)}, **ml,
                       notes=f"M4 Pareto front member ({method}), LGBM-medium; test sealed")
            log_result(script="23_pareto_bicameral.py", feature_set=fs_name,
                       n_features=k, model="lr", seed=A.seed, split="val",
                       calibrated="no",
                       params={"strategy": "none", "front": method,
                               "eval": "12-matched"}, **mr,
                       notes=f"M4 Pareto front member ({method}), LR for fig-12 overlay")
            all_rows.append({"pool": pool, "method": method, "k": k,
                             "features": "|".join(names),
                             "pr_cheap": round(pr_cheap, 4),
                             "pr_lgbm_med": round(ml["pr_auc"], 4),
                             "pr_lr": round(mr["pr_auc"], 4)})
            overlay_rows.append({"pool": pool, "feature_set": fs_name, "k": k,
                                 "method": method, "pr_lr": round(mr["pr_auc"], 4)})
            store[(pool, method, k)] = (mask, ml["pr_auc"], p_lgbm, mr["pr_auc"])
            print(f"  {fs_name:28s} cheap={pr_cheap:.4f} "
                  f"lgbm={ml['pr_auc']:.4f} lr={mr['pr_auc']:.4f}", flush=True)

        # ---- matched-k stats at the paper's k (5, resp. 6 noleak like 16) ----
        k_star = 5 if pool == "all_21" else 6
        if all((pool, m, k_star) in store for m in ["bicameral", "nsga2"]):
            _, pb_pr, pb_p, _ = store[(pool, "bicameral", k_star)]
            _, pn_pr, pn_p, _ = store[(pool, "nsga2", k_star)]
            z, p_d = delong_test(yva, pb_p, pn_p)
            lo, hi = paired_dpr_ci(yva, pb_p, pn_p)
            print(f"[{pool}] matched k={k_star} (LGBM-med P): bic={pb_pr:.4f} "
                  f"nsga2={pn_pr:.4f} DeLong z={z:+.2f} p={p_d:.4f} "
                  f"dPR CI [{lo:+.4f},{hi:+.4f}]", flush=True)

    import pandas as pd
    pd.DataFrame(all_rows).to_csv("results/pareto_front.csv", index=False)
    pd.DataFrame(overlay_rows).to_csv("results/pareto_lr_overlay.csv", index=False)

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pools = A.pools.split(",")
    fig, axes = plt.subplots(1, len(pools), figsize=(6 * len(pools), 4),
                             sharey=True, squeeze=False)
    front_df = pd.DataFrame(all_rows)
    marks = {"bicameral": ("o", "C0"), "nsga2": ("s", "C1"), "random": (".", "0.6")}
    for ax, pool in zip(axes[0], pools):
        for method, (mk, cc) in marks.items():
            g = front_df[(front_df.pool == pool) & (front_df.method == method)]
            g = g.sort_values("k")
            ax.plot(g.k, g.pr_lgbm_med, ls="--", marker=mk, color=cc, label=method)
        ax.axhline(prev, ls=":", color="0.5")
        ax.set_title(pool)
        ax.set_xlabel("Number of features k")
        ax.legend(loc="lower right")
    axes[0][0].set_ylabel("Validation PR-AUC (LGBM-medium, proper)")
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/fig_pareto.png", dpi=300)

    # ---- nominations for the main grid ----
    print("\nNominations for 16_main_grid FEATURE_SETS (add these names):",
          flush=True)
    for pool in pools:
        g = front_df[front_df.pool == pool]
        gb = g[g.method == "bicameral"].sort_values("k")
        knee = gb.loc[(gb.pr_lgbm_med - prev).div(gb.k).idxmax()]
        print(f"  {pool}: knee k={int(knee.k)} "
              f"(features: {knee.features}) pr={knee.pr_lgbm_med:.4f}", flush=True)
    print(f"\nDone in {(time.time()-t_all)/60:.0f} min. "
          f"Test sealed. Registry +{2*len(all_rows)} rows.", flush=True)


if __name__ == "__main__":
    main()
