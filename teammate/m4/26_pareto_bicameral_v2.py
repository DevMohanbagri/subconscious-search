#!/usr/bin/env python3
"""26 - Stage 5 Target 1 FOLLOW-UP: Bicameral-v2 Pareto front (one-sided penalty).

POST-HOC diagnostic, labelled honestly: v1 (23) lost to NSGA-II because its
two-sided penalty PR - 0.01*|k-K| overshot small-k targets (asked k=1, got
k=9). V2 tests the mechanism with a one-sided penalty:

    fitness = PR_cheap - 0.1 * max(0, k - K)

Subsets at k <= K compete on pure PR (no distortion among feasible sets);
overshoot pays 0.1 PR per extra feature. Same pools, targets, budgets
(80/target), seeds (seed+K) and cheap fitness as v1; NSGA-II/random fronts
STAND from 23 (banked, not recomputed).

Efficiency without information leak: v2 keeps its OWN cache namespace but
seeds lookups from v1's banked cache (same subset -> same PR, bit-identical
to a fresh fit, so trajectories are exactly what a fresh run would produce;
only wall time differs). New-vs-reused fit counts are reported.
Proper re-eval: members already evaluated in 23 (same subset) REUSE their
LGBM-med/LR numbers (deterministic eval -> identical); genuinely new subsets
are fit fresh, saved as bicpareto2_k{k}{suffix}, and logged. Matched-k
DeLong stats refit P-vectors (2 cheap fits per pool).

Outputs: results/pareto_v2_front.csv, figures/fig_pareto_v2.png (v1 faded +
v2 bold), registry rows for NEW subsets only. 23's files are never touched.

Protocol: train/val only, test sealed, deterministic.
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
ap.add_argument("--budget", type=int, default=80)
ap.add_argument("--pools", default="all_21,leakage_free")
A = ap.parse_args()

TARGETS = {"all_21": [1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 21],
           "leakage_free": [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18]}
RHO_V2 = 0.1              # one-sided: only k > K pays
SUB_N = 50_000
CACHE_FILE = Path("results/pareto_cache.json")
V1_CSV = Path("results/pareto_front.csv")
TAG = "bicameral_v2"
CACHES = {}


def decode(x):
    idx = tuple(sorted(int(i) for i in np.where(np.asarray(x) > 0.5)[0]))
    if not idx:
        idx = (int(np.argmax(np.asarray(x))),)
    return idx


def mask_of(sub):
    m = 0
    for i in sub:
        m |= 1 << i
    return m


def sub_of(mask, D):
    return tuple(sorted(i for i in range(D) if mask >> i & 1))


class CheapFitnessV2:
    """V1-identical cheap fitness; own cache + evaled; read-only v1 lookup."""

    def __init__(self, cols, Xtr, ytr, Xva, yva, seed, v1lookup):
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
        self.v1 = v1lookup
        self.cache = {}
        self.evaled = set()
        self.fits = 0
        self.reused = 0

    def __call__(self, sub):
        m = mask_of(sub)
        if m not in self.cache:
            if m in self.v1:
                self.cache[m] = self.v1[m]
                self.reused += 1
            else:
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
                self.fits += 1
            self.evaled.add(m)
        return self.cache[m]


def load_v1_lookup(pool):
    """{mask: pr} from v1's banked cache (all three v1 methods: same fitness)."""
    out = json.loads(CACHE_FILE.read_text())
    look = {}
    for me in ["bicameral", "nsga2", "random"]:
        d = out.get(f"{me}|{pool}", {})
        inner = d.get("cache", d)
        for m, v in inner.items():
            look.setdefault(int(m), float(v))
    return look


def dump_cache():
    try:
        out = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
        for (me, po), cf in CACHES.items():
            out[f"{me}|{po}"] = {"cache": {str(m): v for m, v in cf.cache.items()},
                                 "evaled": sorted(cf.evaled)}
        CACHE_FILE.write_text(json.dumps(out))
    except OSError:
        pass


def pareto_front(items):
    items = sorted(items, key=lambda t: (t[1], -t[2]))
    front, best = [], -1.0
    for mask, k, pr in items:
        if pr > best + 1e-12:
            front.append((mask, k, pr))
            best = pr
    return front


def hypervolume(front, prevalence, D):
    hv, f = 0.0, sorted(front, key=lambda t: t[1])
    for j, (_, k, pr) in enumerate(f):
        k_next = f[j + 1][1] if j + 1 < len(f) else D + 1
        hv += max(0.0, pr - prevalence) * (k_next - k)
    return hv


def proper_eval(Xtr, ytr, Xva, yva, seed):
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
    assert V1_CSV.exists(), "run 23 first (v1 front + cache required)"
    import pandas as pd
    v1 = pd.read_csv(V1_CSV)
    v1proper = {(r.pool, r.features): (r.pr_lgbm_med, r.pr_lr)
                for r in v1.itertuples()}
    v1sub = {(r.pool, r.method, r.k): r.features for r in v1.itertuples()}

    df, tr, va, te = load()
    assert te is None, "test must stay sealed"
    ytr = df.loc[tr, TARGET].to_numpy(int)
    yva = df.loc[va, TARGET].to_numpy(int)
    prev = float(yva.mean())
    print(f"v2 one-sided rho={RHO_V2} pools={A.pools} seed={A.seed} budget={A.budget}",
          flush=True)

    all_rows = []
    for pool in A.pools.split(","):
        cols = features(df, drop_leakage=(pool == "leakage_free"))
        D = len(cols)
        Xtr = df.loc[tr, cols].to_numpy(np.float32)
        Xva = df.loc[va, cols].to_numpy(np.float32)
        suffix = "" if pool == "all_21" else "_noleak"
        cf = CheapFitnessV2(cols, Xtr, ytr, Xva, yva, A.seed, load_v1_lookup(pool))
        CACHES[(TAG, pool)] = cf

        t0 = time.time()
        for K in TARGETS[pool]:
            def obj(x, _K=K):
                s = decode(x)
                return -(cf(s) - RHO_V2 * max(0, len(s) - _K))

            opt = Bicameral(D, 0.0, 1.0, seed=A.seed + K)
            xb, fb, _ = opt.optimize(obj, A.budget, local_search=False)
            s = decode(xb)
            print(f"[{pool}] target k={K:2d}: got k={len(s):2d} "
                  f"pr_cheap={cf(s):.4f}", flush=True)
        dump_cache()
        print(f"[{pool}] new fits={cf.fits} reused={cf.reused} "
              f"time={time.time()-t0:.0f}s", flush=True)

        items = [(m, bin(m).count("1"), cf.cache[m]) for m in cf.evaled]
        front = pareto_front(items)
        print(f"[{pool}|v2] front size={len(front)} "
              f"hv_cheap={hypervolume(front, prev, D):.4f}", flush=True)

        store = {}
        for mask, k, pr_cheap in sorted(front, key=lambda t: t[1]):
            sub = sub_of(mask, D)
            names = [cols[i] for i in sub]
            key = "|".join(names)
            if (pool, key) in v1proper:
                pr_lgbm, pr_lr = v1proper[(pool, key)]
                src = "reused-v1"
            else:
                p_lgbm, p_lr = proper_eval(Xtr[:, list(sub)], ytr,
                                           Xva[:, list(sub)], yva, A.seed)
                ml, mr = summarize(yva, p_lgbm), summarize(yva, p_lr)
                pr_lgbm, pr_lr = round(ml["pr_auc"], 4), round(mr["pr_auc"], 4)
                src = "new"
                fs_name = f"bicpareto2_k{k}{suffix}"
                fsets.save(fs_name, names, f"26_pareto_bicameral_v2.py one-sided",
                           pool=pool)
                log_result(script="26_pareto_bicameral_v2.py", feature_set=fs_name,
                           n_features=k, model="lgbm", seed=A.seed, split="val",
                           calibrated="no",
                           params={"strategy": "none", "front": "bicameral_v2",
                                   "pr_cheap": round(pr_cheap, 4)}, **ml,
                           notes="M4 Pareto v2 (post-hoc follow-up); test sealed")
                log_result(script="26_pareto_bicameral_v2.py", feature_set=fs_name,
                           n_features=k, model="lr", seed=A.seed, split="val",
                           calibrated="no",
                           params={"strategy": "none", "front": "bicameral_v2",
                                   "eval": "12-matched"}, **mr,
                           notes="M4 Pareto v2 (post-hoc follow-up), LR overlay")
            all_rows.append({"pool": pool, "k": k, "features": key,
                             "pr_cheap": round(pr_cheap, 4), "pr_lgbm_med": pr_lgbm,
                             "pr_lr": pr_lr, "source": src})
            store[k] = (names, pr_lgbm)
            print(f"  bicpareto2_k{k}{suffix:8s} cheap={pr_cheap:.4f} "
                  f"lgbm={pr_lgbm:.4f} lr={pr_lr:.4f} [{src}]", flush=True)

        k_star = 5 if pool == "all_21" else 6
        if k_star in store and (pool, "nsga2", k_star) in v1sub:
            names_v2, _ = store[k_star]
            names_ns = v1sub[(pool, "nsga2", k_star)].split("|")
            c_v2 = [cols.index(a) for a in names_v2]
            c_ns = [cols.index(a) for a in names_ns]
            p_v2, _ = proper_eval(Xtr[:, c_v2], ytr, Xva[:, c_v2], yva, A.seed)
            p_ns, _ = proper_eval(Xtr[:, c_ns], ytr, Xva[:, c_ns], yva, A.seed)
            z, p_d = delong_test(yva, p_v2, p_ns)
            lo, hi = paired_dpr_ci(yva, p_v2, p_ns)
            print(f"[{pool}] matched k={k_star}: v2={average_precision_score(yva, p_v2):.4f} "
                  f"nsga2={average_precision_score(yva, p_ns):.4f} DeLong z={z:+.2f} "
                  f"p={p_d:.4f} dPR CI [{lo:+.4f},{hi:+.4f}]", flush=True)

    pd.DataFrame(all_rows).to_csv("results/pareto_v2_front.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pools = A.pools.split(",")
    fig, axes = plt.subplots(1, len(pools), figsize=(6 * len(pools), 4),
                             sharey=True, squeeze=False)
    v2df = pd.DataFrame(all_rows)
    marks = {"bicameral": (".", "0.7"), "nsga2": (".", "0.7"), "random": (".", "0.85")}
    for ax, pool in zip(axes[0], pools):
        for method, (mk, cc) in marks.items():
            g = v1[(v1.pool == pool) & (v1.method == method)].sort_values("k")
            ax.plot(g.k, g.pr_lgbm_med, ls=":", marker=mk, color=cc, label=f"v1 {method}")
        g2 = v2df[v2df.pool == pool].sort_values("k")
        ax.plot(g2.k, g2.pr_lgbm_med, ls="-", marker="o", color="C3",
                label="bicameral v2")
        ax.axhline(prev, ls=":", color="0.5")
        ax.set_title(pool)
        ax.set_xlabel("Number of features k")
        ax.legend(loc="lower right", fontsize=8)
    axes[0][0].set_ylabel("Validation PR-AUC (LGBM-medium, proper)")
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/fig_pareto_v2.png", dpi=300)
    print(f"\nDone in {(time.time()-t_all)/60:.1f} min. Test sealed.", flush=True)


if __name__ == "__main__":
    main()
