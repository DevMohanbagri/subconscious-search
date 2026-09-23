#!/usr/bin/env python3
"""ESDRP wrapper experiment: v3 vs the paper's swarm optimizers.

Replicates Sarker et al. (Sci Rep 2026) wrapper setup on the Early Stage
Diabetes Risk Prediction dataset (UCI-529, 520x16+class):
  - 20-D mixed-integer space: 4 RF hyperparams (Table 4) + 16-bit mask
  - fitness = F1 on the 70% train fold (their choice, warts included)
  - sliding-window 70/30 x 10 folds (their Eq. 22-23)
  - majority-vote final model (their Eq. 24-25)
Methods: SMO-Pop v3, FOX, HBA, TSO (mealpy), + random search (missing
from the paper). Every run consumes EXACTLY `evals` true RF trainings
(asserted) with seeds paired per fold across methods.

Deviations from the paper (all disclosed, all toward rigor):
  - fixed shuffle seed (42) and per-fold seeds (seed0+fold); paper: none
  - RF random_state fixed per fold (deterministic fitness); paper: default
  - budget 1000/fold (Arm 1) instead of 5000: their own plots show train
    F1=1.0 by ~150-350 evals; evals-to-perfect is logged to verify every
    run saturates; Arm 2 spot-checks v3 vs TSO at full 5000 on fold 1
  - PLUS clean per-fold numbers (fold-best on its own untouched test),
    which the paper's voted-model evaluation leaks (voted features saw
    every sample via other folds' train sets).

Data: data/esdrp.csv (UCI-529; gitignored per repo convention).
  Source: https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction+dataset
  Fetched via GitHub mirror (UCI blocked in sandbox), verified against
  paper Table 2 (Obesity No=432, Polyuria No=262, 320+/200-).
"""

import argparse
import json
import os
import sys
import time

import numpy as np

N_EST = (100, 300)
DEPTH = (3, 10)
SPLIT = (2, 20)
LEAF = (1, 20)
N_FEATS = 16
DIM = 20


def load_esdrp(path="data/esdrp.csv"):
    import pandas as pd
    d = pd.read_csv(path)
    assert d.shape == (520, 17), d.shape
    X = np.zeros((520, N_FEATS))
    X[:, 0] = d["Age"].values
    cols = list(d.columns[1:16])
    for j, c in enumerate(cols, start=1):
        v = set(d[c].unique())
        if v == {"Yes", "No"}:
            X[:, j] = (d[c].values == "Yes").astype(int)
        elif v == {"Male", "Female"}:
            X[:, j] = (d[c].values == "Male").astype(int)
        else:
            raise ValueError(f"col {c}: {v}")
    y = (d["class"].values == "Positive").astype(int)
    assert y.sum() == 320 and (1 - y).sum() == 200
    return X, y, ["Age"] + cols


def make_folds(n=520, seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    folds = []
    for i in range(10):
        start = (i * n) // 10
        test = np.array([idx[(start + k) % n] for k in range((3 * n) // 10)])
        train = np.array([v for v in idx if v not in set(test)])
        folds.append((train, test))
    return folds


def decode(x):
    # NOTE: Arm 1/Arm 2 ran WITHOUT the min() clips (exact-1.0 bound hits
    # decoded to 301/11). Uniform across methods; effect negligible
    # (<=1 extra tree / 1 extra depth level); disclosed in README.
    x = np.asarray(x, dtype=float)
    n_est = min(int(np.floor(x[0] * (N_EST[1] - N_EST[0] + 1))) + N_EST[0],
                N_EST[1])
    depth = min(int(np.floor(x[1] * (DEPTH[1] - DEPTH[0] + 1))) + DEPTH[0],
                DEPTH[1])
    split = min(int(np.floor(x[2] * (SPLIT[1] - SPLIT[0] + 1))) + SPLIT[0],
                SPLIT[1])
    leaf = min(int(np.floor(x[3] * (LEAF[1] - LEAF[0] + 1))) + LEAF[0],
               LEAF[1])
    mask = tuple(int(v > 0.5) for v in x[4:20])
    return n_est, depth, split, leaf, mask


class Fitness:
    """Train-F1 fitness with exact-budget cap + best-config tracking."""

    def __init__(self, Xtr, ytr, rf_seed, max_evals):
        self.Xtr, self.ytr = Xtr, ytr
        self.rf_seed = rf_seed
        self.max = max_evals
        self.n = 0
        self.best_f1 = -1.0
        self.best_x = None
        self.evals_to_perfect = None

    def __call__(self, x):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import f1_score
        if self.n >= self.max:
            return 1.0 - self.best_f1  # frozen: no new signal (loss form)
        n_est, depth, split, leaf, mask = decode(x)
        if sum(mask) == 0:
            f1 = 0.0
        else:
            cols = [j for j, m in enumerate(mask) if m]
            clf = RandomForestClassifier(
                n_estimators=n_est, max_depth=depth,
                min_samples_split=split, min_samples_leaf=leaf,
                random_state=self.rf_seed, n_jobs=1)
            f1 = float(f1_score(self.ytr,
                                clf.fit(self.Xtr[:, cols], self.ytr)
                                   .predict(self.Xtr[:, cols])))
        self.n += 1
        if f1 > self.best_f1 + 1e-12:
            self.best_f1 = f1
            self.best_x = np.asarray(x, dtype=float).copy()
            if f1 >= 1.0 - 1e-12 and self.evals_to_perfect is None:
                self.evals_to_perfect = self.n
        return 1.0 - f1  # optimizers minimize


def run_one(method, Xtr, ytr, Xte, yte, seed, evals):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, precision_score, \
        recall_score
    fit = Fitness(Xtr, ytr, rf_seed=seed, max_evals=evals)
    t0 = time.time()
    if method == "smo":
        from compare_baselines import run_smo_pop
        run_smo_pop(fit, DIM, 0.0, 1.0, evals, seed)
    elif method == "rs":
        rng = np.random.RandomState(seed)
        for _ in range(evals):
            fit(rng.rand(DIM))
    else:
        from mealpy.utils.space import FloatVar
        cls = {"fox": "FOX.OriginalFOX", "hba": "HBA.OriginalHBA",
               "tso": "TSO.OriginalTSO"}[method]
        mod, name = cls.split(".")
        Cls = getattr(__import__(f"mealpy.swarm_based.{mod}",
                                 fromlist=[name]), name)
        pop = 50
        model = Cls(epoch=max(1, round(evals / pop) - 1), pop_size=pop)
        model.solve({"obj_func": fit,
                     "bounds": FloatVar(lb=[0.0] * DIM, ub=[1.0] * DIM),
                     "minmax": "min"}, seed=seed)
    wall = time.time() - t0
    # v3 is generational: it stops at the last full generation instead of
    # topping up, so it may use a few evals UNDER budget (conservative,
    # reported as evals_used). It must never exceed. Others are exact.
    assert fit.n <= evals, f"budget leak: {fit.n} > {evals}"
    if method != "smo":
        assert fit.n == evals, f"budget leak: {fit.n} != {evals}"
    n_est, depth, split, leaf, mask = decode(fit.best_x)
    cols = [j for j, m in enumerate(mask) if m]
    clf = RandomForestClassifier(
        n_estimators=n_est, max_depth=depth, min_samples_split=split,
        min_samples_leaf=leaf, random_state=seed, n_jobs=1)
    clf.fit(Xtr[:, cols], ytr)
    pred = clf.predict(Xte[:, cols])
    return {
        "method": method, "seed": seed, "evals": evals,
        "evals_used": fit.n,
        "best_train_f1": fit.best_f1,
        "evals_to_perfect": fit.evals_to_perfect,
        "n_est": n_est, "depth": depth, "split": split, "leaf": leaf,
        "mask": "".join(str(m) for m in mask), "n_feats": sum(mask),
        "clean_acc": float(accuracy_score(yte, pred)),
        "clean_f1": float(f1_score(yte, pred)),
        "clean_prec": float(precision_score(yte, pred)),
        "clean_rec": float(recall_score(yte, pred)),
        "wall_s": round(wall, 1),
    }


def run_fold(args):
    method, fold, seed, evals = args
    X, y, _ = load_esdrp()
    folds = make_folds()
    tri, tei = folds[fold - 1]
    rec = run_one(method, X[tri], y[tri], X[tei], y[tei], seed, evals)
    rec["fold"] = fold
    print(f"fold={fold} {method}: trainF1={rec['best_train_f1']:.4f} "
          f"perfect@{rec['evals_to_perfect']} cleanAcc={rec['clean_acc']:.4f} "
          f"feats={rec['n_feats']} ({rec['wall_s']}s)", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["smo", "fox", "hba", "tso", "rs"])
    ap.add_argument("--folds", default="1-10")
    ap.add_argument("--evals", type=int, default=1000)
    ap.add_argument("--seed0", type=int, default=12345)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lo, hi = a.folds.split("-")
    folds = list(range(int(lo), int(hi) + 1))
    tasks = [(a.method, f, a.seed0 + f, a.evals) for f in folds]
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        recs = list(ex.map(run_fold, tasks))
    recs.sort(key=lambda r: r["fold"])
    with open(a.out, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    print(f"Done in {time.time()-t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
