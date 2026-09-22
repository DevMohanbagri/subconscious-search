#!/usr/bin/env python3
"""
Tougher benchmarks: CEC2017 (opfunu) — shifted/rotated/hybrid/composition.

opfunu implements F1, F3..F29 (F2 officially excluded from CEC2017,
F30 not implemented in opfunu) -> 28 functions, bounds [-100, 100]^D.

Usage:
  python cec_benchmark.py --dim 10 --max-evals 10000 --n-runs 5 --seed0 1 \\
      --methods SMO-Pop,CMA-ES,DiffEvol,DualAnneal,RandSearch --out cec_dim10.jsonl
  python cec_benchmark.py --summarize cec_dim10.jsonl   # tables only

Results append as JSON lines; completed (func, method, seed) combos are
skipped, so runs can be split across calls and resumed safely.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

from compare_baselines import (
    run_smo_pop, run_cma_es, run_differential_evolution,
    run_dual_annealing, run_random_search,
)
from opfunu.cec_based import cec2017

METHODS = {
    "SMO-Pop": run_smo_pop,
    "CMA-ES": run_cma_es,
    "DiffEvol": run_differential_evolution,
    "DualAnneal": run_dual_annealing,
    "RandSearch": run_random_search,
}

FUNC_NUMS = [1] + list(range(3, 30))  # F1, F3..F29 (28 funcs)

CLASSES = {
    "unimodal": [1, 3],
    "multimodal": list(range(4, 11)),
    "hybrid": list(range(11, 21)),
    "composition": list(range(21, 30)),
}


def get_problem(n, dim):
    cls = getattr(cec2017, f"F{n}2017")
    return cls(ndim=dim)


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    done.add((r["func"], r["method"], r["seed"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def run(dim=10, max_evals=10000, n_runs=5, seed0=1, methods=None,
        funcs=None, out="cec_results.jsonl"):
    methods = methods or list(METHODS)
    funcs = funcs or FUNC_NUMS
    done = load_done(out)
    total = len(funcs) * len(methods) * n_runs
    skip = sum(1 for fn in funcs for m in methods for r in range(n_runs)
               if (fn, m, seed0 + r) in done)
    print(f"CEC2017 dim={dim} evals={max_evals} runs={n_runs} "
          f"funcs={len(funcs)} methods={methods}", flush=True)
    print(f"{total} combos, {skip} already done, {total - skip} to run.",
          flush=True)
    t_all = time.time()
    fh = open(out, "a")
    try:
        for fn in funcs:
            prob = get_problem(fn, dim)
            fopt = float(prob.f_global)
            lo, hi = -100.0, 100.0

            def func(x, _p=prob):
                return float(_p.evaluate(np.asarray(x, dtype=float)))

            for m in methods:
                mrun = METHODS[m]
                for r in range(n_runs):
                    seed = seed0 + r
                    if (fn, m, seed) in done:
                        continue
                    t0 = time.time()
                    try:
                        best = float(mrun(func, dim, lo, hi, max_evals, seed))
                    except Exception as e:  # never let one cell kill the grid
                        print(f"F{fn} {m} seed={seed} FAILED: {e}", flush=True)
                        best = float("inf")
                    dt = time.time() - t0
                    rec = {"func": fn, "method": m, "seed": seed,
                           "best": best, "error": best - fopt, "time": dt,
                           "dim": dim, "max_evals": max_evals}
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    print(f"F{fn:2d} {m:10s} seed={seed} err={best - fopt:.4g} "
                          f"({dt:.1f}s)", flush=True)
    finally:
        fh.close()
    print(f"Done in {time.time() - t_all:.1f}s -> {out}", flush=True)


def summarize(path, baseline="SMO-Pop"):
    recs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    if not recs:
        print("No records in", path)
        return
    methods = sorted({r["method"] for r in recs},
                     key=list(METHODS).index if set(
                         r["method"] for r in recs) <= set(METHODS) else None)
    funcs = sorted({r["func"] for r in recs})
    # mean error per (func, method)
    mean_err = {}
    for fn in funcs:
        for m in methods:
            errs = [r["error"] for r in recs
                    if r["func"] == fn and r["method"] == m]
            if errs:
                mean_err[(fn, m)] = float(np.mean(errs))
    # ranks per function (1 = best)
    ranks = {m: [] for m in methods}
    best_count = {m: 0 for m in methods}
    for fn in funcs:
        vals = [(mean_err.get((fn, m), np.inf), m) for m in methods]
        vals.sort(key=lambda t: t[0])
        for rank, (_, m) in enumerate(vals, start=1):
            ranks[m].append(rank)
        best_count[vals[0][1]] += 1
    n = len(funcs)
    print(f"\n===== {path} ({n} funcs, "
          f"{len({r['seed'] for r in recs})} seeds) =====")
    print("--- MEAN ERROR per function (lower better) ---")
    hdr = f"{'func':6s}" + "".join(f"{m:>12s}" for m in methods)
    print(hdr)
    for fn in funcs:
        row = f"F{fn:<5d}" + "".join(f"{mean_err.get((fn, m), np.inf):12.4g}"
                                     for m in methods)
        print(row)
    print("--- MEAN RANK (lower better) + #best ---")
    for m in methods:
        print(f"{m:10s} rank={np.mean(ranks[m]):.2f}  best-on={best_count[m]}/{n}")
    # per-class mean rank
    print("--- MEAN RANK by class ---")
    print(f"{'class':12s}" + "".join(f"{m:>12s}" for m in methods))
    for cname, cfuncs in CLASSES.items():
        cf = [f for f in cfuncs if f in funcs]
        if not cf:
            continue
        row = f"{cname:12s}"
        for m in methods:
            idx = [funcs.index(f) for f in cf]
            row += f"{np.mean([ranks[m][i] for i in idx]):12.2f}"
        print(row)
    # Wilcoxon signed-rank vs baseline, paired by function
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None
    if wilcoxon is not None and baseline in methods:
        print(f"--- WILCOXON vs {baseline} (paired by func, "
              f"lower error better) ---")
        xb = np.array([mean_err.get((fn, baseline), np.inf) for fn in funcs])
        for m in methods:
            if m == baseline:
                continue
            xm = np.array([mean_err.get((fn, m), np.inf) for fn in funcs])
            w, p = wilcoxon(xb, xm, alternative="less")
            wins = int(np.sum(xb < xm))
            verb = "SIGNIFICANTLY BETTER" if p < 0.05 else "not significant"
            print(f"{baseline} vs {m}: W={w:.0f} p={p:.4f} "
                  f"wins={wins}/{n} -> {verb}")
    # wall time per method
    print("--- WALL TIME per method ---")
    for m in methods:
        t = sum(r["time"] for r in recs if r["method"] == m)
        print(f"{m:10s} {t:.1f}s total")


def parse_funcs(s):
    out = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return [f for f in out if f in FUNC_NUMS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=10)
    ap.add_argument("--max-evals", type=int, default=10000)
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--methods", type=str, default=",".join(METHODS))
    ap.add_argument("--funcs", type=str, default="all")
    ap.add_argument("--out", type=str, default="cec_results.jsonl")
    ap.add_argument("--summarize", type=str, default=None)
    ap.add_argument("--baseline", type=str, default="SMO-Pop")
    args = ap.parse_args()
    if args.summarize:
        summarize(args.summarize, baseline=args.baseline)
        return
    methods = [m.strip() for m in args.methods.split(",")]
    bad = [m for m in methods if m not in METHODS]
    if bad:
        sys.exit(f"Unknown methods: {bad} (choose from {list(METHODS)})")
    funcs = FUNC_NUMS if args.funcs == "all" else parse_funcs(args.funcs)
    run(dim=args.dim, max_evals=args.max_evals, n_runs=args.n_runs,
        seed0=args.seed0, methods=methods, funcs=funcs, out=args.out)


if __name__ == "__main__":
    main()
