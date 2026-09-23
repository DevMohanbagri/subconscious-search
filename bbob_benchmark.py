#!/usr/bin/env python3
"""
BBOB benchmarks via IOHexperimenter (24 noiseless functions, f1..f24).

Fixed budget per run + COCO-style fixed-target runtimes: 51 log-spaced
targets 1e2..1e-8 on precision (f_best - f_opt); first-hitting eval is
recorded per target during the run (no full histories stored).

Usage:
  python bbob_benchmark.py --dim 10 --max-evals 10000 --instances 1-5 \\
      --methods SMO-Pop,CMA-ES,BIPOP,DiffEvol,DualAnneal,RandSearch \\
      --out bbob_dim10.jsonl
  python bbob_benchmark.py --summarize bbob_dim10.jsonl

JSONL + resume: completed (fid, instance, method, seed) combos are skipped.
Optimizer seed = seed0 + instance (1 trial per instance, COCO-style).
"""

import argparse
import json
import os
import sys
import time

import numpy as np

from compare_baselines import (
    run_smo_pop, run_cma_es, run_bipop, run_differential_evolution,
    run_dual_annealing, run_random_search,
)
from zoo_methods import (
    run_pso, run_ga, run_gwo, run_woa, run_abc, run_shade,
)

METHODS = {
    "SMO-Pop": run_smo_pop,
    "CMA-ES": run_cma_es,
    "BIPOP": run_bipop,
    "DiffEvol": run_differential_evolution,
    "DualAnneal": run_dual_annealing,
    "RandSearch": run_random_search,
    "PSO": run_pso,
    "GA": run_ga,
    "GWO": run_gwo,
    "WOA": run_woa,
    "ABC": run_abc,
    "SHADE": run_shade,
}

FIDS = list(range(1, 25))
GROUPS = {
    "separable": [1, 2, 3, 4, 5],
    "low-cond": [6, 7, 8, 9],
    "high-cond": [10, 11, 12, 13, 14],
    "multi-strong": [15, 16, 17, 18, 19],
    "multi-weak": [20, 21, 22, 23, 24],
}
# 51 COCO targets, easy -> hard
TARGETS = [10.0 ** (2 - i / 5.0) for i in range(51)]


class Recorder:
    """Wraps an IOH problem: counts evals, tracks best + target runtimes."""

    def __init__(self, problem):
        self.p = problem
        self.fopt = float(problem.optimum.y)
        self.n = 0
        self.best = float("inf")
        self.runtimes = [-1] * len(TARGETS)
        self._next = 0  # next unsolved target index (targets descending)

    def __call__(self, x):
        y = float(self.p(list(np.asarray(x, dtype=float))))
        self.n += 1
        if y < self.best:
            self.best = y
            prec = y - self.fopt
            while self._next < len(TARGETS) and prec <= TARGETS[self._next]:
                self.runtimes[self._next] = self.n
                self._next += 1
        return y


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
                    done.add((r["fid"], r["instance"], r["method"], r["seed"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def run(dim=10, max_evals=10000, instances=None, seed0=0, methods=None,
        fids=None, out="bbob_results.jsonl"):
    import ioh
    methods = methods or list(METHODS)
    fids = fids or FIDS
    instances = instances or [1]
    suite = ioh.suite.BBOB(fids, instances, [dim])
    problems = {(p.meta_data.problem_id, p.meta_data.instance): p
                for p in suite}
    done = load_done(out)
    total = len(fids) * len(instances) * len(methods)
    skip = sum(1 for fi in fids for ii in instances for m in methods
               if (fi, ii, m, seed0 + ii) in done)
    print(f"BBOB dim={dim} evals={max_evals} fids={len(fids)} "
          f"instances={instances} methods={methods}", flush=True)
    print(f"{total} combos, {skip} already done, {total - skip} to run.",
          flush=True)
    t_all = time.time()
    fh = open(out, "a")
    try:
        for fi in fids:
            for ii in instances:
                p = problems[(fi, ii)]
                seed = seed0 + ii
                for m in methods:
                    if (fi, ii, m, seed) in done:
                        continue
                    try:
                        p.reset()
                    except Exception:
                        pass
                    rec = Recorder(p)
                    t0 = time.time()
                    try:
                        METHODS[m](rec, dim, -5.0, 5.0, max_evals, seed)
                    except Exception as e:
                        print(f"f{fi} i{ii} {m} FAILED: {e}", flush=True)
                    dt = time.time() - t0
                    row = {"fid": fi, "instance": ii, "dim": dim,
                           "method": m, "seed": seed, "best": rec.best,
                           "precision": rec.best - rec.fopt,
                           "evals": rec.n, "time": dt,
                           "runtimes": rec.runtimes,
                           "max_evals": max_evals}
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    solved = sum(1 for r in rec.runtimes if r > 0)
                    print(f"f{fi:2d} i{ii} {m:10s} prec={rec.best - rec.fopt:.3g} "
                          f"targets={solved}/51 evals={rec.n} ({dt:.1f}s)",
                          flush=True)
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
    methods = sorted({r["method"] for r in recs}, key=list(METHODS).index)
    fids = sorted({r["fid"] for r in recs})
    max_evals = recs[0]["max_evals"]
    n_inst = len({(r["fid"], r["instance"]) for r in recs}) // max(len(fids), 1)
    print(f"\n===== {path} ({len(fids)} funcs x {n_inst} inst, "
          f"budget={max_evals}) =====")
    # ---- fixed-budget: mean precision per func, ranks ----
    mean_prec = {}
    for fi in fids:
        for m in methods:
            ps = [r["precision"] for r in recs
                  if r["fid"] == fi and r["method"] == m]
            if ps:
                mean_prec[(fi, m)] = float(np.mean(ps))
    ranks = {m: [] for m in methods}
    best_count = {m: 0 for m in methods}
    for fi in fids:
        vals = sorted(((mean_prec.get((fi, m), np.inf), m) for m in methods),
                      key=lambda t: t[0])
        for rank, (_, m) in enumerate(vals, start=1):
            ranks[m].append(rank)
        best_count[vals[0][1]] += 1
    print("--- FIXED-BUDGET mean precision per func (lower better) ---")
    print(f"{'func':6s}" + "".join(f"{m:>12s}" for m in methods))
    for fi in fids:
        print(f"f{fi:<5d}" + "".join(f"{mean_prec.get((fi, m), np.inf):12.4g}"
                                     for m in methods))
    print("--- FIXED-BUDGET mean rank + #best ---")
    for m in methods:
        print(f"{m:10s} rank={np.mean(ranks[m]):.2f}  "
              f"best-on={best_count[m]}/{len(fids)}")
    print("--- FIXED-BUDGET mean rank by group ---")
    print(f"{'group':12s}" + "".join(f"{m:>12s}" for m in methods))
    for gname, gf in GROUPS.items():
        gf = [f for f in gf if f in fids]
        if not gf:
            continue
        idx = [fids.index(f) for f in gf]
        print(f"{gname:12s}" + "".join(f"{np.mean([ranks[m][i] for i in idx]):12.2f}"
                                       for m in methods))
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        wilcoxon = None
    if wilcoxon is not None and baseline in methods:
        print(f"--- WILCOXON vs {baseline} on mean precision "
              f"(paired by func) ---")
        xb = np.array([mean_prec.get((fi, baseline), np.inf) for fi in fids])
        for m in methods:
            if m == baseline:
                continue
            xm = np.array([mean_prec.get((fi, m), np.inf) for fi in fids])
            w, p = wilcoxon(xb, xm, alternative="less")
            wins = int(np.sum(xb < xm))
            print(f"{baseline} vs {m}: W={w:.0f} p={p:.4f} "
                  f"wins={wins}/{len(fids)} -> "
                  f"{'SIGNIFICANTLY BETTER' if p < 0.05 else 'not significant'}")
    # ---- fixed-target: ECDF-style solved fractions ----
    print("--- FIXED-TARGET fraction of (run,target) pairs solved ---")
    print(f"{'method':10s} {'solved/total':>12s} {'frac':>7s} "
          f"ECDF@0.1 {'ECDF@0.3':>9s} {'ECDF@1.0':>9s}")
    for m in methods:
        runs = [r for r in recs if r["method"] == m]
        tot = len(runs) * len(TARGETS)
        rt_all = [t for r in runs for t in r["runtimes"]]
        solved = sum(1 for t in rt_all if t > 0)
        e01 = np.mean([0 < t <= 0.1 * max_evals for t in rt_all])
        e03 = np.mean([0 < t <= 0.3 * max_evals for t in rt_all])
        e10 = solved / tot
        print(f"{m:10s} {solved:>6d}/{tot:<6d} "
              f"{e10:6.1%} {e01:9.1%} {e03:9.1%} {e10:9.1%}")
    print("--- FIXED-TARGET solved fraction by group ---")
    print(f"{'group':12s}" + "".join(f"{m:>12s}" for m in methods))
    for gname, gf in GROUPS.items():
        gf = [f for f in gf if f in fids]
        if not gf:
            continue
        row = f"{gname:12s}"
        for m in methods:
            runs = [r for r in recs if r["method"] == m and r["fid"] in gf]
            rt = [t for r in runs for t in r["runtimes"]]
            row += f"{np.mean([t > 0 for t in rt]):12.1%}" if rt else f"{'--':>12s}"
        print(row)
    print("--- WALL TIME per method ---")
    for m in methods:
        print(f"{m:10s} "
              f"{sum(r['time'] for r in recs if r['method'] == m):.1f}s total")


def parse_instances(s):
    out = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def parse_fids(s):
    out = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(f for f in out if f in FIDS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=10)
    ap.add_argument("--max-evals", type=int, default=10000)
    ap.add_argument("--instances", type=str, default="1")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--methods", type=str, default=",".join(METHODS))
    ap.add_argument("--fids", type=str, default="all")
    ap.add_argument("--out", type=str, default="bbob_results.jsonl")
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
    fids = FIDS if args.fids == "all" else parse_fids(args.fids)
    run(dim=args.dim, max_evals=args.max_evals,
        instances=parse_instances(args.instances), seed0=args.seed0,
        methods=methods, fids=fids, out=args.out)


if __name__ == "__main__":
    main()
