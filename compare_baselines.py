#!/usr/bin/env python3
"""Fair comparison: SMO Upgraded vs established black-box optimizers.

Baselines (all given the SAME function-evaluation budget):
  - random_search : uniform sampling (sanity floor)
  - differential_evolution (scipy, polish=False to respect budget)
  - dual_annealing (scipy)
  - cma_es (pycma, the gold-standard evolution strategy)

Metrics identical to smo_upgraded.py: accuracy = 100/(1+best_loss).
"""

import time
import numpy as np
from scipy.optimize import differential_evolution, dual_annealing
import cma

from smo_upgraded import SMOUpgraded, BENCHMARKS, TOLS, accuracy_score
from smo_pop import SMOPop


def run_random_search(func, dim, lo, hi, max_evals, seed):
    rng = np.random.default_rng(seed)
    best = np.inf
    for _ in range(max_evals):
        x = rng.uniform(lo, hi, size=dim)
        y = float(func(x))
        if y < best:
            best = y
    return best


def run_differential_evolution(func, dim, lo, hi, max_evals, seed):
    bounds = [(lo, hi)] * dim
    popsize = 15
    # budget: popsize*dim evals per generation -> match max_evals, no early stop
    maxiter = max(1, round(max_evals / (popsize * dim)) - 1)
    res = differential_evolution(func, bounds, maxiter=maxiter, tol=0,
                                 popsize=popsize, seed=seed, polish=False,
                                 updating='deferred', workers=1)
    return float(res.fun)


def run_dual_annealing(func, dim, lo, hi, max_evals, seed):
    bounds = [(lo, hi)] * dim
    res = dual_annealing(func, bounds, maxfun=max_evals, seed=seed)
    return float(res.fun)


def run_cma_es(func, dim, lo, hi, max_evals, seed):
    width = hi - lo
    x0 = np.random.RandomState(seed).uniform(lo, hi, size=dim)
    res = cma.fmin(func, x0, width / 4.0,
                   options={'bounds': [lo, hi], 'maxfevals': max_evals,
                            'seed': seed, 'verbose': -9, 'verb_log': 0})
    return float(res[1])


def run_smo(func, dim, lo, hi, max_evals, seed):
    opt = SMOUpgraded(dim, lo, hi, seed=seed)
    _, best_y, _ = opt.optimize(func, max_evals)
    return best_y


def run_smo_pop(func, dim, lo, hi, max_evals, seed):
    opt = SMOPop(dim, lo, hi, seed=seed)
    _, best_y, _ = opt.optimize(func, max_evals)
    return best_y


METHODS = [
    ("SMO-v1", run_smo),
    ("SMO-Pop", run_smo_pop),
    ("CMA-ES", run_cma_es),
    ("DiffEvol", run_differential_evolution),
    ("DualAnneal", run_dual_annealing),
    ("RandSearch", run_random_search),
]


def compare(dim=10, max_evals=5000, n_runs=5, seed0=0):
    print(f"Comparison: dim={dim}, budget={max_evals} evals, {n_runs} runs "
          f"per method/function")
    table = {}  # func -> method -> dict
    for fname, (func, lo, hi) in BENCHMARKS.items():
        print(f"\n=== {fname} ===")
        table[fname] = {}
        for mname, mrun in METHODS:
            losses, accs, times = [], [], []
            succ = 0
            for r in range(n_runs):
                t0 = time.time()
                try:
                    best = mrun(func, dim, lo, hi, max_evals, seed0 + r)
                except Exception as e:  # never let one baseline kill the run
                    print(f"  {mname} run {r+1} FAILED: {e}")
                    best = np.inf
                dt = time.time() - t0
                losses.append(best)
                accs.append(accuracy_score(best))
                times.append(dt)
                if best <= TOLS[fname]:
                    succ += 1
            row = {"loss": float(np.mean(losses)),
                   "acc": float(np.mean(accs)),
                   "succ": succ / n_runs,
                   "time": float(np.mean(times))}
            table[fname][mname] = row
            print(f"  {mname:10s} acc={row['acc']:6.2f}% loss={row['loss']:.4g} "
                  f"success={row['succ']*100:.0f}% time={row['time']:.1f}s")
    return table


def print_summary(table):
    names = [m for m, _ in METHODS]
    print("\n================ MEAN ACCURACY % (higher better) ================")
    hdr = f"{'function':10s}" + "".join(f"{m:>12s}" for m in names)
    print(hdr)
    for fname in BENCHMARKS:
        line = f"{fname:10s}" + "".join(
            f"{table[fname][m]['acc']:>11.2f}%" for m in names)
        print(line)
    means = {m: float(np.mean([table[f][m]["acc"] for f in BENCHMARKS]))
             for m in names}
    print("-" * len(hdr))
    print(f"{'MEAN':10s}" + "".join(f"{means[m]:>11.2f}%" for m in names))
    print("=" * len(hdr))
    print("\n================ MEAN LOSS (lower better) =======================")
    print(hdr)
    for fname in BENCHMARKS:
        line = f"{fname:10s}" + "".join(
            f"{table[fname][m]['loss']:>12.4g}" for m in names)
        print(line)
    print("=" * len(hdr))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=10)
    ap.add_argument("--max-evals", type=int, default=5000)
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    table = compare(dim=args.dim, max_evals=args.max_evals,
                    n_runs=args.n_runs, seed0=args.seed0)
    print_summary(table)
    print(f"\nTotal wall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
