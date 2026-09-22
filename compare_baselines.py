#!/usr/bin/env python3
"""Fair comparison: SMO vs established black-box optimizers.

Baselines (all given the SAME function-evaluation budget):
  - random_search : uniform sampling (sanity floor)
  - differential_evolution (scipy, polish=False to respect budget)
  - dual_annealing (scipy)
  - cma_es (pycma, the gold-standard evolution strategy)

Suites: core (sphere/rastrigin/rosenbrock/ackley/griewank),
extra (schwefel/levy/michalewicz/styblinski/ellipsoid/zakharov/
noisy_sphere), or all. Metrics: accuracy = 100/(1+best_loss),
success rate at per-function tolerances, and mean rank.
"""

import time
import numpy as np
from scipy.optimize import differential_evolution, dual_annealing
import cma

from smo_upgraded import SMOUpgraded, BENCHMARKS, TOLS, accuracy_score
from smo_pop import SMOPop
from smo_ghost import SMOGhost
from extra_benchmarks import NEWBENCHMARKS, get_func


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


def run_bipop(func, dim, lo, hi, max_evals, seed):
    width = hi - lo
    x0 = np.random.RandomState(seed).uniform(lo, hi, size=dim)
    res = cma.fmin(func, x0, width / 4.0,
                   options={'bounds': [lo, hi], 'maxfevals': max_evals,
                            'seed': seed, 'verbose': -9, 'verb_log': 0},
                   restarts=100, bipop=True)
    return float(res[1])


def run_smo(func, dim, lo, hi, max_evals, seed):
    opt = SMOUpgraded(dim, lo, hi, seed=seed)
    _, best_y, _ = opt.optimize(func, max_evals)
    return best_y


def run_smo_pop(func, dim, lo, hi, max_evals, seed):
    opt = SMOPop(dim, lo, hi, seed=seed)
    _, best_y, _ = opt.optimize(func, max_evals)
    return best_y


def run_smo_ghost(func, dim, lo, hi, max_evals, seed):
    opt = SMOGhost(dim, lo, hi, seed=seed)
    _, best_y, _ = opt.optimize(func, max_evals)
    return best_y


ALL_METHODS = [
    ("SMO-v1", run_smo),
    ("SMO-Pop", run_smo_pop),
    ("SMO-Ghost", run_smo_ghost),
    ("CMA-ES", run_cma_es),
    ("DiffEvol", run_differential_evolution),
    ("DualAnneal", run_dual_annealing),
    ("RandSearch", run_random_search),
]


def build_suite(suite, dim):
    """Return ordered [(fname, lo, hi, tol, needs_seed)] + func resolver."""
    items = []
    if suite in ("core", "all"):
        for fname, (_, lo, hi) in BENCHMARKS.items():
            items.append((fname, lo, hi, TOLS[fname], False))
    if suite in ("extra", "all"):
        for fname, (_, lo, hi, tol, dims) in NEWBENCHMARKS.items():
            if dims is not None and dim not in dims:
                print(f"  (skip {fname}: no reference optimum for dim={dim})")
                continue
            items.append((fname, lo, hi, tol, fname == "noisy_sphere"))
    return items


def resolve_func(fname, dim, seed):
    if fname in BENCHMARKS:
        return BENCHMARKS[fname][0]
    f, _, _, _ = get_func(fname, dim, seed=seed)
    return f


def compare(dim=10, max_evals=5000, n_runs=5, seed0=0, suite="core",
            methods=None):
    methods = methods or ALL_METHODS
    items = build_suite(suite, dim)
    print(f"Comparison: suite={suite} dim={dim}, budget={max_evals} evals, "
          f"{n_runs} runs per method/function")
    table = {}
    for fname, lo, hi, tol, needs_seed in items:
        print(f"\n=== {fname} ===")
        table[fname] = {}
        for mname, mrun in methods:
            losses, accs, times = [], [], []
            succ = 0
            for r in range(n_runs):
                t0 = time.time()
                try:
                    func = resolve_func(fname, dim, seed0 + r)
                    best = mrun(func, dim, lo, hi, max_evals, seed0 + r)
                except Exception as e:  # never let one baseline kill the run
                    print(f"  {mname} run {r+1} FAILED: {e}")
                    best = np.inf
                dt = time.time() - t0
                losses.append(best)
                accs.append(accuracy_score(best))
                times.append(dt)
                if best <= tol:
                    succ += 1
            row = {"loss": float(np.mean(losses)),
                   "acc": float(np.mean(accs)),
                   "succ": succ / n_runs,
                   "time": float(np.mean(times))}
            table[fname][mname] = row
            print(f"  {mname:10s} acc={row['acc']:6.2f}% loss={row['loss']:.4g} "
                  f"success={row['succ']*100:.0f}% time={row['time']:.1f}s")
    return table


def print_summary(table, methods):
    names = [m for m, _ in methods]
    fnames = list(table.keys())
    print("\n================ MEAN ACCURACY % (higher better) ================")
    hdr = f"{'function':12s}" + "".join(f"{m:>12s}" for m in names)
    print(hdr)
    for fname in fnames:
        line = f"{fname:12s}" + "".join(
            f"{table[fname][m]['acc']:>11.2f}%" for m in names)
        print(line)
    means = {m: float(np.mean([table[f][m]["acc"] for f in fnames]))
             for m in names}
    print("-" * len(hdr))
    print(f"{'MEAN':12s}" + "".join(f"{means[m]:>11.2f}%" for m in names))
    # mean rank (robust to loss-scale differences across functions)
    ranks = {m: [] for m in names}
    for fname in fnames:
        order = sorted(names, key=lambda m: -table[fname][m]["acc"])
        for rank, m in enumerate(order, start=1):
            ranks[m].append(rank)
    print(f"{'MEAN RANK':12s}" + "".join(
        f"{np.mean(ranks[m]):>12.2f}" for m in names))
    print("=" * len(hdr))
    print("\n================ MEAN LOSS (lower better) =======================")
    print(hdr)
    for fname in fnames:
        line = f"{fname:12s}" + "".join(
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
    ap.add_argument("--suite", choices=["core", "extra", "all"], default="core")
    ap.add_argument("--methods", default="",
                    help="comma-separated subset of: "
                         "SMO-v1,SMO-Pop,CMA-ES,DiffEvol,DualAnneal,RandSearch")
    args = ap.parse_args()
    methods = ALL_METHODS
    if args.methods:
        want = set(s.strip() for s in args.methods.split(","))
        methods = [m for m in ALL_METHODS if m[0] in want]
    t0 = time.time()
    table = compare(dim=args.dim, max_evals=args.max_evals,
                    n_runs=args.n_runs, seed0=args.seed0, suite=args.suite,
                    methods=methods)
    print_summary(table, methods)
    print(f"\nTotal wall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
