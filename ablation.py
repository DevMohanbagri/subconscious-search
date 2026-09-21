#!/usr/bin/env python3
"""Ablation study: what does each SMO-Pop v3 component contribute?

Configs (all else = full v3 defaults):
  full       - everything on (reference)
  no-active  - plain CMA-ES (negative weights zeroed)
  no-sub     - no subconscious stream (pure CMA + restarts + LS)
  no-credit  - fixed 50/50 gate instead of credit assignment
  no-restart - never restart (patience = inf)
  no-LS      - no L-BFGS-B polish anywhere
  no-phase   - restart polish kept, terminal hopping phase off
  cma-only   - pure Active-CMA-ES (no sub, no LS, no restarts)

Functions: sphere (sanity), rastrigin + schwefel (multimodal),
rosenbrock (valley), ellipsoid (ill-conditioned).
Budget-identical: dim=10, 5000 evals, seeds 1-5.
"""

import time
import numpy as np

from smo_pop import SMOPop
from smo_upgraded import sphere, rastrigin, rosenbrock, accuracy_score
from extra_benchmarks import get_func

FUNCS = {
    "sphere": (sphere, -5.0, 5.0),
    "rastrigin": (rastrigin, -5.12, 5.12),
    "rosenbrock": (rosenbrock, -5.0, 5.0),
    "schwefel": None,   # filled per-dim via get_func
    "ellipsoid": None,
}

CONFIGS = {
    "full": ({}, {}),
    "no-active": ({"active_cma": False}, {}),
    "no-sub": ({"subconscious": False}, {}),
    "no-credit": ({"credit_gate": False}, {}),
    "no-restart": ({}, {"patience_gens": 10 ** 9}),
    "no-LS": ({}, {"local_search": False}),
    "no-phase": ({"use_bh_phase": False}, {}),
    "cma-only": ({"subconscious": False},
                 {"local_search": False, "patience_gens": 10 ** 9}),
}


def resolve_funcs(dim):
    out = {}
    for name, spec in FUNCS.items():
        if spec is None:
            f, lo, hi, _ = get_func(name, dim)
            out[name] = (f, lo, hi)
        else:
            out[name] = spec
    return out


def run(dim=10, max_evals=5000, seeds=(1, 2, 3, 4, 5)):
    funcs = resolve_funcs(dim)
    table = {c: {} for c in CONFIGS}
    for fname, (f, lo, hi) in funcs.items():
        print(f"\n=== {fname} (dim={dim}) ===", flush=True)
        for cname, (init_kw, opt_kw) in CONFIGS.items():
            accs, losses = [], []
            t0 = time.time()
            for s in seeds:
                opt = SMOPop(dim, lo, hi, seed=s, **init_kw)
                _, by, _ = opt.optimize(f, max_evals, **opt_kw)
                accs.append(accuracy_score(by))
                losses.append(by)
            dt = time.time() - t0
            table[cname][fname] = (float(np.mean(accs)), float(np.mean(losses)))
            print(f"  {cname:10s} acc={np.mean(accs):6.2f}% loss={np.mean(losses):.4g} "
                  f"({dt:.0f}s)", flush=True)
    return table


def print_summary(table):
    fnames = next(iter(table.values())).keys()
    print("\n================ ABLATION: mean acc % (higher better) ============")
    hdr = f"{'config':10s}" + "".join(f"{f:>12s}" for f in fnames) + f"{'MEAN':>12s}"
    print(hdr)
    for cname, rows in table.items():
        accs = [rows[f][0] for f in fnames]
        line = f"{cname:10s}" + "".join(f"{a:>11.2f}%" for a in accs)
        line += f"{np.mean(accs):>11.2f}%"
        print(line)
    print("\n================ DELTA vs full (pp; negative = component helped) =")
    print(hdr)
    for cname, rows in table.items():
        ds = [rows[f][0] - table["full"][f][0] for f in fnames]
        line = f"{cname:10s}" + "".join(f"{d:>+12.2f}" for d in ds)
        line += f"{np.mean(ds):>+12.2f}"
        print(line)
    print("=" * len(hdr))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=10)
    ap.add_argument("--max-evals", type=int, default=5000)
    args = ap.parse_args()
    t0 = time.time()
    table = run(dim=args.dim, max_evals=args.max_evals)
    print_summary(table)
    print(f"\nTotal wall time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
