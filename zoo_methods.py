#!/usr/bin/env python3
"""Classic nature-inspired baselines via mealpy (unified interface).

Budget fairness: nominal epoch x pop_size ~= max_evals, PLUS a hard cap
wrapper: after max_evals true evaluations the wrapper returns the
worst-seen-so-far (no new information, no crash risk from inf), so every
method gets exactly max_evals of real signal. Reported best = best seen
within budget (tracked by the wrapper, immune to post-budget artifacts).
"""

import numpy as np


class _Cap:
    def __init__(self, func, max_evals):
        self.f = func
        self.max = max_evals
        self.n = 0
        self.best = float("inf")
        self.worst = None

    def __call__(self, x):
        if self.n >= self.max:
            return self.worst
        y = float(self.f(np.asarray(x, dtype=float)))
        self.n += 1
        if y < self.best:
            self.best = y
        if self.worst is None or y > self.worst:
            self.worst = y
        return y


def _run_mealpy(cls, func, dim, lo, hi, max_evals, seed, pop_size=100):
    from mealpy.utils.space import FloatVar
    cap = _Cap(func, max_evals)
    epoch = max(1, round(max_evals / pop_size) - 1)
    model = cls(epoch=epoch, pop_size=pop_size)
    problem = {"obj_func": cap,
               "bounds": FloatVar(lb=[float(lo)] * dim, ub=[float(hi)] * dim),
               "minmax": "min"}
    model.solve(problem, seed=seed)
    assert cap.n == max_evals, f"budget leak: {cap.n} != {max_evals}"
    return cap.best


def run_pso(func, dim, lo, hi, max_evals, seed):
    from mealpy.swarm_based.PSO import OriginalPSO
    return _run_mealpy(OriginalPSO, func, dim, lo, hi, max_evals, seed)


def run_ga(func, dim, lo, hi, max_evals, seed):
    # NOTE: mealpy's OriginalGA.evolve is an empty stub (v3.0.3); BaseGA
    # is the functional classic-GA implementation.
    from mealpy.evolutionary_based.GA import BaseGA
    return _run_mealpy(BaseGA, func, dim, lo, hi, max_evals, seed)


def run_gwo(func, dim, lo, hi, max_evals, seed):
    from mealpy.swarm_based.GWO import OriginalGWO
    return _run_mealpy(OriginalGWO, func, dim, lo, hi, max_evals, seed)


def run_woa(func, dim, lo, hi, max_evals, seed):
    from mealpy.swarm_based.WOA import OriginalWOA
    return _run_mealpy(OriginalWOA, func, dim, lo, hi, max_evals, seed)


def run_abc(func, dim, lo, hi, max_evals, seed):
    from mealpy.swarm_based.ABC import OriginalABC
    return _run_mealpy(OriginalABC, func, dim, lo, hi, max_evals, seed)


def run_shade(func, dim, lo, hi, max_evals, seed):
    from mealpy.evolutionary_based.SHADE import OriginalSHADE
    return _run_mealpy(OriginalSHADE, func, dim, lo, hi, max_evals, seed)


def run_fox(func, dim, lo, hi, max_evals, seed):
    from mealpy.swarm_based.FOX import OriginalFOX
    return _run_mealpy(OriginalFOX, func, dim, lo, hi, max_evals, seed)


def run_hba(func, dim, lo, hi, max_evals, seed):
    from mealpy.swarm_based.HBA import OriginalHBA
    return _run_mealpy(OriginalHBA, func, dim, lo, hi, max_evals, seed)


def run_tso(func, dim, lo, hi, max_evals, seed):
    from mealpy.swarm_based.TSO import OriginalTSO
    return _run_mealpy(OriginalTSO, func, dim, lo, hi, max_evals, seed)
