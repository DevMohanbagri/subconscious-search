#!/usr/bin/env python3
"""Extended benchmark suite: 6 more functions + noisy variant.

Complements smo_upgraded.BENCHMARKS (sphere/rastrigin/rosenbrock/ackley/
griewank) with different landscape pathologies:
  schwefel    - multimodal, deceptive (optimum far from center, no global trend)
  levy        - multimodal, sharp narrow global basin
  michalewicz - multimodal, steep narrow valleys (dim-10 optimum known)
  styblinski  - multimodal, 2^n local minima, separable
  ellipsoid   - unimodal, separable, condition number 1e6 (covariance test)
  zakharov    - unimodal, narrow curved valley (covariance test)
  noisy_sphere- sphere + N(0, 0.5^2) per eval (seeded RNG; robustness test)

All are shifted so the global optimum value is 0 (like the core suite),
except noisy_sphere (~0 in expectation). michalewicz is only registered
for dims with a known reference optimum (currently dim=10).
"""

import numpy as np


def schwefel(x):
    x = np.asarray(x, dtype=float)
    return float(418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x)))))


def levy(x):
    x = np.asarray(x, dtype=float)
    w = 1.0 + (x - 1.0) / 4.0
    term1 = np.sin(np.pi * w[0]) ** 2
    term3 = (w[-1] - 1.0) ** 2 * (1.0 + np.sin(2 * np.pi * w[-1]) ** 2)
    mid = np.sum((w[:-1] - 1.0) ** 2
                 * (1.0 + 10.0 * np.sin(np.pi * w[:-1] + 1.0) ** 2))
    return float(term1 + mid + term3)


MICH_OPTS = {10: 9.66015171}  # dim -> -f_min (m=10), literature values


def michalewicz(x, m=10):
    x = np.asarray(x, dtype=float)
    i = np.arange(1, len(x) + 1)
    return float(-np.sum(np.sin(x) * np.sin(i * x ** 2 / np.pi) ** (2 * m)))


STB_1D = 39.166165703  # single-dim minimum (separable => n * this)


def styblinski_tang(x):
    x = np.asarray(x, dtype=float)
    return float(np.sum(x ** 4 - 16 * x ** 2 + 5 * x) / 2.0
                 + STB_1D * len(x))


def ellipsoid(x, cond=1e6):
    x = np.asarray(x, dtype=float)
    n = len(x)
    w = cond ** (np.arange(n) / max(n - 1, 1))
    return float(np.sum(w * x ** 2))


def zakharov(x):
    x = np.asarray(x, dtype=float)
    i = np.arange(1, len(x) + 1)
    s1 = np.sum(x ** 2)
    s2 = np.sum(0.5 * i * x)
    return float(s1 + s2 ** 2 + s2 ** 4)


def noisy_sphere_factory(seed=0, sigma=0.5):
    rng = np.random.default_rng(seed)

    def f(x):
        x = np.asarray(x, dtype=float)
        return float(np.sum(x ** 2) + rng.normal(0, sigma))

    return f


# name -> (func_or_factorymarker, lo, hi, tol, dims_or_None)
# tol = success threshold on best loss; dims limits registration.
NEWBENCHMARKS = {
    "schwefel":  (schwefel, -500.0, 500.0, 1.0, None),
    "levy":      (levy, -10.0, 10.0, 1e-3, None),
    "michalewicz": (None, 0.0, np.pi, 1e-2, [10]),  # shifted, see below
    "styblinski": (styblinski_tang, -5.0, 5.0, 1e-1, None),
    "ellipsoid": (ellipsoid, -5.0, 5.0, 1e-6, None),
    "zakharov":  (zakharov, -5.0, 10.0, 1e-2, None),
    "noisy_sphere": ("factory", -5.0, 5.0, 5e-1, None),
}


def get_func(name, dim, seed=0):
    """Materialize (func, lo, hi, tol) for a suite+dim (+seed for noisy)."""
    entry, lo, hi, tol, dims = NEWBENCHMARKS[name]
    if dims is not None and dim not in dims:
        raise ValueError(f"{name} has no reference optimum for dim={dim}")
    if name == "michalewicz":
        off = MICH_OPTS[dim]

        def f(x, _off=off):
            return michalewicz(x) + _off

        return f, lo, hi, tol
    if name == "noisy_sphere":
        return noisy_sphere_factory(seed=seed), lo, hi, tol
    return entry, lo, hi, tol


def verify(dims=(2, 10)):
    """Check f(x_opt) ~= 0 for each function."""
    print("verifying optima (expect ~0):")
    for dim in dims:
        print(f"  dim={dim}")
        print("    schwefel  ", schwefel(np.full(dim, 420.9687)))
        print("    levy      ", levy(np.full(dim, 1.0)))
        if dim in MICH_OPTS:
            from scipy.optimize import minimize
            # reference value check only (true argmin has no closed form)
            print("    mich ref  ", -MICH_OPTS[dim], "(literature f_min)")
        x1d = -2.903534  # one of the 1-D minima
        print("    styblinski", styblinski_tang(np.full(dim, x1d)))
        print("    ellipsoid ", ellipsoid(np.zeros(dim)))
        print("    zakharov  ", zakharov(np.zeros(dim)))


if __name__ == "__main__":
    verify()
