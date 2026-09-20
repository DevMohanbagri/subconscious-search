# subconscious-search

**SMO Upgraded: Natural Metric Latent Space & Separable CMA Conscious Phase.**

Black-box optimizer combining three high-impact phases:

1. **Conscious — `FastSeparableNES`**: coordinate-wise natural gradient
   adaptation (Sep-CMA-ES style diagonal evolution strategy with 1/5th
   success-rule step-size control).
2. **Subconscious latent engine — `FastSupervisedLatentSpace` + `FastMaternSurrogate`**:
   supervised elite-weighted SVD projection (whitened latent coords) with a
   Matérn-3/2 kernel-regression surrogate and analytical LCB-style
   uncertainty for acquisition scoring.
3. **Gate — `SigmoidGate`**: annealed sigmoid selection between the conscious
   and subconscious proposals,
   `p(sub) = sigmoid((score_sub − score_con) / T)`, `T: 1.0 → 0.05`.
   Hard selection (no destructive x-space averaging); both sides are
   max-over-samples surrogate scores so the comparison is unbiased.

The `SMOUpgraded` orchestrator adds: surrogate pre-filtering of conscious
candidates, elite+recent memory subsampling with vectorized latent mapping,
epsilon-global escape candidates, and a relative-improvement stagnation
restart (sigma reset + annealing reboot + guided global leap).

## Run

```bash
pip install numpy scipy
python smo_upgraded.py --dim 10 --max-evals 5000 --n-runs 5
# comparison vs baselines (needs scipy + cma):
pip install cma
python compare_baselines.py --dim 10 --max-evals 5000 --n-runs 5
```

## Comparison vs established optimizers (dim=10, 5000 evals, 5 runs)

Same eval budget for every method. Full log: `comparison_results.txt`.

| function   | SMO (ours) | CMA-ES  | DiffEvol | DualAnneal | RandSearch |
|------------|------------|---------|----------|------------|------------|
| sphere     | 100.00%    | 100.00% | 88.95%   | 100.00%    | 6.37%      |
| rastrigin  | 7.95%      | 7.02%   | 2.14%    | 76.71%     | 1.27%      |
| rosenbrock | 21.14%     | 99.61%  | 2.84%    | 84.01%     | 0.05%      |
| ackley     | 89.14%     | 100.00% | 37.77%   | 100.00%    | 11.02%     |
| griewank   | 91.41%     | 99.04%  | 66.75%   | 96.14%     | 68.31%     |
| **MEAN**   | **61.93%** | **81.13%** | **39.69%** | **91.37%** | **17.40%** |

Takeaways: SMO beats Differential Evolution and Random Search on all 5
functions, ties CMA-ES on sphere/rastrigin, but trails it on rosenbrock
(full covariance wins in the curved valley — SMO is diagonal-only) and
trails Dual Annealing overall (SciPy's local-search hybrid is very strong
on smooth benchmarks). SMO's surrogate overhead (~8 s/run vs <0.5 s) pays
off only when true function evals are expensive — the intended use case.

## Results (dim=10, 5 runs)

Accuracy = `100 / (1 + best_loss)` (100% at the global optimum).
Gap-closed = fraction of the init→optimum gap eliminated.
Full per-run logs: `results_dim10.txt` (2k evals), `results_dim10_5k.txt` (5k evals).

**5000 evals:**

| function   | mean acc | best run | mean loss | gap-closed | success |
|------------|----------|----------|-----------|------------|---------|
| sphere     | 100.00%  | 100.00%  | 7.8e-07   | 100.0%     | 100%    |
| rastrigin  | 7.95%    | 20.07%   | 19.9      | ~86%       | 0%      |
| rosenbrock | 21.14%   | 38.05%   | 4.31      | ~100%      | 0%      |
| ackley     | 89.14%   | 99.97%   | 0.23      | ~98%       | 80%     |
| griewank   | 91.41%   | 95.54%   | 0.098     | ~91%       | 0%      |

**2000 evals:**

| function   | mean acc | best run | mean loss | success |
|------------|----------|----------|-----------|---------|
| sphere     | 98.92%   | 99.82%   | 0.011     | 0%      |
| rastrigin  | 7.15%    | 16.24%   | 20.3      | 0%      |
| rosenbrock | 6.67%    | 10.25%   | 23.7      | 0%      |
| ackley     | 77.36%   | 92.55%   | 0.38      | 0%      |
| griewank   | 91.35%   | 95.48%   | 0.099     | 0%      |

Success tolerances: sphere 1e-4, rastrigin 1.0, rosenbrock 1.0,
ackley 1e-2, griewank 1e-2. ~8 s/run at 5k evals (NumPy/SciPy, CPU).
