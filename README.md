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
```

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
