# subconscious-search

**SMO: a gated conscious/subconscious black-box optimizer.**

Two versions:
- **v1** (`smo_upgraded.py`) — diagonal Sep-CMA-ES conscious step +
  supervised latent space + Matérn surrogate + annealed sigmoid gate.
  Single-incumbent, one eval per iteration.
- **v2** (`smo_pop.py`, current best) — **full-covariance Active-CMA-ES**
  conscious core in a **generational population loop**: each generation
  evaluates λ CMA offspring plus k surrogate-selected subconscious
  candidates, where k is set by a **credit-assignment gate** (each stream
  earns future allocation from the improvements it produces, bandit-style).
  Stagnation triggers **IPOP restarts** (population doubling, fresh
  covariance, mean recentered on the global best).

## Run

```bash
pip install numpy scipy
python smo_pop.py --dim 10 --max-evals 5000 --n-runs 5        # v2 (best)
python smo_upgraded.py --dim 10 --max-evals 5000 --n-runs 5   # v1
# comparison vs baselines (needs scipy + cma):
pip install cma
python compare_baselines.py --dim 10 --max-evals 5000 --n-runs 5 --seed0 1
```

## Results (dim=10, 5 runs, 5000 evals)

Accuracy = `100 / (1 + best_loss)` (100% at the global optimum).
Full per-run logs: `results_pop_dim10_5k.txt` (v2), `results_dim10_5k.txt` (v1).

| function   | v2 acc  | v2 loss | v1 acc  | v1 loss |
|------------|---------|---------|---------|---------|
| sphere     | 100.00% | 1.0e-27 | 100.00% | 7.8e-07 |
| rastrigin  | 10.01%  | 12.3    | 7.95%   | 19.9    |
| rosenbrock | 96.64%  | 0.037   | 21.14%  | 4.31    |
| ackley     | 100.00% | 4.4e-14 | 89.14%  | 0.23    |
| griewank   | 99.03%  | 0.0099  | 91.41%  | 0.098   |
| **MEAN**   | **81.14%** |      | **61.93%** |      |

v2 also runs ~7x faster per run (~1.5 s vs ~8 s).

## Comparison vs established optimizers (dim=10, 5000 evals, seeds 1–5)

Same eval budget for every method. Full log: `comparison_results_v2.txt`.

| function   | SMO-Pop (v2) | SMO-v1 | CMA-ES | DiffEvol | DualAnneal | RandSearch |
|------------|--------------|--------|--------|----------|------------|------------|
| sphere     | 100.00%      | 100.00%| 100.00%| 92.26%   | 100.00%    | 6.02%      |
| rastrigin  | 8.69%        | 6.93%  | 7.51%  | 2.27%    | 76.71%     | 1.33%      |
| rosenbrock | 96.64%       | 20.96% | 83.62% | 2.88%    | 84.01%     | 0.04%      |
| ackley     | 100.00%      | 75.18% | 100.00%| 38.40%   | 100.00%    | 10.82%     |
| griewank   | 99.27%       | 91.16% | 99.04% | 66.72%   | 95.60%     | 66.22%     |
| **MEAN**   | **80.92%**   | **58.85%** | **78.03%** | **40.51%** | **91.26%** | **16.89%** |

Takeaways: SMO-Pop beats pycma CMA-ES head-to-head (better on
rosenbrock/griewank/rastrigin, tied on sphere/ackley) and beats
Differential Evolution and Random Search everywhere. Only Dual Annealing
ranks higher overall, almost entirely via Rastrigin, where its local
search cracks basins that trap every evolution strategy. Remaining
frontier: Rastrigin-class multimodality.
