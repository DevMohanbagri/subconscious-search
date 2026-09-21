# subconscious-search

**SMO: a gated conscious/subconscious black-box optimizer.**

Two versions:
- **v1** (`smo_upgraded.py`) — diagonal Sep-CMA-ES conscious step +
  supervised latent space + Matérn surrogate + annealed sigmoid gate.
  Single-incumbent, one eval per iteration.
- **v2/v3** (`smo_pop.py`, current best) — **full-covariance Active-CMA-ES**
  conscious core in a **generational population loop**: each generation
  evaluates λ CMA offspring plus k surrogate-selected subconscious
  candidates, where k is set by a **credit-assignment gate** (each stream
  earns future allocation from the improvements it produces, bandit-style).
  Stagnation drains the incumbent basin with budget-counted L-BFGS-B, then
  IPOP-restarts; **two consecutive stalled windows switch all remaining
  budget to a terminal hopping phase** (block-coordinate annealing walk
  with Cauchy jumps + periodic L-BFGS-B drainage), which cracks
  Rastrigin-class multimodality. Smooth runs never stall twice, so they
  never pay for it.

## Run

```bash
pip install numpy scipy
python smo_pop.py --dim 10 --max-evals 5000 --n-runs 5        # v3 (best)
python smo_upgraded.py --dim 10 --max-evals 5000 --n-runs 5   # v1
# comparison vs baselines (needs scipy + cma):
pip install cma
python compare_baselines.py --dim 10 --max-evals 5000 --n-runs 5 --seed0 1
```

## Results (dim=10, 5 runs, 5000 evals)

Accuracy = `100 / (1 + best_loss)` (100% at the global optimum).
Full per-run logs: `results_popv3_dim10_5k.txt` (v3),
`results_pop_dim10_5k.txt` (v2), `results_dim10_5k.txt` (v1).

| function   | v3 acc  | v3 loss | v2 acc | v1 acc  |
|------------|---------|---------|--------|---------|
| sphere     | 100.00% | 1.0e-27 | 100.00%| 100.00% |
| rastrigin  | 50.09%  | 1.79    | 10.01% | 7.95%   |
| rosenbrock | 96.64%  | 0.037   | 96.64% | 21.14%  |
| ackley     | 100.00% | 4.4e-14 | 100.00%| 89.14%  |
| griewank   | 99.27%  | 0.0074  | 99.03% | 91.41%  |
| **MEAN**   | **89.20%** |      | **81.14%** | **61.93%** |

## Comparison vs established optimizers (dim=10, 5000 evals, seeds 1–5)

Same eval budget for every method. Full log: `comparison_results_v3.txt`.

| function   | SMO-Pop (v3) | SMO-v1 | CMA-ES | DiffEvol | DualAnneal | RandSearch |
|------------|--------------|--------|--------|----------|------------|------------|
| sphere     | 100.00%      | 100.00%| 100.00%| 92.26%   | 100.00%    | 6.02%      |
| rastrigin  | 50.09%       | 6.93%  | 7.51%  | 2.27%    | 76.71%     | 1.33%      |
| rosenbrock | 96.64%       | 20.96% | 83.62% | 2.88%    | 84.01%     | 0.04%      |
| ackley     | 100.00%      | 75.18% | 100.00%| 38.40%   | 100.00%    | 10.82%     |
| griewank   | 99.27%       | 91.16% | 99.04% | 66.72%   | 95.60%     | 66.22%     |
| **MEAN**   | **89.20%**   | **58.85%** | **78.03%** | **40.51%** | **91.26%** | **16.89%** |

Takeaways: SMO-Pop v3 is best-in-class on rosenbrock (96.64%, beating
both CMA-ES and Dual Annealing) and griewank, tied on sphere/ackley, and
within 2 points of Dual Annealing overall — trailing only on rastrigin,
where annealing specialists still lead but the gap closed from 69 to 27
points (7x accuracy gain over v2).

## SMO on real data: BRFSS 2015 diabetes (`ml_benchmark.py`)

Binary 50/50 split (70,692 rows, 21 features; data mirrors UCI id 891 —
place CSVs in `data/`, gitignored). Stratified 70/15/15 split, seed 42.
Full log: `ml_results.txt`.

| method | test acc | test F1 | test AUC |
|---|---|---|---|
| dummy (majority) | 50.00% | 0.0000 | — |
| logreg (sklearn LBFGS) | 74.76% | 0.7522 | 0.8250 |
| **SMO-direct** (22-D weights) | **74.55%** | 0.7489 | 0.8239 |
| randomforest (300) | 73.64% | 0.7471 | 0.8119 |
| hgb (defaults) | 75.26% | 0.7630 | 0.8301 |
| mlp (64x32) | 73.21% | 0.7298 | 0.8082 |
| randsearch-HPO (108 evals) | 75.29% | 0.7625 | 0.8301 |
| **SMO-HPO** (108 evals) | **75.11%** | 0.7618 | 0.8299 |

Takeaways: SMO trained logistic weights from scratch to within 0.2pp of
LBFGS (74.55 vs 74.76%, 2000 evals, 0.8 s). For HPO all three (defaults /
random / SMO) tie at ~75.1–75.3%: this dataset plateaus there and tuning
barely matters (SMO actually found the best *validation* loss, 0.50110
vs 0.50129; test noise flips the ranking). Note: the 3-class `012` file
is 84/14/2 imbalanced, so raw accuracy is misleading there
(dummy = 84.24%; HGB = 84.91%, macro-F1 0.40).
