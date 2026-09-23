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

## Extended suite: 7 more functions, dim 10 (`comparison_extra_dim10.txt`)

schwefel / levy / michalewicz / styblinski / ellipsoid / zakharov /
noisy_sphere — same 5k-eval budget, seeds 1–5.

| function   | SMO-Pop | CMA-ES | DiffEvol | DualAnneal | RandSearch |
|------------|---------|--------|----------|------------|------------|
| schwefel   | 0.15%   | 0.10%  | 0.07%    | 99.99%     | 0.05%      |
| levy       | 100.00% | 100.00%| 65.50%   | 100.00%    | 6.90%      |
| michalewicz| 50.29%  | 35.60% | 21.76%   | 90.61%     | 16.30%     |
| styblinski | 62.00%  | 2.57%  | 5.06%    | 100.00%    | 0.98%      |
| ellipsoid  | 100.00% | 100.00%| 0.61%    | 88.75%     | 0.00%      |
| zakharov   | 100.00% | 100.00%| 7.51%    | 100.00%    | 2.19%      |
| noisy_sph. | 100.00% | 100.00%| 100.00%  | 65.70%     | 6.09%      |
| **MEAN**   | **73.21%** | **62.61%** | **28.64%** | **92.15%** | **4.65%** |
| **M.RANK** | **1.71** | **2.29** | **3.71** | **2.29**  | **5.00**   |

Takeaways: SMO has the best mean rank (most consistent — top-2 on 6/7)
and beats CMA-ES on mean again, but Dual Annealing's mean is higher via
schwefel/styblinski blowouts: deceptive landscapes with smooth basins are
annealing+LS territory, and schwefel (optimum near the corner, no global
trend) defeats the whole CMA family. SMO beats DA on ellipsoid
(covariance learning > numeric gradients at cond 1e6) and noisy_sphere
(LS chases noise; ES averages it out).

## Stress test: all 11 functions, dim 30 (`comparison_all_dim30.txt`)

Same 5k-eval budget (starvation rations in 30-D), 3 runs, seeds 1–3
(michalewicz skipped: no dim-30 reference optimum).

| function   | SMO-Pop | CMA-ES | DiffEvol | DualAnneal | RandSearch |
|------------|---------|--------|----------|------------|------------|
| sphere     | 100.00% | 100.00%| 2.03%    | 100.00%    | 0.91%      |
| rastrigin  | 2.03%   | 1.70%  | 0.34%    | 4.86%      | 0.29%      |
| rosenbrock | 3.61%   | 3.41%  | 0.00%    | 100.00%    | 0.00%      |
| ackley     | 100.00% | 100.00%| 10.28%   | 19.20%     | 8.50%      |
| griewank   | 100.00% | 100.00%| 48.84%   | 100.00%    | 47.41%     |
| schwefel   | 0.03%   | 0.02%  | 0.01%    | 33.61%     | 0.01%      |
| levy       | 88.49%  | 63.24% | 1.43%    | 26.55%     | 0.79%      |
| styblinski | 0.76%   | 0.63%  | 0.21%    | 100.00%    | 0.18%      |
| ellipsoid  | 0.06%   | 0.03%  | 0.00%    | 2.06%      | 0.00%      |
| zakharov   | 0.47%   | 0.43%  | 0.38%    | 100.00%    | 0.13%      |
| noisy_sph. | 100.00% | 100.00%| 2.29%    | 1.27%      | 0.91%      |
| **MEAN**   | **45.05%** | **42.68%** | **5.99%** | **53.45%** | **5.38%** |
| **M.RANK** | **2.00** | **2.45** | **3.91** | **1.64**  | **5.00**   |

Takeaways: at dim 30 the CMA family is eval-starved (learning a 30x30
covariance needs ~1k+ evals before it can converge), so Dual Annealing's
gradient-based local search wins valleys/multimodal (rosenbrock,
zakharov, styblinski, schwefel, rastrigin, ellipsoid — least-bad on the
last two). SMO is a solid 2nd by mean and rank, beats CMA-ES on 9/11,
wins levy outright (88.5 vs 63.2/26.6), and owns the robustness corner:
ackley (DA's LS drowns in ripples, 19%) and noisy_sphere (DA chases
noise, 1.3%). Regimes are now clear: DA = smooth/multimodal + high-D
valleys; SMO = ill-conditioning, sharp basins, noise, rippled landscapes.

## Tougher benchmarks: CEC2017 (`cec_benchmark.py`, `cec_dim10.txt`, `cec_dim30.txt`)

28 functions via `opfunu` (F1, F3–F29; F2 officially excluded, F30
unimplemented) — shifted, rotated, hybrid, and composition landscapes,
bounds [−100, 100]^D. Fixed budget of 10k evals for all methods
(disclosed: official CEC protocol is 10000×D; this is a fixed-budget
comparison, 5 seeds dim-10 / 3 seeds dim-30). Metric is mean error to
the known optimum; ranks are per-function (1 = best); significance is
Wilcoxon signed-rank paired by function. Raw per-run records:
`cec_dim10.jsonl` (840 runs), `cec_dim30.jsonl` (504 runs).

Dim 10 (mean error rank, #functions won, Wilcoxon vs SMO):

| method | mean rank | #best/28 | Wilcoxon vs SMO |
|--------|-----------|----------|-----------------|
| **SMO-Pop** | **2.00** | **10** | — |
| CMA-ES | 2.39 | 10 | 16/28 wins, p=0.29 n.s. |
| DualAnneal | 3.36 | 4 | 23/28 wins, p=0.0003 ✅ |
| BIPOP | 3.50 | 2 | 20/28 wins, p=0.020 ✅ |
| DiffEvol | 3.79 | 2 | 25/28 wins, p<0.0001 ✅ |
| RandSearch | 5.96 | 0 | 28/28 wins, p<0.0001 ✅ |

Dim-10 class ranks: SMO is best-or-tied in every class (unimodal 1.50
tie, multimodal 2.14, hybrid 2.30, composition 1.67). Dual Annealing —
the overall leader on the custom suite — collapses to rank 3.36:
rotation + shifts + hybrids defeat its local search, exactly as
predicted. Notably SMO also beats BIPOP significantly (p=0.020):
at a tight 10k budget on hard rotated/hybrid landscapes, BIPOP's
restart overhead burns evals without time to pay off, while SMO's
stall-gated design (no restart until stagnation is proven) wins.
SMO's highlights: F1 bent cigar to 3.6e-06, best-on-10.
Residual weak spots: F28/F29 compositions (CMA 1640/4746 vs SMO
1873/14900) and F10/F18 (plain CMA wins big).

Dim 30 (same format):

| method | mean rank | #best/28 | Wilcoxon vs SMO |
|--------|-----------|----------|-----------------|
| CMA-ES | 1.89 | 12 | 12/28 SMO wins, p=0.55 n.s. |
| DualAnneal | 2.50 | 9 | 14/28 SMO wins, p=0.93 n.s. |
| **SMO-Pop** | **2.61** | 7 | — |
| BIPOP | 3.04 | 0 | 13/28 SMO wins, p=0.42 n.s. |
| DiffEvol | 4.96 | 0 | 28/28 wins, p<0.0001 ✅ |
| RandSearch | 6.00 | 0 | 28/28 wins, p<0.0001 ✅ |

Dim-30 is a 3-way tie at the top (CMA 1.89 / DA 2.50 / SMO 2.61, no
significant pairwise differences); BIPOP sinks to 4th with 0/28 wins —
again, restarts need budget headroom that 10k evals in dim 30 don't
give. Class split: DA owns hybrids (1.70) and unimodal (1.50 tie);
SMO is 2nd on multimodal (2.29) and composition (2.56). SMO's dim-30
tax is visible on smooth unimodal F1 (DA 0.03, CMA 144, SMO 4138):
subconscious overhead slows the sprint where plain CMA/LS converges —
evidence for the roadmap (diagonal-start covariance, surrogate
prescreening to *save* evals).

Net across both dims: **SMO is never significantly beaten by anyone
(reverse Wilcoxon p≥0.07 everywhere), significantly beats DE/RS
everywhere, DA at dim 10, and BIPOP at dim 10**, and ties plain
CMA-ES both dims. The custom-suite story (best rank, DA ahead on
mean) holds up on tougher rotated ground — with the failure modes
now precisely located (dim-30 smooth-unimodal overhead, F28/F29-class
compositions).

## v5-track: rank-based surrogate (`FastRankSurrogate`) — neutral result

Motivation: the v4 autopsy + Loshchilov et al. (2010) prescribe
rank-based ("comparison-based") surrogates over value-regression ones,
which smooth barrier ridges into fake valleys. Implemented as
`FastRankSurrogate` in `smo_upgraded.py`: pairwise-logistic
(RankNet-style) utility model on random-Fourier features, trained by
SGD on ~1500 sampled memory pairs per generation, plus a
Matérn-UCB-style novelty bonus (same `beta` knob, same interface).
Opt-in via `SMOPop(..., rank_surrogate=True)` / `--rank-surrogate`;
default OFF is **bit-identical v3** (verified: exact seed-1 matches).
Also pluggable into `SMOGhost` (forwards `**kwargs`, zero changes).

Surrogate-level check (synthetic 8-D bowl + sharp barrier ridge, 400
memory / 300 test points): ranking quality (Kendall tau, higher better)
**0.33 rank vs 0.20 Matérn** — the ranker is genuinely better at
ordering barrier landscapes.

Optimizer-level result — neutral in all three architectures:
1. **v3 core suite** (`results_rank_dim10_5k.txt`): 23/25 runs
   bit-identical to v3 (sphere/ackley/griewank exact; rastrigin 50.09%
   exact; rosenbrock 96.59 vs 96.64).
2. **v3 CEC dim-10** (140 SMO-Rank runs appended to `cec_dim10.jsonl`;
   `cec_dim10.txt` stays the 6-method baseline snapshot): head-to-head
   12 rank-wins / 8 pop-wins / 8 exact ties, Wilcoxon n.s. both ways
   (p=0.40/0.60).
3. **Ghost dreams+conflicts scored by rank** (5-seed): rastrigin 55.21
   (vs 55.72 ghost+Matérn), rosenbrock 82.47 (vs 83.76) — identical.

Mechanism (instrumented): on rastrigin seed-1 the two surrogates agree
on the top pick in **0/210 generations**, yet trajectories stay
identical — the picked candidates never win/improve, so the
credit-gated stream (sub_frac ~0.03–0.07) never feeds back into CMA.
v3's bottleneck is NOT surrogate ranking quality; it is that the gate
rationally starves a rarely-decisive stream. Rank candidates do win
slightly more credit (F15 sub_frac 0.073 vs 0.027) without moving
finals. Cost: 2.7× wall-clock (736s vs 271s on CEC dim-10) for zero
accuracy gain — not worth it in this architecture. Kept as verified
infrastructure for a future v5 in which the subconscious stream earns
real allocation.

## Experiment 1: forced subconscious quantity — negative (diagnostic)

Question: is the stream starved (needs MORE turns) or weak (more turns
= more waste)? Implemented `--sub-boost N` (N guaranteed extra
subconscious evals/generation; 0 = bit-identical v3). Rastrigin seed-1:
sub_frac 0.028 → 0.107 → 0.215 for N = 0/2/5 — the lever works.
Core suite, dim-10/5k/seeds 1–5 (`results_subboost_dim10_5k.txt`):

| function | v3 (N=0) | N=2 | N=5 |
|---|---|---|---|
| sphere | 100.00% (1.0e-27) | 100.00% (2.4e-25) | 100.00% (8.2e-20) |
| rastrigin | 50.09% | 47.40% | 29.65% |
| rosenbrock | 96.64% | 75.88% | 47.22% |
| ackley | 100.00% | 100.00% | 100.00% |
| griewank | 99.27% | 99.27% | 99.27% |

Monotone dose-response damage: rosenbrock −21pp/−49pp, rastrigin
−3pp/−20pp, precision depth decayed on sphere/ackley. Forced quantity
is pure budget theft from CMA — the credit gate's stinginess is
*optimal*, not a bug. Conclusion: the stream is QUALITY-capped, not
quantity-capped. v5 must go through prescreening (save-evals, exp-2)
and/or braver-pool + rank-filter (exp-3), never raw allocation.

## BBOB via IOHexperimenter (`bbob_benchmark.py`, `bbob_dim10.txt`, `bbob_dim20.txt`)

The venue-standard suite: 24 noiseless BBOB functions (f1–f24),
bounds [−5, 5]^D, 6 methods incl. a pycma **BIPOP** baseline
(`run_bipop` in `compare_baselines.py`). Dim 10: 10k evals, instances
1–5 (720 runs); dim 20: 20k evals, instances 1–3 (432 runs). Both
fixed-budget (mean precision, ranks, Wilcoxon paired by function) and
fixed-target analysis (fraction of run×target pairs solved over 51
COCO targets 1e2–1e-8). Raw records: `bbob_dim10.jsonl`,
`bbob_dim20.jsonl`.

Dim 10 — fixed-budget rank / fixed-target solved:

| method | mean rank | #best/24 | targets solved | Wilcoxon vs SMO |
|--------|-----------|----------|----------------|-----------------|
| BIPOP | 2.08 | 5 | 59.8% | SMO wins 6/24, n.s. |
| CMA-ES | 2.54 | 9 | 55.6% | SMO wins 12/24, n.s. |
| **SMO-Pop** | **2.71** | 2 | **50.4%** | — |
| DualAnneal | 3.08 | 8 | 40.8% | SMO wins 14/24, n.s. |
| DiffEvol | 4.58 | 0 | 15.2% | ✅ SMO, p<0.0001 |
| RandSearch | 6.00 | 0 | 4.9% | ✅ SMO, p<0.0001 |

Dim 20 — same format:

| method | mean rank | #best/24 | targets solved | Wilcoxon vs SMO |
|--------|-----------|----------|----------------|-----------------|
| BIPOP | 2.33 | 4 | 45.8% | SMO wins 10/24, n.s. |
| CMA-ES | 2.38 | 8 | 44.5% | SMO wins 13/24, n.s. |
| **SMO-Pop** | **2.54** | 5 | **40.2%** | — |
| DualAnneal | 3.08 | 7 | 32.2% | SMO wins 12/24, n.s. |
| DiffEvol | 4.79 | 0 | 7.5% | ✅ SMO, p<0.0001 |
| RandSearch | 5.88 | 0 | 2.8% | ✅ SMO, p<0.0001 |

Honest read: on BBOB's structured landscapes BIPOP is the best method
(rank ~2.1–2.3, most targets solved) — restarts pay off here, the
mirror image of CEC2017 where they burned budget. SMO ranks 3rd both
dims but **inside a 4-way statistical tie** (no significant difference
vs BIPOP/CMA/DA in either direction; reverse p≥0.13), significantly
ahead of DE/RS. Group detail: SMO is 2nd on structured multimodal
f15–19 (rank 1.80 dim-10; wins f16+f19 outright) and **1st on
separable f1–5 at dim 20 (rank 1.80)** — but pays a precision-depth tax
on smooth/conditioned functions (f10: CMA 1.6e-13 vs SMO 5.6e-05 at
dim 10; f10/f11 similar at dim 20): evals spent on the subconscious
stream and hopping are evals plain CMA spends converging to 1e-14.
Dual Annealing sprints early (best ECDF@10% budget both dims) then
stalls on conditioning. New roadmap item from this suite: precision
refinement (cheaper convergence to 1e-12 once the right basin is found).

Cross-suite net (custom + CEC2017 + BBOB): SMO is top-3 by rank on all
6 tables, #1 twice, and **never significantly beaten by any method on
any suite** — the consistency story holds at venue-standard level. A
performance-superiority claim over BIPOP/CMA is NOT supported (and not
claimed); the novelty claim rests on the mechanisms (M1–M3,
`PRIOR_ART.md`), for which SMO is now shown competitive with the
restart state of the art.

## Ablation study (`ablation.py`, `ablation_results.txt`)

Each v3 component removed in isolation; dim=10, 5k evals, seeds 1–5.
Mean accuracy %, and delta vs full in percentage points:

| config    | sphere | rastrigin | rosenbr. | schwefel | ellipsoid | MEAN |
|-----------|--------|-----------|----------|----------|-----------|------|
| full      | 100.00 | 50.09     | 96.64    | 0.15     | 100.00    | 69.37|
| no-active | 100.00 | 35.11     | 78.80    | 0.16     | 94.30     | 61.67|
| no-sub    | 100.00 | 48.70     | 67.54    | 0.16     | 100.00    | 63.28|
| no-credit | 100.00 | 30.10     | 69.61    | 0.15     | 100.00    | 59.97|
| no-restart| 100.00 | 8.72      | 96.64    | 0.06     | 100.00    | 61.08|
| no-LS     | 100.00 | 8.69      | 96.64    | 0.06     | 100.00    | 61.08|
| no-phase  | 100.00 | 8.52      | 96.64    | 0.06     | 100.00    | 61.04|
| cma-only  | 100.00 | 6.66      | 67.54    | 0.07     | 100.00    | 54.85|

Deltas vs full (pp): rastrigin is carried by the memetic escape system
(no-restart/no-LS/no-phase each −41pp; coupled by design) plus the
credit gate (−20pp) and active update (−15pp). Rosenbrock is carried by
the subconscious stream (−29pp) + credit gate (−27pp) + active update
(−18pp). Ellipsoid needs the active update (−5.7pp). Schwefel defeats
every configuration equally (see frontier note above) — deltas are
noise. Net: full v3 beats pure active-CMA by +14.5pp mean
(69.37 vs 54.85).

## SMO-Ghost v4: ideas #8–#14 (`smo_ghost.py`) — a documented negative result

Ideas #8 Ghost Landscape, #9 Conflict Search, #10 Latent Vector Memory,
#11 Salience/Surprise, #12 Adaptive Memory Decay, #13 Dreaming, and
#14 Uncertainty-driven exploration, implemented in a **separate file**
(`smo_ghost.py`) behind 4 no-op hooks in `smo_pop.py`. Ghost-OFF matches
v3 **bit-for-bit** (sphere s1 7.147898581204078e-28, rastrigin s1
1.989918114186608 — exact), so the harness is verified and all deltas
below are real.

Ghost-ON hardware: a sliding 512-ghost memory with salience + surprise
scoring and adaptive half-life decay, scoring-bonus/uncertainty/noise
gating on the v3 streams, plus up to 96 extra evals/gen of mask-crossover
dreams and extrapolated conflict probes (blend-dreams were tried first
and failed worse — latent midpoints decode to barrier ridges the
smoothing surrogate over-scores).

Result: the extras are **net negative**, verifiably, on the core suite
(dim=10, 5k evals; full log `comparison_ghost_dim10.txt`):

| function   | SMO-Pop (v3) | SMO-Ghost | Δ ghost−v3 |
|------------|--------------|-----------|------------|
| sphere     | 100.00%      | 100.00%   | 0.00pp     |
| rastrigin  | 50.09%       | 55.72%    | +5.63pp ⚠  |
| rosenbrock | 96.64%       | 83.76%    | −12.88pp   |
| ackley     | 100.00%      | 100.00%   | 0.00pp     |
| griewank   | 99.27%       | 98.83%    | −0.44pp    |
| **MEAN RANK** | **1.40**  | **2.40**  | worse      |

⚠ The 5-seed rastrigin "+5.63pp" is noise: a decisive 12-seed rerun
(seeds 1–12) reverses it — v3 60.48±34.67 (mean loss 1.49) vs ghost
40.20±28.00 (mean loss 2.42), ghost worse on 8 of 12 seeds. Same story
on rosenbrock (12-seed: v3 97.75±3.69 vs ghost 79.56±34.38, mean loss
0.0245 vs 1.0). A ghost-ON mini-ablation confirms the mechanism: with
dream/conflict evals disabled the scoring bonus + adaptive noise are
byte-inert (config matches v3 exactly), so **all** of the damage rides
on the speculative evals — the smoothing SurrogateGP can't rank
boundary-crossing candidates on multimodal landscapes and early ghost
evaluations poison basin selection before v3's escape system engages.
Schwefel spot check: 0.21% vs 0.16%, both total failures as before.

Kept in the repo as infrastructure (hooks + verified harness + negative
evidence) for a future v5 that would need a non-smoothing ranker
(e.g. landscape-aware surrogate) before speculative evals can help.
**v3 stays the recommended optimizer.**

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
