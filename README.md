# subconscious-search

**Bicameral: a gated conscious/subconscious black-box optimizer.**

Two versions:
- **v1** (`bicameral_v1.py`) — diagonal Sep-CMA-ES conscious step +
  supervised latent space + Matérn surrogate + annealed sigmoid gate.
  Single-incumbent, one eval per iteration.
- **v2/v3** (`bicameral.py`, current best) — **full-covariance Active-CMA-ES**
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
python src/bicameral.py --dim 10 --max-evals 5000 --n-runs 5        # v3 (best)
python src/bicameral_v1.py --dim 10 --max-evals 5000 --n-runs 5   # v1
# comparison vs baselines (needs scipy + cma):
pip install cma
python src/compare_baselines.py --dim 10 --max-evals 5000 --n-runs 5 --seed0 1
```

Layout: all code lives under `src/` (optimizers + benchmark harnesses,
plus `src/diabetes/` for the diabetes-project experiments 21-29);
`results/{core,cec,bbob,esdrp,ml,diabetes}/` hold raw logs (`.jsonl`/`.log`)
and summary snapshots (`.txt`/`.csv`); `docs/` holds notes and reference
PDFs; `data/` (CSVs versioned, rest gitignored) holds datasets.

## Results (dim=10, 5 runs, 5000 evals)

Accuracy = `100 / (1 + best_loss)` (100% at the global optimum).
Full per-run logs: `results/core/results_popv3_dim10_5k.txt` (v3),
`results/core/results_pop_dim10_5k.txt` (v2), `results/core/results_dim10_5k.txt` (v1).

| function   | v3 acc  | v3 loss | v2 acc | v1 acc  |
|------------|---------|---------|--------|---------|
| sphere     | 100.00% | 1.0e-27 | 100.00%| 100.00% |
| rastrigin  | 50.09%  | 1.79    | 10.01% | 7.95%   |
| rosenbrock | 96.64%  | 0.037   | 96.64% | 21.14%  |
| ackley     | 100.00% | 4.4e-14 | 100.00%| 89.14%  |
| griewank   | 99.27%  | 0.0074  | 99.03% | 91.41%  |
| **MEAN**   | **89.20%** |      | **81.14%** | **61.93%** |

## Comparison vs established optimizers (dim=10, 5000 evals, seeds 1–5)

Same eval budget for every method. Full log: `results/core/comparison_results_v3.txt`.

| function   | Bicameral (v3) | Bicameral-v1 | CMA-ES | DiffEvol | DualAnneal | RandSearch |
|------------|--------------|--------|--------|----------|------------|------------|
| sphere     | 100.00%      | 100.00%| 100.00%| 92.26%   | 100.00%    | 6.02%      |
| rastrigin  | 50.09%       | 6.93%  | 7.51%  | 2.27%    | 76.71%     | 1.33%      |
| rosenbrock | 96.64%       | 20.96% | 83.62% | 2.88%    | 84.01%     | 0.04%      |
| ackley     | 100.00%      | 75.18% | 100.00%| 38.40%   | 100.00%    | 10.82%     |
| griewank   | 99.27%       | 91.16% | 99.04% | 66.72%   | 95.60%     | 66.22%     |
| **MEAN**   | **89.20%**   | **58.85%** | **78.03%** | **40.51%** | **91.26%** | **16.89%** |

Takeaways: Bicameral v3 is best-in-class on rosenbrock (96.64%, beating
both CMA-ES and Dual Annealing) and griewank, tied on sphere/ackley, and
within 2 points of Dual Annealing overall — trailing only on rastrigin,
where annealing specialists still lead but the gap closed from 69 to 27
points (7x accuracy gain over v2).

## Extended suite: 7 more functions, dim 10 (`results/core/comparison_extra_dim10.txt`)

schwefel / levy / michalewicz / styblinski / ellipsoid / zakharov /
noisy_sphere — same 5k-eval budget, seeds 1–5.

| function   | Bicameral | CMA-ES | DiffEvol | DualAnneal | RandSearch |
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

Takeaways: Bicameral has the best mean rank (most consistent — top-2 on 6/7)
and beats CMA-ES on mean again, but Dual Annealing's mean is higher via
schwefel/styblinski blowouts: deceptive landscapes with smooth basins are
annealing+LS territory, and schwefel (optimum near the corner, no global
trend) defeats the whole CMA family. Bicameral beats DA on ellipsoid
(covariance learning > numeric gradients at cond 1e6) and noisy_sphere
(LS chases noise; ES averages it out).

## Stress test: all 11 functions, dim 30 (`results/core/comparison_all_dim30.txt`)

Same 5k-eval budget (starvation rations in 30-D), 3 runs, seeds 1–3
(michalewicz skipped: no dim-30 reference optimum).

| function   | Bicameral | CMA-ES | DiffEvol | DualAnneal | RandSearch |
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
last two). Bicameral is a solid 2nd by mean and rank, beats CMA-ES on 9/11,
wins levy outright (88.5 vs 63.2/26.6), and owns the robustness corner:
ackley (DA's LS drowns in ripples, 19%) and noisy_sphere (DA chases
noise, 1.3%). Regimes are now clear: DA = smooth/multimodal + high-D
valleys; Bicameral = ill-conditioning, sharp basins, noise, rippled landscapes.

## Tougher benchmarks: CEC2017 (`cec_benchmark.py`, `results/cec/cec_dim10.txt`, `results/cec/cec_dim30.txt`)

28 functions via `opfunu` (F1, F3–F29; F2 officially excluded, F30
unimplemented) — shifted, rotated, hybrid, and composition landscapes,
bounds [−100, 100]^D. Fixed budget of 10k evals for all methods
(disclosed: official CEC protocol is 10000×D; this is a fixed-budget
comparison, 5 seeds dim-10 / 3 seeds dim-30). Metric is mean error to
the known optimum; ranks are per-function (1 = best); significance is
Wilcoxon signed-rank paired by function. Raw per-run records:
`results/cec/cec_dim10.jsonl` (840 runs), `results/cec/cec_dim30.jsonl` (504 runs).

Dim 10 (mean error rank, #functions won, Wilcoxon vs Bicameral):

| method | mean rank | #best/28 | Wilcoxon vs Bicameral |
|--------|-----------|----------|-----------------|
| **Bicameral** | **2.00** | **10** | — |
| CMA-ES | 2.39 | 10 | 16/28 wins, p=0.29 n.s. |
| DualAnneal | 3.36 | 4 | 23/28 wins, p=0.0003 ✅ |
| BIPOP | 3.50 | 2 | 20/28 wins, p=0.020 ✅ |
| DiffEvol | 3.79 | 2 | 25/28 wins, p<0.0001 ✅ |
| RandSearch | 5.96 | 0 | 28/28 wins, p<0.0001 ✅ |

Dim-10 class ranks: Bicameral is best-or-tied in every class (unimodal 1.50
tie, multimodal 2.14, hybrid 2.30, composition 1.67). Dual Annealing —
the overall leader on the custom suite — collapses to rank 3.36:
rotation + shifts + hybrids defeat its local search, exactly as
predicted. Notably Bicameral also beats BIPOP significantly (p=0.020):
at a tight 10k budget on hard rotated/hybrid landscapes, BIPOP's
restart overhead burns evals without time to pay off, while Bicameral's
stall-gated design (no restart until stagnation is proven) wins.
Bicameral's highlights: F1 bent cigar to 3.6e-06, best-on-10.
Residual weak spots: F28/F29 compositions (CMA 1640/4746 vs Bicameral
1873/14900) and F10/F18 (plain CMA wins big).

Dim 30 (same format):

| method | mean rank | #best/28 | Wilcoxon vs Bicameral |
|--------|-----------|----------|-----------------|
| CMA-ES | 1.89 | 12 | 12/28 Bicameral wins, p=0.55 n.s. |
| DualAnneal | 2.50 | 9 | 14/28 Bicameral wins, p=0.93 n.s. |
| **Bicameral** | **2.61** | 7 | — |
| BIPOP | 3.04 | 0 | 13/28 Bicameral wins, p=0.42 n.s. |
| DiffEvol | 4.96 | 0 | 28/28 wins, p<0.0001 ✅ |
| RandSearch | 6.00 | 0 | 28/28 wins, p<0.0001 ✅ |

Dim-30 is a 3-way tie at the top (CMA 1.89 / DA 2.50 / Bicameral 2.61, no
significant pairwise differences); BIPOP sinks to 4th with 0/28 wins —
again, restarts need budget headroom that 10k evals in dim 30 don't
give. Class split: DA owns hybrids (1.70) and unimodal (1.50 tie);
Bicameral is 2nd on multimodal (2.29) and composition (2.56). Bicameral's dim-30
tax is visible on smooth unimodal F1 (DA 0.03, CMA 144, Bicameral 4138):
subconscious overhead slows the sprint where plain CMA/LS converges —
evidence for the roadmap (diagonal-start covariance, surrogate
prescreening to *save* evals).

Net across both dims: **Bicameral is never significantly beaten by anyone
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
`FastRankSurrogate` in `bicameral_v1.py`: pairwise-logistic
(RankNet-style) utility model on random-Fourier features, trained by
SGD on ~1500 sampled memory pairs per generation, plus a
Matérn-UCB-style novelty bonus (same `beta` knob, same interface).
Opt-in via `Bicameral(..., rank_surrogate=True)` / `--rank-surrogate`;
default OFF is **bit-identical v3** (verified: exact seed-1 matches).
Also pluggable into `Reverie` (forwards `**kwargs`, zero changes).

Surrogate-level check (synthetic 8-D bowl + sharp barrier ridge, 400
memory / 300 test points): ranking quality (Kendall tau, higher better)
**0.33 rank vs 0.20 Matérn** — the ranker is genuinely better at
ordering barrier landscapes.

Optimizer-level result — neutral in all three architectures:
1. **v3 core suite** (`results/core/results_rank_dim10_5k.txt`): 23/25 runs
   bit-identical to v3 (sphere/ackley/griewank exact; rastrigin 50.09%
   exact; rosenbrock 96.59 vs 96.64).
2. **v3 CEC dim-10** (140 Bicameral-Rank runs appended to `results/cec/cec_dim10.jsonl`;
   `results/cec/cec_dim10.txt` stays the 6-method baseline snapshot): head-to-head
   12 rank-wins / 8 pop-wins / 8 exact ties, Wilcoxon n.s. both ways
   (p=0.40/0.60).
3. **Reverie dreams+conflicts scored by rank** (5-seed): rastrigin 55.21
   (vs 55.72 reverie+Matérn), rosenbrock 82.47 (vs 83.76) — identical.

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
Core suite, dim-10/5k/seeds 1–5 (`results/core/results_subboost_dim10_5k.txt`):

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

## Experiment 2: surrogate prescreening of CMA offspring — negative

Design: sample 3×/5× CMA offspring per generation, surrogate-score in
latent space, truly evaluate only the best λ (`--prescreen-mult N`,
1 = bit-identical v3, verified). Per-gen eval cost unchanged — only
offspring quality changes. 5-seed screen (`results/core/results_prescreen_dim10_5k.txt`)
looked mixed-promising: rosenbrock 96.64 → 99.58 → 99.99 (loss ×370!),
but rastrigin 50.09 → 24.61 → 34.92 and precision depth damaged
elsewhere (sphere 1e-27 → 0.003 → 2e-14; griewank monotone-worse).

Decisive 12-seed test (seeds 1–12, + prescreen×rank-surrogate arm):

| function | v3 | pre×5 | pre×5+rank |
|---|---|---|---|
| rosenbrock acc | 97.75±3.69 | 96.53±10.29 | 73.78±30.72 |
| rastrigin acc | 60.48±34.67 | 31.63±12.00 | 27.95±12.65 |

The 5-seed rosenbrock "win" was noise — pre×5 is neutral-to-worse with
3× worse reliability, and the rank combo collapses (73.78). On
rastrigin prescreening halves accuracy while *collapsing variance*
(34.7 → 12.0): reliable mediocrity — the filter systematically removes
the lucky basin jumps behind v3's high mean. Mechanism: prescreening is
a greed filter. It deletes the offspring that look bad to a
history-trained model — precisely the exploratory steps a population
needs (basin jumps; valley-wall probes that train the covariance; bad
offspring the active update learns from). CMA's update already does
optimal selection-from-randomness; pre-filtering the randomness
impoverishes it. Second lesson in a row: **the 5-seed core suite is a
screen, the 12-seed is the verdict** (same reversal as reverie v4).
Exp-2 closed as negative.

## Experiment 3: braver subconscious search (`--brave-mult`) — negative

Design: ONE knob — pool size ×N *and* local width ×N (`--brave-mult N`,
free: same top-k true evals/gen), crossed with rank {off, on} in a 2×2
screen on the core suite (dim10/5k/seeds1–5). brave=1 is bit-identical
v3 (verified: sphere/rastrigin seed-1 anchors exact); activation
verified (pool 256 → 1024, spread 1.108 → 2.602). Pre-registered
hypothesis: a bolder pool raises stream quality → the gate funds it
(sub_frac up organically) → rastrigin/rosenbrock improve, others
unharmed. Verdict rule: any arm >5pp mean-acc over v3 with no function
collapsing >5pp → 12-seed verdict on rastrigin+rosenbrock.
(`results/core/results_brave_dim10_5k.txt`)

5-seed screen looked mixed: rastrigin collapsed (50.09 → 34.27),
rosenbrock blipped +2.3 (96.64 → 98.96), rank changed nothing. A
post-hoc brave=2 dose curve then broke the story: rastrigin went
50.09 → **24.26** → 34.27 — *non-monotone* in dose, i.e. basin-lottery
noise, not real harm — while rosenbrock stayed +2–3pp in all four
brave arms with collapsing variance (±0.32 at brave=2). Neither
signal could be trusted at n=5, so the verdict ran anyway (full
suite, 12 seeds, pure-bravery arms since rank proved inert):
`results/core/results_verdict_brave_dim10_5k.txt`.

| function (12 seeds) | v3 | brave×2 | brave×4 |
|---|---|---|---|
| rastrigin acc | 60.48±34.67 | 28.23±11.52 | 39.54±21.36 |
| rosenbrock acc | 97.75±3.69 | 79.90±34.57 | 92.26±21.88 |
| griewank / sphere / ackley | 99.03 / 100 / 100 | 98.95 / 100 / 100 | 99.37 / 100 / 100 |

The screen's rosenbrock "win" was noise — completely reversed across
seed blocks (seeds 1–5 favored brave, 6–12 crushed it: three
catastrophic acc≈20% runs stuck in rosenbrock's famous f=4 local-min
basin at (−1,1,…,1)). Pooled bravery harms rastrigin −26.6pp
(p=0.031; global hits 5/12 vs 0/12 at brave×2, Fisher p=0.037) and
replicates in fresh seeds 6–12 alone. Rosenbrock pooled −11.7pp is
marginal (p=0.075, failure-mode driven).

Mechanism, pinned by elimination: bravery perturbs trajectories two
ways — shared-RNG consumption (re-rolls all downstream CMA samples)
and adopted far-flung sub-wins. A re-roll is provably bias-free
(fresh i.i.d. draws either way), so *systematic* harm must ride the
second channel: brave early wins teleport best_x, and IPOP restarts
recenter the CMA mean there — random relocation instead of
systematic basin work. This predicts exactly what the verdict shows:
real harm where restarts fire (rastrigin, restarts 2/2 every run),
luck-only where they don't (rosenbrock failures all had
restarts=0 — no channel exists, matching the n.s. tests). The
stall/BH machinery is exonerated (bh_phase 12/12 in every arm; BH
and all polish steps are elitist-safe by construction). The gate, to
its credit, never funded the stream (sub_frac sat at the ~0.03 floor
in all arms — rastrigin brave runs were starved *harder*, 0.020).
Rank filter: inert even at 4× pool (≤0.9pp on rastrigin, 0.05pp on
rosenbrock at brave=1 only, bit-exact 0.00 on all other
function/arm cells) — the v5-track neutral replicates under
bravery. Third lesson in a row: **the 5-seed suite
is a screen, the 12-seed is the verdict**. Bravery rejected;
v3 unchanged. Exp-3 closed as negative.

## Paper-2 track: Gaussian-copula subconscious proposer — negative

Design: replace the latent-space pool with dependence-preserving
samples — a Gaussian copula fit on archive elites each generation
(empirical marginals + normal-scores correlation, Sklar's split),
same 25% global mixture, SAME surrogate scoring and credit gate, so
the comparison is purely proposer-vs-proposer (`--copula-pool`,
default off = bit-identical v3, anchors verified; dependence
check: elite off-diag corr 0.485 → sample corr 0.521).
Pre-registered hypothesis: copula ≥ latent on dependence-heavy
rosenbrock (valley-following samples), neutral-or-worse on separable
rastrigin; honest prior stated as LOW (~20%: the stream is
gate-starved, so the proposer rarely binds). Verdict rule: any
copula arm >5pp over v3 with no >5pp collapse → 12-seed verdict.
(`results/core/results_copula_dim10_5k.txt`)

5-seed screen: rastrigin 50.09 → 22.18 (−28pp, success 0/5, variance
*collapsed* ±27.9 → ±10.9 — the reliable-mediocrity signature),
rosenbrock +2.0 (96.64 → 98.65, tighter variance). The collapse
broke the verdict rule on paper, but the screen→verdict reversals of
reverie/exp-2/exp-3 demanded the 12-seed check anyway (v3 vs copula,
full suite): `results/core/results_verdict_copula_dim10_5k.txt`.

| function (12 seeds) | v3 | copula |
|---|---|---|
| rastrigin acc | 60.48±34.67 | 23.65±12.76 |
| rosenbrock acc | 97.75±3.69 | 91.60±21.86 |
| griewank / sphere / ackley | 99.03 / 100 / 100 | 98.87 / 100 / 100 |

Rastrigin harm **confirmed**: −36.8pp, p=0.0052, fresh-seeds-6–12
alone p=0.029, ZERO global hits in 12 runs vs v3's five (Fisher
p=0.037). Rosenbrock blip reversed again (−6.2pp n.s., one f=4-basin
failure) — fourth screen↔verdict rosenbrock reversal in a row.
Griewank/sphere/ackley flat.

Mechanism (same elimination as exp-3): RNG re-rolls are bias-free,
so systematic harm must ride adopted copula points → IPOP restart
recentering (restarts 2/2 every run) → early commitment to the
first good basin. Dependence-preserving sampling IS
basin-preserving sampling: elites concentrate in the current-best
basin and the copula faithfully reproduces it, while v3's diffuse
latent pool adopts more diverse points. The gate starved the stream
anyway (sub_frac at the 0.03 floor) — harm via redirection, not
budget theft; BH exonerated (12/12 everywhere, elitist-safe).
Rank+copula combo near-inert (rosenbrock/sphere bit-identical,
≤1.8pp noise wiggles elsewhere) — rank stays dead.

Scope note for Paper 2: this kills the *drop-in replacement*
(Gaussian copula, elites-only, same gate) — decisively. It does not
test heavy-tailed (t) copulas, vines, all-archive fits, or a copula
*third* stream alongside the latent pool. Any of those is a live
follow-up; the replacement premise is closed as negative.

## BBOB via IOHexperimenter (`bbob_benchmark.py`, `results/bbob/bbob_dim10.txt`, `results/bbob/bbob_dim20.txt`)

The venue-standard suite: 24 noiseless BBOB functions (f1–f24),
bounds [−5, 5]^D, 6 methods incl. a pycma **BIPOP** baseline
(`run_bipop` in `compare_baselines.py`). Dim 10: 10k evals, instances
1–5 (720 runs); dim 20: 20k evals, instances 1–3 (432 runs). Both
fixed-budget (mean precision, ranks, Wilcoxon paired by function) and
fixed-target analysis (fraction of run×target pairs solved over 51
COCO targets 1e2–1e-8). Raw records: `results/bbob/bbob_dim10.jsonl`,
`results/bbob/bbob_dim20.jsonl`.

Dim 10 — fixed-budget rank / fixed-target solved:

| method | mean rank | #best/24 | targets solved | Wilcoxon vs Bicameral |
|--------|-----------|----------|----------------|-----------------|
| BIPOP | 2.08 | 5 | 59.8% | Bicameral wins 6/24, n.s. |
| CMA-ES | 2.54 | 9 | 55.6% | Bicameral wins 12/24, n.s. |
| **Bicameral** | **2.71** | 2 | **50.4%** | — |
| DualAnneal | 3.08 | 8 | 40.8% | Bicameral wins 14/24, n.s. |
| DiffEvol | 4.58 | 0 | 15.2% | ✅ Bicameral, p<0.0001 |
| RandSearch | 6.00 | 0 | 4.9% | ✅ Bicameral, p<0.0001 |

Dim 20 — same format:

| method | mean rank | #best/24 | targets solved | Wilcoxon vs Bicameral |
|--------|-----------|----------|----------------|-----------------|
| BIPOP | 2.33 | 4 | 45.8% | Bicameral wins 10/24, n.s. |
| CMA-ES | 2.38 | 8 | 44.5% | Bicameral wins 13/24, n.s. |
| **Bicameral** | **2.54** | 5 | **40.2%** | — |
| DualAnneal | 3.08 | 7 | 32.2% | Bicameral wins 12/24, n.s. |
| DiffEvol | 4.79 | 0 | 7.5% | ✅ Bicameral, p<0.0001 |
| RandSearch | 5.88 | 0 | 2.8% | ✅ Bicameral, p<0.0001 |

Honest read: on BBOB's structured landscapes BIPOP is the best method
(rank ~2.1–2.3, most targets solved) — restarts pay off here, the
mirror image of CEC2017 where they burned budget. Bicameral ranks 3rd both
dims but **inside a 4-way statistical tie** (no significant difference
vs BIPOP/CMA/DA in either direction; reverse p≥0.13), significantly
ahead of DE/RS. Group detail: Bicameral is 2nd on structured multimodal
f15–19 (rank 1.80 dim-10; wins f16+f19 outright) and **1st on
separable f1–5 at dim 20 (rank 1.80)** — but pays a precision-depth tax
on smooth/conditioned functions (f10: CMA 1.6e-13 vs Bicameral 5.6e-05 at
dim 10; f10/f11 similar at dim 20): evals spent on the subconscious
stream and hopping are evals plain CMA spends converging to 1e-14.
Dual Annealing sprints early (best ECDF@10% budget both dims) then
stalls on conditioning. New roadmap item from this suite: precision
refinement (cheaper convergence to 1e-12 once the right basin is found).

Cross-suite net (custom + CEC2017 + BBOB): Bicameral is top-3 by rank on all
6 tables, #1 twice, and **never significantly beaten by any method on
any suite** — the consistency story holds at venue-standard level. A
performance-superiority claim over BIPOP/CMA is NOT supported (and not
claimed); the novelty claim rests on the mechanisms (M1–M3,
`PRIOR_ART.md`), for which Bicameral is now shown competitive with the
restart state of the art.

### Zoo panel: v3 vs classic nature-inspired optimizers (BBOB dim-10)

Pre-registered prediction: Bicameral's CMA core beats the 90s–00s zoo on
conditioned landscapes; would NOT extend to modern L-SHADE-class DE.
Tested via `mealpy` (`zoo_methods.py`): PSO, GA, GWO, WOA, ABC, plus
SHADE as a modern-DE reference, plus FOX/HBA/TSO (the three optimizers
from the ESDRP diabetes paper, see next section) — 1080 runs appended
to `results/bbob/bbob_dim10.jsonl` (now 1800 rows; `results/bbob/bbob_dim10.txt` stays the
6-method snapshot, full 15-method log in `results/bbob/bbob_zoo_dim10.txt`). Budget
fairness: nominal epoch×pop≈10k plus a hard cap wrapper asserting
**exactly** 10000 true evals per run (post-budget calls return
worst-seen, no new signal). Same IOH dim-10 protocol, instances 1–5.

| method | mean rank | #best/24 | targets | Wilcoxon vs Bicameral (Bicameral wins) |
|--------|-----------|----------|---------|----------------------------|
| BIPOP | 2.88 | 5 | 59.8% | 6/24, n.s. |
| **Bicameral** | **3.67** | 1 | **50.4%** | — |
| CMA-ES | 3.75 | 9 | 55.6% | 12/24, n.s. |
| DualAnneal | 4.88 | 6 | 40.8% | 14/24, n.s. |
| SHADE | 5.71 | 1 | 23.1% | ✅ 20/24, p=0.0005 |
| PSO | 6.79 | 0 | 23.7% | ✅ 20/24, p<0.0001 |
| GWO | 7.29 | 0 | 17.6% | ✅ 21/24, p<0.0001 |
| HBA | 7.71 | 0 | 18.1% | ✅ 21/24, p<0.0001 |
| DiffEvol | 8.04 | 0 | 15.2% | ✅ 23/24, p<0.0001 |
| TSO | 8.92 | 1 | 16.1% | ✅ 20/24, p<0.0001 |
| GA | 9.21 | 1 | 12.8% | ✅ 22/24, p<0.0001 |
| WOA | 11.42 | 0 | 13.0% | ✅ 23/24, p<0.0001 |
| ABC | 11.50 | 0 | 12.0% | ✅ 23/24, p<0.0001 |
| RandSearch | 13.96 | 0 | 4.9% | ✅ 24/24, p<0.0001 |
| FOX | 14.29 | 0 | 4.6% | ✅ 23/24, p<0.0001 |

Prediction confirmed — and then some: Bicameral significantly beats all nine
zoo methods, **including the modern SHADE reference** (20/24,
p=0.0005), and solves 2× the fixed-target pairs of the best zoo member
(50.4% vs 23.7% PSO). Group detail: the zoo collapses hardest on
conditioned functions (high-cond ranks: SHADE 5.8, PSO 6.0, HBA 7.6,
GWO 8.0, TSO 10.8, rest 11+), exactly where covariance learning pays.
Striking: **FOX ranks below random search** (14.29 vs 13.96, 4.6% vs
4.9% targets) — verified functional (beats RS head-to-head on f1/f9),
just a weak optimizer under defaults; it also loses to RS on the
ESDRP wrapper below, so the weakness replicates on two grounds.
Disclosed caveats: (1) `mealpy` implementations (incl. a stub
`OriginalGA` — used functional `BaseGA` instead), not authors' code;
(2) default pop=100 for all zoo methods — different tunings could shift
zoo-vs-zoo order, but every run consumed exactly 10000 evals;
(3) ABC evaluates ~2×/epoch so the cap truncates it mid-schedule
(budget fairness by design); (4) SHADE is designed for 10000×D-scale
budgets and is underpowered at fixed 10k — this is a fixed-budget
comparison, not a SHADE obituary; (5) dim-10 BBOB only. Earned claim,
scoped: *v3 significantly outperforms 9 representative nature-inspired
optimizers (PSO, GA, GWO, WOA, ABC, SHADE, FOX, HBA, TSO) on BBOB
dim-10 at a fixed 10k budget* — the first positive superiority result
in the repo beyond vanilla DE/RS.

## ESDRP wrapper: v3 beats the paper's swarm optimizers on their own problem

Sarker et al. (Sci Rep 2026, PDF in `docs/papers/`) use FOX/HBA/TSO as wrapper
optimizers for RF feature-selection + hyperparameter tuning on the
ESDRP diabetes dataset (UCI-529, 520×16). We replicate their problem
exactly — same 20-D space (4 RF hparams + 16-bit mask), same
train-F1 fitness, same sliding-window 70/30×10 folds, same
majority-vote pipeline — and swap in Bicameral v3 as the optimizer, plus
the random-search baseline the paper lacks (`esdrp_wrapper.py`,
`results/esdrp/esdrp_arm1_*.jsonl`, `results/esdrp/esdrp_results.txt`; data in `data/esdrp.csv`,
verified against their Table 2). Every run consumes exactly 1000
evals/fold (asserted; v3 uses ~6 under by generational granularity),
seeds paired per fold across methods.

Optimizer capability (best train-F1 found, 1000 evals/fold):

| method | mean train-F1 | perfect 1.0 folds | median evals-to-perfect | clean test-F1 |
|--------|---------------|-------------------|-------------------------|---------------|
| **Bicameral** | **1.0000** | **10/10** | **104** | 0.9695 |
| TSO | 0.9989 | 9/10 | 163 | 0.9683 |
| HBA | 0.9989 | 7/10 | 101 | 0.9722 |
| RandSearch | 0.9921 | 0/10 | — | 0.9609 |
| FOX | 0.9677 | 0/10 | — | 0.9452 |

Voted-model replication (their leaky pipeline) vs paper-reported:

| method | ours (acc / feats) | paper (acc / feats) |
|--------|--------------------|---------------------|
| Bicameral | 97.18 / 12 | — |
| HBA | 97.69 / 16 | 97.24 / 10 |
| TSO | 97.69 / 15 | 98.14 / 14 |
| RandSearch | 96.73 / 9 | — |
| FOX | 95.26 / 15 | 98.01 / 13 |

Findings: (1) **v3 is the most reliable wrapper** — only method
perfect on all 10 folds; (2) **FOX loses to random search** on both
train-F1 (0.9677 vs 0.9921) and clean test (Wilcoxon Bicameral>FOX p=0.008;
v3's BBOB result replicates on real data); (3) our voted hparams
converge to (300,10,2,1) — the paper's (10,2,1) corner — independently
reproducing their "memorize with deep unpruned trees" finding, with
`n_estimators` confirmed irrelevant; (4) **feature reduction is
shuffle-unstable**: our HBA keeps all 16 features (vs their 10) under
an identical pipeline with a different shuffle — the "10 features"
claim doesn't survive a seed change; (5) honest tie: on *clean*
per-fold test-F1 all memorizers are statistically tied (~0.97,
Wilcoxon n.s. at n=10 except vs FOX) — the separation is in optimizer
capability/speed, not generalization. Arm 2 spot-check at their full
5000-eval budget (fold 1): both v3 and TSO perfect, v3 12× faster to
perfect (89 vs 1118 evals); v3's trajectory deterministically
reproduces Arm 1. Disclosed deviations: fixed shuffle/RF seeds
(paper: none), deterministic fitness (paper: stochastic default),
1000-eval budget (their plots show saturation by ~350; Arm 2 confirms
at 5000). Footnote: the first two Arm-1 runs allowed depth-11 at
exact-bound hits; affected folds (Bicameral f8, HBA f7) were rerun
in-spec and still perfected — final numbers fully in-spec.

### Task 2: clean protocol — inner-validation fitness + nested evaluation

Arm 1 replicated the paper's train-F1 fitness, warts included: it
plateaus at 1.0 (26/50 runs), so most runs tie and first-hit
tie-breaking decides — memorization scores, search quality barely
matters. Task 2 re-runs all 5 methods with ONE change: fitness = F1 on
a stratified 75/25 inner split of the train fold (fixed seeds), so
memorization can't score; fold-bests are retrained on the full train
fold and tested on the untouched outer test = proper nested CV
(`--fitness innerval`, `results/esdrp/esdrp_clean_*.jsonl`,
`results/esdrp/esdrp_clean_results.txt`). No voted model — that
pipeline leaks by construction. Pre-registered prediction: the
plateau breaks and search quality separates the methods.

| method | fit-F1 | perfect/10 | clean outer-F1 | feats | Wilcoxon vs Bicameral (outer / fit) |
|--------|--------|------------|----------------|-------|-------------------------------|
| **Bicameral** | **0.9902** | **3** | **0.9596** | 9.8 | — |
| TSO | 0.9805 | 1 | 0.9401 | 9.0 | ✅ p=0.037 / p=0.016 |
| HBA | 0.9757 | 1 | 0.9346 | 9.2 | ✅ p=0.010 / p=0.008 |
| RandSearch | 0.9734 | 0 | 0.9426 | 9.4 | ✅ p=0.027 / p=0.002 |
| FOX | 0.9570 | 0 | 0.9282 | 9.0 | ✅ p=0.002 / p=0.002 |

Prediction confirmed decisively: with honest fitness Bicameral is
significantly better than **all four** on both inner-val fit and
nested outer test (8–10/10 fold-wins), where Arm 1's clean numbers
were all ties. When the fitness actually discriminates, v3's
surrogate-guided search converts to real generalization wins. Two
honest notes: (1) absolute outer numbers drop vs Arm 1 for everyone
(inner-val selection on 91 samples is noisier than train selection on
364 — the paired *ordering* is the finding, not the levels; neither
is comparable to the paper's leaky 98.14); (2) voted hparams still
show (depth,split,leaf)=(10,2,1) for Bicameral/TSO — deep trees generalize
fine here given the strong signal, so the separation comes from
better *masks*, and honest fitness selects leaner models across the
board (~9 feats vs Arm 1's 10–13).

## Ablation study (`ablation.py`, `results/core/ablation_results.txt`)

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

## Bicameral-Reverie v4: ideas #8–#14 (`reverie.py`) — a documented negative result

Ideas #8 Reverie Landscape, #9 Conflict Search, #10 Latent Vector Memory,
#11 Salience/Surprise, #12 Adaptive Memory Decay, #13 Dreaming, and
#14 Uncertainty-driven exploration, implemented in a **separate file**
(`reverie.py`) behind 4 no-op hooks in `bicameral.py`. Reverie-OFF matches
v3 **bit-for-bit** (sphere s1 7.147898581204078e-28, rastrigin s1
1.989918114186608 — exact), so the harness is verified and all deltas
below are real.

Reverie-ON hardware: a sliding 512-reverie memory with salience + surprise
scoring and adaptive half-life decay, scoring-bonus/uncertainty/noise
gating on the v3 streams, plus up to 96 extra evals/gen of mask-crossover
dreams and extrapolated conflict probes (blend-dreams were tried first
and failed worse — latent midpoints decode to barrier ridges the
smoothing surrogate over-scores).

Result: the extras are **net negative**, verifiably, on the core suite
(dim=10, 5k evals; full log `results/core/comparison_reverie_dim10.txt`):

| function   | Bicameral (v3) | Bicameral-Reverie | Δ reverie−v3 |
|------------|--------------|-----------|------------|
| sphere     | 100.00%      | 100.00%   | 0.00pp     |
| rastrigin  | 50.09%       | 55.72%    | +5.63pp ⚠  |
| rosenbrock | 96.64%       | 83.76%    | −12.88pp   |
| ackley     | 100.00%      | 100.00%   | 0.00pp     |
| griewank   | 99.27%       | 98.83%    | −0.44pp    |
| **MEAN RANK** | **1.40**  | **2.40**  | worse      |

⚠ The 5-seed rastrigin "+5.63pp" is noise: a decisive 12-seed rerun
(seeds 1–12) reverses it — v3 60.48±34.67 (mean loss 1.49) vs reverie
40.20±28.00 (mean loss 2.42), reverie worse on 8 of 12 seeds. Same story
on rosenbrock (12-seed: v3 97.75±3.69 vs reverie 79.56±34.38, mean loss
0.0245 vs 1.0). A reverie-ON mini-ablation confirms the mechanism: with
dream/conflict evals disabled the scoring bonus + adaptive noise are
byte-inert (config matches v3 exactly), so **all** of the damage rides
on the speculative evals — the smoothing SurrogateGP can't rank
boundary-crossing candidates on multimodal landscapes and early reverie
evaluations poison basin selection before v3's escape system engages.
Schwefel spot check: 0.21% vs 0.16%, both total failures as before.

Kept in the repo as infrastructure (hooks + verified harness + negative
evidence) for a future v5 that would need a non-smoothing ranker
(e.g. landscape-aware surrogate) before speculative evals can help.
**v3 stays the recommended optimizer.**

## Bicameral on real data: BRFSS 2015 diabetes (`ml_benchmark.py`)

Binary 50/50 split (70,692 rows, 21 features, `data/` — both BRFSS
CSVs plus ESDRP are versioned in the repo for reproducibility).
Stratified 70/15/15 split, seed 42.
Full log: `results/ml/ml_results.txt`.

| method | test acc | test F1 | test AUC |
|---|---|---|---|
| dummy (majority) | 50.00% | 0.0000 | — |
| logreg (sklearn LBFGS) | 74.76% | 0.7522 | 0.8250 |
| **Bicameral-direct** (22-D weights) | **74.55%** | 0.7489 | 0.8239 |
| randomforest (300) | 73.64% | 0.7471 | 0.8119 |
| hgb (defaults) | 75.26% | 0.7630 | 0.8301 |
| mlp (64x32) | 73.21% | 0.7298 | 0.8082 |
| randsearch-HPO (108 evals) | 75.29% | 0.7625 | 0.8301 |
| **Bicameral-HPO** (108 evals) | **75.11%** | 0.7618 | 0.8299 |

Takeaways: Bicameral trained logistic weights from scratch to within 0.2pp of
LBFGS (74.55 vs 74.76%, 2000 evals, 0.6 s) — statistically a tie on
accuracy (McNemar p=0.12) with a hairsbreadth but real AUC concession
(0.8239 vs 0.8250, DeLong p=0.001, n=10.6k). For HPO all three
(defaults / random / Bicameral) tie at ~75.1–75.3% with ALL paired
tests n.s. (p=0.26–0.94): this dataset plateaus there and tuning is
certified pointless (Bicameral found the best *validation* loss,
0.50110 vs 0.50129, in half the wall time, 75s vs 141s — test noise
flips the ranking). Note: the 3-class `012` file
is 84/14/2 imbalanced, so raw accuracy is misleading there —
full 3-class protocol in the next subsection (dummy 84.24%,
HGB 85.03%, macro-F1 0.40).

### Task 4: paired stats + SHAP for the ML benchmark

The table was point estimates only — the weakest section
statistically. `ml_benchmark.py` now adds, stolen honestly from the two
diabetes papers: **[4]** paired tests for the headline comparisons —
DeLong test on AUC (Paper-2 method), exact McNemar on accuracy, and
bootstrap 95% CIs, with verdicts above; **[5]** SHAP explainability
(Paper-1 method): global mean|SHAP| ranking is clinically coherent —
GenHlth (0.664), HighBP (0.468), BMI (0.412), Age (0.333), HighChol
(0.308) — plus per-instance top features for a TP/TN/FP/FN. One
honest flag from the instances: the FN case shows HvyAlcoholConsump=1
pushing *against* a diabetes prediction (shap −0.61) — a correlation
artifact worth follow-up, not a medical claim. Pipeline was first
validated on an ESDRP smoke run, where SHAP's top-3
(Polydipsia/Polyuria/Gender) independently reproduced the paper's SHAP
finding on the same dataset.

### 012 multiclass: same protocol on Diabetes_012 (`ml_benchmark_012.py`)

Full log: `results/ml/ml_results_012.txt`. Same splits/protocol as
binary (70/15/15, seed 42), 3-class target (0 = none 84.2%, 1 =
prediabetes 1.8%, 2 = diabetes 13.9%; n = 253,680). Multiclass
ports: macro-F1 + macro-AUC(OvR), 66-D Bicameral-direct (softmax),
multiclass HPO loss, per-class OvR DeLong + Stuart-Maxwell
(the k-class McNemar; validated: 2-class reduces to McNemar
chi2 = 3.3333 exactly) + bootstrap CIs, per-class SHAP,
confusion matrices.

| method | test acc | macro-F1 | macro-AUC | F1 per class |
|---|---|---|---|---|
| dummy (majority) | 84.24% | 0.305 | — | .914/.000/.000 |
| logreg (LBFGS) | 84.63% | 0.394 | 0.780 | .915/.000/.268 |
| Bicameral-direct, 2k evals | 81.78% | **0.424** | 0.715 | .900/.036/.336 |
| Bicameral-direct, 6k evals (probe) | 84.51% | 0.401 | 0.770 | .914/.000/.287 |
| randomforest (300) | 84.36% | 0.401 | 0.747 | .914/.003/.287 |
| hgb (defaults) | 85.03% | 0.400 | 0.787 | .918/.000/.282 |
| mlp (64x32) | 84.51% | 0.399 | 0.773 | .914/.000/.283 |
| randsearch-HPO | 84.97% | 0.402 | 0.789 | .917/.000/.288 |
| Bicameral-HPO | 84.97% | 0.400 | 0.789 | .917/.000/.282 |

Findings. (1) Prediabetes (class 1) is unpredictable from these
features: every sklearn method scores F1 = 0.000 (RF manages a
single hit: 1/695). (2) Can Bicameral train multiclass? YES at
proportional budget — the 6k post-hoc probe (91 evals/dim, parity
with binary's 22-D/2k) matches LBFGS (84.51 vs 84.63%, macro-F1
0.401 vs 0.394). At the flat 2k budget it underfits accuracy
(81.78%, below dummy) while topping macro-F1 (0.424, only nonzero
class-1 F1) — an under-optimization signature, honestly not a win.
(3) HPO trio ties again (84.97/84.97/85.03; Bicameral found
val 0.39152 vs RS 0.39146, test-identical): tuning certified
pointless on a second dataset. Large-n cautionary pair:
Stuart-Maxwell calls Bic-HPO vs RS-HPO SIG (p = 0.0001) at
*identical* 84.97% accuracy (error-pattern, not error-rate,
difference), and one DeLong in twelve hits p = 0.024 with no
pattern — both reported as noise, not findings. (4) SHAP global
top-5 (GenHlth/Age/HighBP/BMI/HighChol) matches the binary run's
set; class-1 drivers (Age/BMI/HighChol/Income) look sensible
despite zero recall. (5) Best accuracy beats dummy by 0.8pp —
the 012 task is a near-plateau for everyone.
