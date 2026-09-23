# The ideas behind subconscious-search

**Author's statement — DRAFT, 2026-09-21.**
This document records the conceptual model and the numbered ideas behind
this project, in the author's own framing. It is the "why" to the README's
"what". Status notes cite measured evidence from the repo's benchmark logs.

---

## The core model: an optimizer built like a mind

My starting intuition is that human problem-solving runs on two systems —
a slow, deliberate, model-building **conscious** process and a fast,
associative, generative **subconscious** one — and that a black-box
optimizer should be built the same way:

- **Conscious** = full-covariance CMA-ES core. Deliberate, statistical,
  builds an explicit model of the landscape (mean + covariance + step
  size) and moves carefully. Slow to commit, hard to fool.
- **Subconscious** = a generative stream of candidates from compressed
  experience: a supervised latent space over everything ever evaluated,
  a surrogate (intuition: "this *feels* promising"), memory traces of
  explored regions, recombination of past impressions (dreams).
- **Attention/gate** = a credit-assignment mechanism that watches which
  stream actually produces improvements and allocates future evaluation
  budget accordingly — like attention flowing to whichever mental process
  is currently paying off.
- **Intuition** = compressed experience, not magic: the surrogate and the
  memory are just past evaluations folded into a fast guess. When the
  guess is wrong (multimodal barriers, deceptive basins), the conscious
  core and the escape machinery overrule it.

Two convictions follow from this model and constrain every design choice:

1. **Fairness of attention.** Every evaluation any stream spends —
   including local-search probes — counts against one shared budget.
   (Implemented as `EvalCounter` in `smo_pop.py`.)
2. **Do no harm.** A new mental faculty must never make the mind dumber:
   every addition is gated so smooth runs never pay for it, and must
   survive an ablation proving it earns its budget.

## Idea catalog

| # | Idea | Lives in | Status |
|---|------|----------|--------|
| 1 | Full-covariance conscious core (the deliberate mind needs a real landscape model; diagonal-only can't follow curved valleys) | v3, `smo_pop.py::FastFullCMA` | ✅ supported — rosenbrock 21.1% → 96.6% |
| 2 | Generational population loop: each generation evaluates λ conscious offspring PLUS k subconscious candidates | v3, `SMOPop.optimize` | ✅ supported — core mean 61.9% (v1) → 89.2% |
| 3 | Memetic escape: drain the stuck basin with local search, then restart; two *consecutive* stalls hand ALL remaining budget to a terminal hopping phase | v3, `_local_polish` + `_basin_hopping` | ✅ supported — rastrigin 8.5% (no-phase) → 50.1% |
| 4–7 | *(working ideas from earlier sessions — titles to be confirmed by author; likely: the credit-assignment gate, the active covariance update, archive memory matrices / subconscious pool, exact budget accounting)* | v1/v3 | ✅ all supported by ablation (no-credit −20pp rastrigin / −27pp rosenbrock; no-active −7.7pp mean; no-sub −29pp rosenbrock) |
| 8 | Ghost Landscape: a memory of explored regions whose traces attract (good) or repel (trapped) new search | v4, `smo_ghost.py::GhostMemory` | ❌ negative — extras net-negative (see below) |
| 9 | Conflict Search: mine impressions that disagree (near in latent space, far in value) and probe between/through them | v4, `conflict_probes()` | ❌ negative as eval-spending probes |
| 10 | Latent Vector Memory: store impressions as latent codes, not raw points | v4, ghost memory | ⚠️ implemented; inert-or-negative via the evals it feeds |
| 11 | Salience/Surprise: unexpected results earn stronger, longer traces | v4, surprise scoring | ⚠️ implemented; inert without speculative evals |
| 12 | Adaptive Memory Decay: traces fade; useful ones refresh | v4, half-life decay | ⚠️ implemented; inert without speculative evals |
| 13 | Dreaming: recombine strong impressions into new candidates (tried blending, then crossover) | v4, `dream()` | ❌ negative — blend 50→27%; crossover still negative at 12 seeds |
| 14 | Uncertainty-driven exploration: sample wider where memory is thin | v4, unfamiliarity gating | ⚠️ implemented; inert without speculative evals |

### The v4 lesson (my interpretation)

Ideas #8–#14 failed *as evaluation-spending machinery* for a precise,
diagnosed reason: a smoothing surrogate cannot rank boundary-crossing
speculative candidates on multimodal landscapes, and early ghost
evaluations poison basin selection (12-seed: rastrigin 40.2 vs 60.5,
rosenbrock 79.6 vs 97.8). With speculative evals disabled, the scoring /
decay / noise machinery is byte-inert — so the failure is localized,
not diffuse. I keep the infrastructure because the diagnosis prescribes
the fix: a non-smoothing (rank-based or local) intuition module. A mind
whose intuition smooths away cliffs will walk off them; the fix is
better intuition, not lobotomy.

## What I claim

- The dual-process architecture and the numbered ideas above are mine:
  the mapping from "conscious / subconscious / intuition / dreaming" to
  these specific computational mechanisms is my original design.
- The folk-psychology vocabulary itself (subconscious, intuition,
  System-1/System-2-style duality) belongs to the commons (James, Freud,
  Kahneman/Tversky and others); I claim the *mapping*, not the words.
- The technical contribution — credit-gated dual-stream allocation,
  double-stall phase switching, and their ablated effects — stands on
  the benchmark evidence with or without the story.

## Provenance

- This statement: 2026-09-21 (draft for author review).
- The git history of this repository timestamps the ideas as implemented.
- Evidence: `results/core/results_popv3_dim10_5k.txt`, `results/core/ablation_results.txt`,
  `results/core/comparison_results_v3.txt`, `results/core/comparison_ghost_dim10.txt`,
  `results/core/results_ghost_dim10_5k.txt`.
