# Prior-art search: is SMO-Pop v3 a new optimization technique?

**Date: 2026-09-22. Method: systematic web search over the five
claim-bearing parts of v3 (details below); ~25 queries, ~60 sources
screened.** This is a diligent first pass, not an exhaustive scholarly
review: before any formal publication claim, re-verify with Google
Scholar / supervisor / librarian, especially the "appears novel"
verdicts. All "known" verdicts below name checkable citations.

## Bottom line

| # | Claim-bearing part | Verdict |
|---|--------------------|---------|
| M1 | Credit-assignment gate (dual-stream allocation by realized improvement) | Principle KNOWN (adaptive operator selection); **instantiation appears NOVEL** |
| M2 | Double-stall → terminal hopping phase + memetic drainage | Ingredients KNOWN; **escalation policy appears NOVEL (narrow)** |
| M3 | Full composition (surrogate-assisted memetic CMA-ES) | **NOVEL composition**, known ingredients |
| M4 | v1 supervised latent space + Matérn surrogate (supporting) | Idea KNOWN; instantiation distinctive; choice has a KNOWN weakness (see v5 note) |
| M5 | Conscious/subconscious dual-process framing | Specific mapping unused; metaphor space CROWDED — claim mechanism, not metaphor |

**Overall: v3 clears the bar for "a new optimization technique" as a
novel composition with two narrow mechanism/policy novelties and
ablation evidence — provided the must-cite list below is cited, the
claims are scoped as phrased here, and (for paper level) BBOB +
BIPOP/saACM baselines are added.**

---

## M1. Credit-assignment gate — principle known, instantiation novel

**What v3 does** (`smo_pop.py`, gate ~L487–556): two streams (λ CMA
offspring + k surrogate-selected latent candidates). Each generation
that improves the global best adds relative improvement to the winning
stream's credit; both credits decay ×0.97/generation; allocation is
`p_sub = sigmoid((log credit_sub − log credit_con)/T)` with annealed T
(soft early → greedy late); k is clipped to progress-shrunk bounds
(`credit_gate=False` fixes p_sub = 0.5; ablation: −20pp rastrigin,
−27pp rosenbrock).

**Closest prior art (MUST CITE):**
- **Adaptive Operator Selection framework**: credit assignment +
  selection rule; Probability Matching / Adaptive Pursuit —
  Thierens (2005) [12]. v3's gate is PM-shaped (soft allocation from
  recent production).
- **Bandit-based AOS**: Fialho, Da Costa, Schoenauer & Sebag (2008,
  DMAB) [13]; Ex-DMAB / extreme-value credit (2009) [14]; rank-based
  AUC + UCB follow-ups. v3 uses winner-take-all improvement credit
  (extreme-flavored) with decay instead of UCB exploration.
- **FRRMAB** (Li, Fialho, Kwong & Zhang, 2014) [15]: sliding window of
  recent *fitness improvement rates* per operator + decaying mechanism
  — the closest published reward to v3's decayed improvement credits.
- **mVIE** (Maesani et al., 2018) [23]: an adaptive scheduler that
  probabilistically switches between exploitation (local CMA-ES units)
  and exploration (DE recombination) based on moving averages of recent
  improvement rates — the closest full-system relative of the gate
  idea, in constrained optimization.
- **SHADE** (Tanabe & Fukunaga, 2013) [16]: success-history adaptation
  (same "reward what recently worked" spirit, applied to DE parameters).

**What appears distinct:** no found work allocates *evaluation counts
between CMA offspring and surrogate-selected latent-space candidates*
via annealed-softmax on log decayed-improvement-credits inside a single
generational surrogate-assisted CMA-ES loop.

**Safe claim:** "A novel AOS-style credit gate for dual-stream
surrogate-assisted CMA-ES, in the tradition of Thierens [12],
Fialho et al. [13–14], FRRMAB [15] and mVIE [23]."

## M2. Double-stall terminal phase + memetic drainage — narrow policy novelty

**What v3 does** (`smo_pop.py`, trigger ~L444–473, stall block ~L559–596,
`_basin_hopping` ~L335–401, `_local_polish` ~L307–333): generation-based
patience windows (30 gens); a stalled window (<1e-3 rel. improvement)
triggers L-BFGS-B basin drainage then IPOP doubling + recenter on the
global best; **two consecutive** stalled windows (backup: ≥1 restart +
final 35% budget) hand **ALL remaining budget** to a terminal phase —
cheap 1-eval/step Metropolis walk (block-coordinate Gaussian diffusion
+ 35% Cauchy jumps, annealed T) with L-BFGS-B drains every 150 steps or
on new session best, monotonic session best. Smooth runs never stall
twice → zero regression risk (ablation no-phase: rastrigin 50.1→8.5).

**Prior art for every ingredient (MUST CITE the used ones):**
- Basin Hopping — Wales & Doye (1997) [17]; monotonic BH — Leary
  (2000) [18]; Iterated Local Search — Lourenço, Martin & Stützle
  (2003) [19]; Generalized/ Dual Annealing — Xiang et al. (1997) [20]
  (the walk rhythm is closest to DA; cite as family).
- IPOP restarts — Auger & Hansen (2005) [2]; BIPOP — Hansen (2009) [5];
  active CMA — Jastrebski & Arnold (2006) [3]; base CMA-ES — Hansen &
  Ostermeier (2001) [1].
- CMA-ES + local-search memetics: 2-stage memetic CMA-ES with
  Simplex/BOBYQA/L-BFGS-B (2014) [22] (closest structural relative:
  sequential CMA-then-LS); local-search chains with LS-intensity
  adaptation, MA-LSCh-CMA — Molina, Lozano & Herrera (2010) [21];
  stagnation-triggered CMA-ES+LBFGS inside DE — jSO_CMA-ES_LBFGS
  (2022) [24]; population BH — BHPOP (2024) [25].

**What appears distinct:** the *escalation policy* as a whole —
drain → double+recenter → (2nd consecutive stall) → 100%-of-remainder
terminal hopping with backup trigger and demonstrated zero-regression
gating. Stagnation-triggered LS exists [21,22,24]; the consecutive-stall
counter driving an irreversible all-budget phase flip was not found.

**Safe claim:** "A stall-streak-triggered terminal-phase policy for
memetic CMA-ES" + the ablation numbers. Do not claim BH/ILS/DA novelty.

## M3. Full composition — novel, with strong neighbors

**Closest full-system relatives (MUST CITE / benchmark against):**
- **saACM-ES / s\*ACM-ES / BIPOP-saACM-ES + HCMA** (Loshchilov,
  Schoenauer & Sebag, 2010–2013) [8,9]: self-adaptive surrogate-assisted
  CMA-ES with Ranking-SVM comparison-based surrogate; HCMA hybrids
  BIPOP-saACM with STEP line search + NEWUOA. Strongest neighbor:
  same base algorithm + surrogate + hybridization instinct. v3 differs
  in surrogate type (Matérn regression in latent space vs Ranking SVM),
  adaptation target (stream eval-counts vs surrogate lifelength +
  hyperparams), and escape machinery (memetic + hopping vs restarts /
  STEP + NEWUOA).
- **mVIE** (2018) [23]: adaptive local/global scheduler (constrained).
- **2-stage memetic CMA-ES** (2014) [22]: CMA-ES then LS.
- Also in-family: lmm-CMA (Kern, Hansen & Koumoutsakos, 2006) [7],
  lq-CMA-ES (Hansen, 2019), DTS-CMA-ES (Pitra et al.).

**Gap:** v3 has been benchmarked against plain CMA-ES, DE, DA, RS —
not against BIPOP-aCMA-ES or the saACM family. **Minimum for a
paper-level novelty claim: add a BIPOP baseline** (cheap via existing
recipes [5]); ideally discuss/limit-compare with saACM.

## M4. Latent space + surrogate — known idea, distinctive instance, known weakness

- Supervised dimensionality reduction + Kriging/GP surrogate: **KPLS**
  (Bouhlel et al., 2016) [10] is the closest published idea (PLS latent
  directions + Kriging). v3's supervised-whitening + SVD latent space
  with Matérn-3/2 acquisition-ranked pool is a distinctive instance,
  but the *idea* is known — cite [10].
- Random embeddings for high-dim surrogate optimization: **REMBO**
  (Wang et al., 2013/2016) [11] — related family, worth citing as
  alternative philosophy (random vs supervised reduction).
- **Known weakness of v3's choice:** "Comparison-based optimizers need
  comparison-based surrogates" (Loshchilov, Schoenauer & Sebag, 2010)
  [8] argues value-regression surrogates break CMA-ES's rank invariance;
  rank-based SVM surrogates fix it. This paper is the theoretical
  backing for BOTH the v4-ghost autopsy (smoothing surrogate mis-ranks
  speculative candidates) AND the roadmap's #1 item (rank-based/ordinal
  surrogate for v5). Cite it there, not as an afterthought.
- v1's diagonal core is sep-CMA-ES (Ros & Hansen, 2008) [4] — cite;
  roadmap's diagonal-start idea is dd-CMA (Akimoto & Hansen, 2016/2020)
  [6] — cite when implemented.

## M5. Framing — unused mapping, crowded metaphor space, taken names

- No found optimizer uses a **conscious/subconscious dual-stream**
  mapping specifically. But mind-metaphor optimizers exist and the
  territory is crowded: Brain Storm Optimization — Shi (2011) [26];
  Human Mental Search (2017) [27]; Higher-Order Thinking Skills
  Optimizer (2026, cognitive processes → adaptive operator selection);
  Conscious-Neighborhood JSO (2025). None is a dual-stream CMA-ES.
- Dual-process (System 1/2) framing: Kahneman (2011) [29]; note his own
  caveat that S1/S2 are an *abstraction*, not brain modules — keep
  IDEAS.md consistent with that.
- **Reviewer climate:** Sörensen (2015), "Metaheuristics — the metaphor
  exposed" [28], plus follow-ups (Camacho-Villalón et al.). The field
  actively punishes metaphor-based novelty claims. Consequence: the
  paper's contribution claim must be M1–M3 (mechanisms + evidence);
  the dual-process story belongs in the introduction/motivation and in
  IDEAS.md — exactly where it is now.
- **NAME COLLISIONS (both blocking for publication):**
  - **SMO** = Sequential Minimal Optimization, Platt (1998),
    MSR-TR-98-14 [30] — one of the most famous acronyms in ML.
  - **GHOST** = General meta-Heuristic Optimization Solving Tool,
    Richoux et al. (2015), a combinatorial CSP/COP solver [31].
    Different domain, same literature pool — rename if v4/ghost is
    ever published. Recommended: rename project (e.g. CGES /
    DUET-CMA / credit-gated dual-stream CMA-ES) before any preprint.

---

## Must-cite bibliography (numbered as referenced above)

**CMA family**
1. Hansen, N. & Ostermeier, A. (2001). Completely Derandomized
   Self-Adaptation in Evolution Strategies. *Evol. Comput.* 9(2).
   https://cma-es.github.io/
2. Auger, A. & Hansen, N. (2005). A Restart CMA-ES With Increasing
   Population Size (IPOP). CEC 2005.
   https://sci2s.ugr.es/sites/default/files/files/TematicWebSites/EAMHCO/contributionsCEC05/auger05ARCMA.pdf
3. Jastrebski, K. & Arnold, D. (2006). Improving evolution strategies
   through active covariance matrix adaptation. IEEE WCCI.
   (see bibliography at https://cma-es.github.io/)
4. Ros, R. & Hansen, N. (2008). A Simple Modification in CMA-ES
   Achieving Linear Time and Space Complexity (sep-CMA-ES). PPSN X.
   https://link.springer.com/chapter/10.1007/978-3-540-87700-4_30
5. Hansen, N. (2009). Benchmarking a BI-Population CMA-ES on the
   BBOB-2009 Function Testbed (BIPOP). BBOB workshop.
   (accessible description: https://deap.readthedocs.io/en/master/examples/bipop_cmaes.html)
6. Akimoto, Y. & Hansen, N. (2016/2020). Diagonal Acceleration for
   CMA-ES (dd-CMA). *Evol. Comput.* 28(3).
   https://direct.mit.edu/evco/article/28/3/405/94999/Diagonal-Acceleration-for-Covariance-Matrix

**Surrogate-assisted**
7. Kern, S., Hansen, N. & Koumoutsakos, P. (2006). Local Meta-models
   for Optimization Using Evolution Strategies (lmm-CMA). PPSN.
   https://www.researchgate.net/publication/220702281_Local_Meta-models_for_Optimization_Using_Evolution_Strategies
8. Loshchilov, I., Schoenauer, M. & Sebag, M. (2010).
   Comparison-Based Optimizers Need Comparison-Based Surrogates.
   PPSN XI. https://link.springer.com/chapter/10.1007/978-3-642-15844-5_37
9. Loshchilov, I., Schoenauer, M. & Sebag, M. (2012–2013). saACM-ES /
   BIPOP-saACM-ES / HCMA (BI-population CMA-ES with Surrogate Models
   and Line Searches). http://loshchilov.com/saacm.html
   https://www.researchgate.net/publication/236863907_BI-population_CMA-ES_Algorithms_with_Surrogate_Models_and_Line_Searches
10. Bouhlel, M.A. et al. (2016). Improving kriging surrogates of
    high-dimensional design models by PLS dimension reduction (KPLS).
    https://www.researchgate.net/publication/280894863_Improving_kriging_surrogates_of_high-dimensional_design_models_by_Partial_Least_Squares_dimension_reduction
11. Wang, Z. et al. (2013/2016). Bayesian Optimization in a Billion
    Dimensions via Random Embeddings (REMBO). IJCAI/JAIR.
    https://ml.informatik.uni-freiburg.de/wp-content/uploads/papers/16-JAIR-REMBO.pdf

**Adaptive selection / credit**
12. Thierens, D. (2005). An adaptive pursuit strategy for allocating
    operator probabilities. GECCO.
    (background: https://link.springer.com/article/10.1007/s10515-025-00501-z)
13. Fialho, A., Da Costa, L., Schoenauer, M. & Sebag, M. (2008).
    Adaptive operator selection with dynamic multi-armed bandits. GECCO.
    https://dl.acm.org/doi/10.1145/1389095.1389272
14. Fialho, A. et al. (2009). Extreme compass / Ex-DMAB for adaptive
    operator selection. CEC / GECCO companion.
    https://dl.acm.org/doi/pdf/10.1145/1570256.1570305
15. Li, K., Fialho, A., Kwong, S. & Zhang, Q. (2014). Adaptive operator
    selection with bandits for MOEA/D (FRRMAB). *IEEE TEVC* 18(1).
    https://scholars.cityu.edu.hk/en/publications/adaptive-operator-selection-with-bandits-for-a-multiobjective-evo/
16. Tanabe, R. & Fukunaga, A. (2013). Success-history based parameter
    adaptation for DE (SHADE). CEC.
    https://www.researchgate.net/publication/312472606_Success-History_based_parameter_adaptation_for_differential_evolution

**Memetic / hopping / annealing**
17. Wales, D.J. & Doye, J.P.K. (1997). Global Optimization by
    Basin-Hopping. *J. Phys. Chem. A* 101.
    (accessible description: https://machinelearningmastery.com/basin-hopping-optimization-in-python/)
18. Leary, R.H. (2000). Global Optimization on Funneling Landscapes
    (monotonic BH). *J. Global Optim.* 18:367–383.
    (cited e.g. in https://arxiv.org/pdf/2108.05229)
19. Lourenço, H.R., Martin, O.C. & Stützle, T. (2003). Iterated Local
    Search. Handbook of Metaheuristics.
    https://link.springer.com/chapter/10.1007/0-306-48056-5_11
20. Xiang, Y. et al. (1997). Generalized Simulated Annealing / Dual
    Annealing. *Phys. Lett. A* 233.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html
21. Molina, D., Lozano, M. & Herrera, F. (2010). Memetic algorithms
    for continuous optimisation based on local search chains
    (MA-LSCh-CMA). https://pubmed.ncbi.nlm.nih.gov/20064025/
22. A CMA-ES-based 2-stage memetic framework (CMA-ES + Simplex/BOBYQA/
    L-BFGS-B). (2014). IEEE.
    https://ieeexplore.ieee.org/document/7007819/
23. Maesani, A. et al. (2018). Memetic Viability Evolution (mVIE).
    https://www.researchgate.net/publication/386729704_Memetic_Viability_Evolution_for_Constrained_Optimization
    (also https://pmc.ncbi.nlm.nih.gov/articles/PMC5427971/)
24. jSO_CMA-ES_LBFGS (2022). Hybrid cooperative DE assisted by CMA-ES
    with LBFGS. https://link.springer.com/article/10.1007/s00521-021-06849-z
25. BHPOP: population-based basin hopping (2024).
    https://arxiv.org/html/2403.05877v1

**Framing / names**
26. Shi, Y. (2011). Brain Storm Optimization.
    (description: https://www.mdpi.com/2227-7390/10/8/1303)
27. Mousavirad, S.J. & Ebrahimpour, H. (2017). Human Mental Search.
    https://link.springer.com/article/10.1007/s10489-017-0903-6
28. Sörensen, K. (2015). Metaheuristics — the metaphor exposed.
    *Int. Trans. Oper. Res.* 22:3–18.
    https://onlinelibrary.wiley.com/doi/abs/10.1111/itor.12001
29. Kahneman, D. (2011). Thinking, Fast and Slow (System 1/2).
    (background: https://www.suebehaviouraldesign.com/blog/kahneman-fast-slow-thinking)
30. Platt, J.C. (1998). Sequential Minimal Optimization (SMO).
    MSR-TR-98-14.
    https://www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/
31. Richoux, F. et al. (2015). GHOST: A Combinatorial Optimization
    Solver. https://hal.science/hal-01152231/document

## What remains for a paper-level claim

1. Rename (SMO and GHOST both taken) — before any preprint.
2. Add BIPOP-aCMA-ES baseline; discuss saACM family [9] explicitly.
3. BBOB/COCO run (the venue-standard evidence; CEC2017 done here is
   strong supporting evidence, not a substitute).
4. More seeds + full stats + wall-clock/complexity analysis.
5. Keep the contribution claim on M1–M3; metaphor stays in
   introduction + IDEAS.md.
