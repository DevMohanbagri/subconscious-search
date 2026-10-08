# Diabetes experiment outputs (M4, exps 21-29)

Durable record of every M4 run. The friend's clone is ephemeral (wiped
repeatedly); THESE copies survive. Populated by
`src/diabetes/tools/mirror_results.sh` after each stage completes
(exps 28/29 also write their comparison CSVs here directly).

- `logs/` -- full stdout of every stage (`21_...29_*.log`).
  Known gap: the `26_*`/`27_*` logs were lost to sandbox wipes before they
  could be mirrored; their results survive in `tables/registry.csv`.
- `tables/` -- `registry.csv` (append-only record incl. our rows),
  `pareto_front.csv`, `bicameral_reduct.csv`, `mf_tuning_pilot.csv`,
  `feature_sets.json` (with our saved sets), etc.
- `analysis/` -- cross-run comparison tables: `dataset_compare.csv` (exp 28:
  50/50-vs-imbalanced duel) and `dataset_compare_ind.csv` (exp 29:
  independent-stream redo).
- `figures/` -- `fig_pareto.png`, `fig_reduct.png`, `fig_mf_p.png`, overlays.

Headline results: the HPO duel (exps 22/28/29) ends in a TIE on both datasets
under independent random streams; exp 23 NSGA-II beats Bicameral-v1 on
Pareto-HV; exps 24/27 greedy reducts beat Bicameral; exp 25 default fuzzy MFs
sit at ceiling. See `logs/29_hpo_independent.log` for the final verdicts.
