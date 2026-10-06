# M4 run outputs (mirrored from the friend clone)

Populated by `../mirror_results.sh` after each stage completes. The friend's
clone is ephemeral (wiped twice already); THESE copies are the durable record.

- `logs/` -- `21_..25_*.log` (full stdout of every stage)
- `tables/` -- `registry.csv` (append-only record incl. our rows),
  `pareto_front.csv`, `bicameral_reduct.csv`, `mf_tuning_pilot.csv`,
  `feature_sets.json` (with our saved sets), etc.
- `figures/` -- `fig_pareto.png`, `fig_reduct.png`, `fig_mf_p.png`, overlays.
