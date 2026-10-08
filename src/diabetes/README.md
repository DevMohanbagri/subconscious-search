# M4 diabetes work (exps 21-29): our contributions to the friend's diabetes-project

The friend's repo (`sohammawai/diabetes-project`) cannot be pushed from here,
and its sandbox clone was wiped repeatedly, losing uncommitted work. This
directory is the wipe-proof home of everything M4 added; the friend's
checkout remains the source of truth for *execution*.

- `experiments/` -- our stage scripts (run inside a `diabetes-project` clone):
  - `21_bicameral_compare.py` -- optimizer shootout (direct + HPO) + paired stats
  - `22_bicameral_hpo_seeds.py` -- 5-seed Bicameral-HPO vs RandomSearch-HPO
  - `23_pareto_bicameral.py` -- Pareto front (Bicameral vs NSGA-II vs random)
  - `24_reduct_bicameral.py` -- rough-set reduct search (Bicameral vs greedy)
  - `25_mf_bicameral.py` -- fuzzy membership-function tuning
  - `26_pareto_bicameral_v2.py` -- Pareto, one-sided k-penalty (follow-up)
  - `27_reduct_bicameral_x.py` -- reduct at 400/K budget (follow-up)
  - `28_dataset_compare.py` -- 50/50-vs-imbalanced HPO duel (both datasets)
  - `29_hpo_independent.py` -- independent-stream HPO redo (definitive duel)
  - `rebuild_dedup.py` -- rebuild the friend's `data/dedup.csv` without UCI access
- `friend_snapshot/` -- pristine archive of the friend's repo @ `0a86385`
  (tracked files only); makes M4's work reconstructible anywhere. See its README.
- `tools/mirror_results.sh` -- copy run outputs from the friend clone into
  `results/diabetes/` (idempotent; run after each stage).

Run outputs live in `results/diabetes/` (`logs/`, `tables/`, `figures/`, `analysis/`).
