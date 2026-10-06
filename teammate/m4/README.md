# M4 mirror (backup of our contributions to the friend's diabetes-project)

The friend's repo cannot be pushed from here, and its clone already vanished
once (losing uncommitted work). This directory mirrors every M4 script we add
there, so nothing is ever lost again. Source of truth for execution is still
the friend's checkout; this is the wipe-proof backup.

- `21_bicameral_compare.py` — optimizer shootout (direct + HPO) + paired stats
- `22_bicameral_hpo_seeds.py` — 5-seed Bicameral-HPO vs RandomSearch-HPO
- `23_pareto_bicameral.py` — Stage 5 Target 1: Pareto front (Bicameral vs NSGA-II vs random)
