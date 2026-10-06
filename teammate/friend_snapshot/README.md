# Friend's repo snapshot (pristine delivery, no M4 files)

Provenance: `sohammawai/diabetes-project` @ `0a86385` ("registry migrated to v2"),
archived via `git archive 0a86385` on 2026-10-06. This is EXACTLY what the
friend delivered (tracked files only) -- none of M4's scripts or outputs.

Why this exists: the sandbox has already wiped the friend's clone twice,
losing uncommitted work. Our repo survives wipes; this snapshot plus
`../m4/` makes M4's work fully reconstructible anywhere.

Layout (next to this folder):
- `friend_snapshot/` (here) -- friend's bundle: stages 0-4 + demo + test scripts,
  `splits.npz` (frozen, never regenerate), baseline `results/registry.csv` (v2).
- `m4/` -- OUR contributions: `21_*..25_*` scripts + `rebuild_dedup.py`.
- `m4_results/` -- COPIES of run outputs (logs, CSVs, figures) mirrored here
  after each stage completes, so results survive a wipe too.

NOT included (regenerable, excluded deliberately):
- `data/dedup.csv` (22 MB) -- rebuild without UCI access via:
      cd <fresh friend clone> && python3 ../subconscious-search/teammate/m4/rebuild_dedup.py
  (asserts the md5 against splits.npz; must print HASH-MATCH).

Full restore after a wipe:
1. `git clone https://github.com/sohammawai/diabetes-project.git`
2. `pip install numpy scipy pandas scikit-learn lightgbm pymoo matplotlib`
3. rebuild dedup (above), verify `python3 -c "from data_io import load; load()"`
4. `cp ../subconscious-search/teammate/m4/2*.py .` and relaunch the stage chain.
