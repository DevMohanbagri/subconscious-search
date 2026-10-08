#!/usr/bin/env bash
# Copy diabetes-experiment outputs from the (ephemeral) friend clone into this repo.
# Idempotent -- run after each stage completes, or all at once at the end.
# Usage: ./src/diabetes/tools/mirror_results.sh   (from anywhere; paths are absolute)
set -u
SRC=/home/user/diabetes-project
DST=/home/user/subconscious-search/results/diabetes
mkdir -p "$DST"/{logs,tables,figures,analysis}
[ -d "$SRC/results" ] || { echo "NO FRIEND CLONE -- nothing to mirror"; exit 1; }
for f in 21_bicameral_compare.log 22_bicameral_hpo_seeds.log 23_pareto_bicameral.log \
         24_reduct_bicameral.log 25_mf_bicameral.log 26_pareto_bicameral_v2.log \
         27_reduct_bicameral_x.log 28_dataset_compare.log 29_hpo_independent.log; do
  [ -f "$SRC/results/$f" ] && cp "$SRC/results/$f" "$DST/logs/"
done
for f in registry.csv pareto_front.csv pareto_lr_overlay.csv pareto_cache.json \
         bicameral_reduct.csv rst_heldout_reduct.csv mf_tuning_pilot.csv mf_tuned_p.json \
         fs_k_curve.csv main_table_val.csv; do
  [ -f "$SRC/results/$f" ] && cp "$SRC/results/$f" "$DST/tables/"
done
for f in dataset_compare.csv dataset_compare_ind.csv; do
  [ -f "$SRC/results/$f" ] && cp "$SRC/results/$f" "$DST/analysis/"
done
for f in fig_pareto.png fig_reduct.png fig_mf_p.png fig_fs_k_curve.png fig_fuzzy_tiers.png \
         fig_rst_heldout_path.png; do
  [ -f "$SRC/figures/$f" ] && cp "$SRC/figures/$f" "$DST/figures/"
done
[ -f "$SRC/feature_sets.json" ] && cp "$SRC/feature_sets.json" "$DST/tables/"
echo "mirrored:"; find "$DST" -type f | sort
