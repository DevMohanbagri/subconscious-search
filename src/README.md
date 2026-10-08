# Source code

All runnable code lives here. Scripts use paths relative to the repo root,
so run them from there, e.g.:

```bash
python src/bicameral.py --dim 10 --max-evals 5000 --n-runs 5
PYTHONPATH=src python -c "from bicameral import Bicameral"
```

- `bicameral.py`, `bicameral_v1.py`, `reverie.py`, `zoo_methods.py` -- optimizers
- `bbob_benchmark.py`, `cec_benchmark.py`, `extra_benchmarks.py`,
  `compare_baselines.py`, `ablation.py` -- synthetic benchmark harnesses
- `ml_benchmark.py`, `ml_benchmark_012.py`, `esdrp_wrapper.py` -- real-data harnesses
- `diabetes/` -- M4 diabetes-project contributions (exps 21-29),
  friend-repo snapshot, and mirror tooling (see `diabetes/README.md`)
