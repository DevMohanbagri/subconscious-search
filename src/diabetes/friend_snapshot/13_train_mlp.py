"""13 - Train ONE configuration from the command line, e.g.
    python 13_train_mlp.py --feature-set rst_heldout_noleak --strategy none --seed 0"""
import argparse
import config  # noqa: F401
from models import run_config, STRATEGIES

ap = argparse.ArgumentParser()
ap.add_argument("--feature-set", default="all_21")
ap.add_argument("--strategy", default="none", choices=STRATEGIES)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--model", default="mlp", choices=["mlp", "lr", "lgbm"])
a = ap.parse_args()
print(run_config(a.model, a.feature_set, a.strategy, a.seed, script="13_train_mlp.py"))
