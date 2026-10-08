"""fsets.py - the single registry of named feature sets (feature_sets.json)."""
import json, datetime
from pathlib import Path

PATH = Path("feature_sets.json")


def load_sets():
    return json.loads(PATH.read_text()) if PATH.exists() else {}


def get(name):
    s = load_sets()
    assert name in s, f"unknown feature set '{name}'. Known: {sorted(s)}"
    return s[name]["features"]


def save(name, features, source, **info):
    """Add or overwrite one named set. `source` says which script and setting produced it."""
    s = load_sets()
    s[name] = {"features": list(features), "k": len(features), "source": source,
               "saved": datetime.date.today().isoformat(), **info}
    PATH.write_text(json.dumps(s, indent=2))
