"""One-time migration: widen results/registry.csv to the v2 FIELDS. Run from the repo root."""
import csv, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_io import FIELDS, REGISTRY

backup = REGISTRY.with_name("registry_v1_backup.csv")
assert not backup.exists(), "backup already exists: the migration has already run"
shutil.copy2(REGISTRY, backup)
with backup.open(newline="") as f:
    rows = list(csv.DictReader(f))
extra_cols = set(rows[0]) - set(FIELDS) if rows else set()
assert not extra_cols, f"old columns missing from FIELDS: {extra_cols}"
with REGISTRY.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in FIELDS})
print(f"migrated {len(rows)} rows -> {len(FIELDS)} columns; backup at {backup}")
