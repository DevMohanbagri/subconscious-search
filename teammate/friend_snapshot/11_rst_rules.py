"""11 - IF-THEN rules from granules of the leakage-free reduct, in the fuzzy layer's vocabulary."""
import json
import numpy as np, pandas as pd
from data_io import load, TARGET
from rst import extract_rules, rule_stats

df, tr, va, _ = load()


def to_vocab(x):
    """Crisp groups whose boundaries match the 0.5 crossings of the fuzzy sets in fuzzy.py."""
    out = pd.DataFrame(index=x.index)
    out["BMI"] = np.select([x.BMI < 25, x.BMI < 30, x.BMI < 35], [0, 1, 2], 3)   # Normal..SeverelyObese
    out["Age"] = np.select([x.Age <= 5, x.Age <= 9], [0, 1], 2)                   # Young <45, Middle 45-64, Senior 65+
    for c in ["HighBP", "HighChol", "HeartDiseaseorAttack"]:
        out[c] = x[c]
    out[TARGET] = x[TARGET]
    return out


NAMES = {"BMI": ["Normal", "Overweight", "Obese", "SeverelyObese"], "Age": ["Young", "Middle", "Senior"],
         "HighBP": ["No", "Yes"], "HighChol": ["No", "Yes"], "HeartDiseaseorAttack": ["No", "Yes"]}
ATTRS = ["BMI", "Age", "HighBP", "HighChol", "HeartDiseaseorAttack"]

trv, vav = to_vocab(df.loc[tr]), to_vocab(df.loc[va])
rules = extract_rules(trv, ATTRS, TARGET, min_support=300, hi_lift=2.0, lo_lift=0.5)
kept = []
for r in rules:
    n_v, rate_v, lift_v = rule_stats(vav, r["if"], TARGET)        # does it hold on validation?
    holds = lift_v >= 2.0 if r["kind"] == "high" else lift_v <= 0.5
    rate = r["rate"]
    then = ("Priority" if rate >= 0.40 else "Elevated") if r["kind"] == "high" else "Minimal"
    words = " AND ".join(f"{a} is {NAMES[a][v]}" for a, v in r["if"].items())
    print(f"{'kept' if holds else 'DROP'}  IF {words} THEN {then}   train n={r['support']} rate={rate:.3f} "
          f"lift={r['lift']:.2f} | val n={n_v} lift={lift_v:.2f}")
    if holds:
        kept.append({**r, "if_words": {a: NAMES[a][v] for a, v in r["if"].items()}, "then": then,
                     "val_support": n_v, "val_lift": round(lift_v, 3)})
json.dump(kept, open("rules_rst.json", "w"), indent=2, default=int)
print(f"{len(kept)} of {len(rules)} rules kept -> rules_rst.json")
