"""Unit checks for fuzzy.py. Run: python tests_fuzzy.py"""
import numpy as np
from fuzzy import trapmf, VARIABLES, OUTPUT, TIERS, MamdaniFIS

# 1. every non-binary variable is a partition: memberships sum to 1 across its range
for var, lo, hi in [("P", 0, 1), ("BMI", 12, 98), ("Age", 1, 13), ("M", 0, 1)]:
    x = np.linspace(lo, hi, 1001)
    total = sum(trapmf(x, *p) for p in VARIABLES[var].values())
    assert np.allclose(total, 1.0), var
# 2. neighbouring sets cross at 0.5 exactly at the clinical cut points
assert np.isclose(trapmf(25, *VARIABLES["BMI"]["Normal"]), 0.5)
assert np.isclose(trapmf(30, *VARIABLES["BMI"]["Obese"]), 0.5)
assert np.isclose(trapmf(5.5, *VARIABLES["Age"]["Middle"]), 0.5)
# 3. one rule firing fully returns its output set's centroid (12.67 for Minimal by hand;
#    the 201-point grid lands within 0.2 of the exact value)
for term, want in [("Minimal", 12.67), ("Watch", 37.5), ("Elevated", 62.5), ("Priority", 87.33)]:
    fis = MamdaniFIS([{"id": "T", "if": [("P", "VeryHigh")], "then": term}])
    score, _ = fis.infer({"P": np.array([0.9])})
    assert abs(score[0] - want) < 0.2, (term, score[0])
    assert TIERS[fis.tier(score)[0]] == term
print("fuzzy.py: all checks passed")
