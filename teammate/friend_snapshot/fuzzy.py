"""fuzzy.py - a small, fully transparent Mamdani fuzzy inference engine in NumPy."""
import json
import numpy as np


def trapmf(x, a, b, c, d):
    """Trapezoid: 0 below a, rising to 1 at b, flat to c, falling to 0 at d.
    a == b gives a left shoulder (1 for everything below c); c == d gives a right shoulder."""
    x = np.asarray(x, dtype=float)
    rise = np.clip((x - a) / (b - a), 0.0, 1.0) if b > a else np.ones_like(x)
    fall = np.clip((d - x) / (d - c), 0.0, 1.0) if d > c else np.ones_like(x)
    return np.minimum(rise, fall)


BINARY = {"No": (0, 0, 0, 1), "Yes": (0, 1, 1, 1)}   # crisp 0/1 inputs as degenerate fuzzy sets

# Neighbouring sets cross at 0.5 exactly at the clinical cut point, and memberships sum to 1.
VARIABLES = {
    # calibrated model probability; crossovers at 0.08, 0.25, 0.45 (about 0.5x, 1.6x, 3x prevalence)
    "P":   {"Low": (0, 0, 0.06, 0.10), "Moderate": (0.06, 0.10, 0.20, 0.30),
            "High": (0.20, 0.30, 0.40, 0.50), "VeryHigh": (0.40, 0.50, 1, 1)},
    "BMI": {"Normal": (0, 0, 24, 26), "Overweight": (24, 26, 29, 31),
            "Obese": (29, 31, 34, 36), "SeverelyObese": (34, 36, 100, 100)},
    "Age": {"Young": (1, 1, 4, 7), "Middle": (4, 7, 8, 11), "Senior": (8, 11, 13, 13)},
    "M":   {"Low": (0, 0, 0.15, 0.35), "Medium": (0.15, 0.35, 0.5, 0.7), "High": (0.5, 0.7, 1, 1)},
    "HighBP": BINARY, "HighChol": BINARY, "HeartDiseaseorAttack": BINARY,
}
# Output sets are mirror images about 50 and cross at 25, 50 and 75, so the tier rule is simply:
# score < 25 Minimal, 25-50 Watch, 50-75 Elevated, > 75 Priority.
OUTPUT = {"Minimal": (0, 0, 20, 30), "Watch": (20, 30, 45, 55),
          "Elevated": (45, 55, 70, 80), "Priority": (70, 80, 100, 100)}
TIERS = ["Minimal", "Watch", "Elevated", "Priority"]


def modifiable_score(bmi, phys_activity, fruits, veggies):
    """The modifiable items of the FINDRISC score mapped to BRFSS columns, scaled to 0-1:
    BMI 25-30 = 1 point, BMI over 30 = 3 points (smoothed by the BMI fuzzy sets),
    no leisure-time activity = 2 points, neither fruit nor vegetables daily = 1 point; maximum 6."""
    v = VARIABLES["BMI"]
    over = trapmf(bmi, *v["Overweight"])
    obese = trapmf(bmi, *v["Obese"]) + trapmf(bmi, *v["SeverelyObese"])
    no_produce = (np.asarray(fruits) == 0) & (np.asarray(veggies) == 0)
    points = 1 * over + 3 * obese + 2 * (1 - np.asarray(phys_activity)) + 1 * no_produce
    return points / 6.0


class MamdaniFIS:
    """min for AND, min implication, max aggregation, centroid defuzzification."""

    def __init__(self, rules, variables=VARIABLES, output=OUTPUT, n_grid=201):
        self.rules, self.variables, self.output = rules, variables, output
        self.z = np.linspace(0, 100, n_grid)
        self.out_mf = np.stack([trapmf(self.z, *output[t]) for t in TIERS])        # (T, Z)
        for r in rules:
            for var, term in r["if"]:
                assert term in variables[var], f"rule {r['id']}: unknown term {var}={term}"
            assert r["then"] in output, f"rule {r['id']}: unknown output {r['then']}"

    def firing(self, inputs):
        """(N, R) firing strengths: the minimum of each rule's antecedent memberships."""
        cols = []
        for r in self.rules:
            mus = [trapmf(inputs[var], *self.variables[var][term]) for var, term in r["if"]]
            cols.append(np.minimum.reduce(mus))
        return np.stack(cols, axis=1)

    def infer(self, inputs, chunk=5000):
        """Returns (crisp score 0-100, firing strengths). Rules are first merged per output
        term (max), which keeps memory at N x 4 x 201 instead of N x R x 201."""
        W = self.firing(inputs)
        S = np.stack([W[:, [i for i, r in enumerate(self.rules) if r["then"] == t]].max(axis=1)
                      if any(r["then"] == t for r in self.rules) else np.zeros(len(W)) for t in TIERS], axis=1)
        score = np.empty(len(W))
        for s in range(0, len(W), chunk):
            agg = np.minimum(S[s:s + chunk, :, None], self.out_mf[None]).max(axis=1)   # (n, Z)
            area = agg.sum(axis=1)
            score[s:s + chunk] = np.where(area > 0, (agg * self.z).sum(axis=1) / np.maximum(area, 1e-12), np.nan)
        return score, W

    def tier(self, score):
        """Index into TIERS of the output set with the highest membership at the crisp score."""
        return np.stack([trapmf(score, *self.output[t]) for t in TIERS], axis=1).argmax(axis=1)

    def trace(self, W_row, top=3, min_strength=0.05):
        """Human-readable explanation: the strongest rules that fired for one person."""
        order = np.argsort(-W_row)[:top]
        return [f"{self.rules[i]['id']} ({W_row[i]:.2f}): IF " +
                " AND ".join(f"{v} is {t}" for v, t in self.rules[i]["if"]) + f" THEN {self.rules[i]['then']}"
                for i in order if W_row[i] >= min_strength]


def load_rules(path="fuzzy_rules.json"):
    rules = json.load(open(path))
    for r in rules:
        r["if"] = [tuple(x) for x in r["if"]]
    return rules
