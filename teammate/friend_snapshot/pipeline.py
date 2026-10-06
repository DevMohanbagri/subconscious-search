"""pipeline.py - the deployed system: answers -> calibrated probability -> fuzzy tier, rule trace,
modifiable-risk split (Design B). Used by app.py and 99_final_test.py, so both run identical code."""
import numpy as np, pandas as pd
from models import load_predictor
from fuzzy import MamdaniFIS, load_rules, modifiable_score, TIERS

HEALTHY = {"BMI": 23, "PhysActivity": 1, "Fruits": 1, "Veggies": 1}       # Design B reference values
LABELS = {"BMI": "body weight (BMI)", "PhysActivity": "leisure-time physical activity",
          "Fruits": "daily fruit", "Veggies": "daily vegetables"}


def _to_healthy(frame, cols):
    f = frame.copy()
    for c in cols:
        ref = HEALTHY[c]
        worse = f[c] > ref if c == "BMI" else f[c] < ref
        f.loc[worse, c] = ref
    return f


class RiskPipeline:
    def __init__(self, rules_path="fuzzy_rules_v2.json", spec_path="models/deployed.json"):
        self.predict_p = load_predictor(spec_path)
        self.fis = MamdaniFIS(load_rules(rules_path))

    def _score(self, frame):
        p = self.predict_p(frame)
        X = {"P": p, "BMI": frame.BMI.to_numpy(float), "Age": frame.Age.to_numpy(float),
             "M": modifiable_score(frame.BMI, frame.PhysActivity, frame.Fruits, frame.Veggies),
             "HighBP": frame.HighBP.to_numpy(), "HighChol": frame.HighChol.to_numpy(),
             "HeartDiseaseorAttack": frame.HeartDiseaseorAttack.to_numpy()}
        score, W = self.fis.infer(X)
        return p, score, W

    def assess(self, frame):
        frame = frame.reset_index(drop=True)
        p, score, W = self._score(frame)
        out = pd.DataFrame({"p_diabetes": p, "risk_score": score,
                            "tier": [TIERS[t] for t in self.fis.tier(score)]})
        # Design B: what does the model associate with each modifiable factor? Reset one at a time.
        gains = pd.DataFrame({c: np.clip(score - self._score(_to_healthy(frame, [c]))[1], 0, None)
                              for c in HEALTHY})
        base = self._score(_to_healthy(frame, list(HEALTHY)))[1]
        out["baseline_score"] = base
        out["modifiable_part"] = np.clip(score - base, 0, None)
        out["largest_modifiable"] = np.where(gains.max(axis=1) >= 0.5, gains.idxmax(axis=1).map(LABELS), "none")
        out["trace"] = [" | ".join(self.fis.trace(W[i])) for i in range(len(frame))]
        return out
