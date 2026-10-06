"""discretise.py - three binning schemes for BMI, MentHlth, PhysHlth. Fit on train, apply anywhere."""
import numpy as np
from sklearn.tree import DecisionTreeClassifier

BINNED = ["BMI", "MentHlth", "PhysHlth"]
CLINICAL = {"BMI": [18.5, 25, 30, 35],   # WHO: underweight / normal / overweight / obese I / obese II+
            "MentHlth": [1, 14],         # 0 / 1-13 / 14-30 (14+ = frequent distress in BRFSS work)
            "PhysHlth": [1, 14]}
LABELS = {"BMI": ["Underweight", "Normal", "Overweight", "Obese I", "Obese II+"],
          "MentHlth": ["0 days", "1-13", "14-30"], "PhysHlth": ["0 days", "1-13", "14-30"]}


def fit_cuts(train, scheme, target="Diabetes_binary"):
    """{column: interior cut points}, learned from TRAINING ROWS ONLY. Same bin count in every scheme."""
    cuts = {}
    for c in BINNED:
        k = len(CLINICAL[c]) + 1
        x = train[c].to_numpy(float)
        if scheme == "clinical":
            cuts[c] = list(CLINICAL[c])
        elif scheme == "equal_width":
            cuts[c] = list(np.linspace(x.min(), x.max(), k + 1)[1:-1])
        elif scheme == "supervised":        # entropy-driven cut points (CART on one variable)
            t = DecisionTreeClassifier(criterion="entropy", max_leaf_nodes=k,
                                       min_samples_leaf=0.02, random_state=0)
            t.fit(x.reshape(-1, 1), train[target])
            cuts[c] = sorted(t.tree_.threshold[t.tree_.feature >= 0].tolist())
        else:
            raise ValueError(scheme)
    return cuts


def apply_cuts(df, cuts):
    """Replace each binned column by its bin index 0..k-1; bins are [a, b)."""
    out = df.copy()
    for c, cp in cuts.items():
        out[c] = np.searchsorted(np.asarray(cp, float), out[c].to_numpy(float), side="right")
    return out
