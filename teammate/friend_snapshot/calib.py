"""calib.py - probability calibration fitted on the validation split, evaluated by cross-fitting."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class Calibrator:
    """method = 'platt' (logistic curve on the logit), 'isotonic' (monotone steps) or 'none'."""

    def __init__(self, method="platt"):
        self.method = method

    def fit(self, p, y):
        if self.method == "platt":
            self.m = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p)[:, None], y)
        elif self.method == "isotonic":
            self.m = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
        return self

    def predict(self, p):
        if self.method == "platt":
            return self.m.predict_proba(_logit(p)[:, None])[:, 1]
        if self.method == "isotonic":
            return self.m.predict(np.asarray(p, float))
        return np.asarray(p, float)


def crossfit(p, y, method, seed=0):
    """Calibrated value for every validation row from a calibrator that never saw that row
    (2 folds). Use these to REPORT calibration; fit the deployed calibrator on all of validation."""
    p, y, out = np.asarray(p, float), np.asarray(y), np.empty(len(p))
    for a, b in StratifiedKFold(n_splits=2, shuffle=True, random_state=seed).split(p[:, None], y):
        out[b] = Calibrator(method).fit(p[a], y[a]).predict(p[b])
    return out
