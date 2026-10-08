"""metrics.py - every metric the project reports, defined once."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, precision_recall_curve


def recall_at_precision(y, p, target=0.30):
    """Highest recall reachable with precision >= target, and the threshold that gives it."""
    prec, rec, thr = precision_recall_curve(y, p)
    ok = prec[:-1] >= target
    if not ok.any():
        return 0.0, np.nan
    i = int(np.argmax(np.where(ok, rec[:-1], -1.0)))
    return float(rec[i]), float(thr[i])


def precision_at_recall(y, p, target=0.80):
    """Highest precision reachable with recall >= target, and its threshold."""
    prec, rec, thr = precision_recall_curve(y, p)
    ok = rec[:-1] >= target
    if not ok.any():
        return 0.0, np.nan
    i = int(np.argmax(np.where(ok, prec[:-1], -1.0)))
    return float(prec[i]), float(thr[i])


def at_threshold(y, p, t):
    """Flag everyone with p >= t (t chosen on validation, applied unchanged to test)."""
    y = np.asarray(y); flag = np.asarray(p) >= t; tp = int(np.sum(flag & (y == 1)))
    return {"precision": tp / max(int(flag.sum()), 1), "recall": tp / max(int((y == 1).sum()), 1),
            "flag_rate": float(flag.mean())}


def ece(y, p, n_bins=10):
    """Expected calibration error, equal-frequency bins (most predictions are small)."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        return float(abs(p.mean() - y.mean()))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    return float(sum((idx == b).mean() * abs(p[idx == b].mean() - y[idx == b].mean())
                     for b in range(len(edges) - 1) if (idx == b).any()))


def summarize(y, p):
    """The standard metric bundle; keys match registry columns."""
    y = np.asarray(y); p = np.asarray(p, float)
    return {"pr_auc": round(average_precision_score(y, p), 4),
            "roc_auc": round(roc_auc_score(y, p), 4),
            "brier": round(brier_score_loss(y, p), 4),
            "prevalence": round(float(y.mean()), 4),
            "accuracy": round(float(((p >= 0.5) == y).mean()), 4),
            "rec_at_p30": round(recall_at_precision(y, p, 0.30)[0], 4),
            "prec_at_r80": round(precision_at_recall(y, p, 0.80)[0], 4),
            "ece": round(ece(y, p), 4)}


def _strat_idx(rng, y):
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    return np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])


def bootstrap_ci(fn, y, p, n_boot=2000, seed=0):
    """Point estimate and 95% stratified-bootstrap interval of fn(y, p)."""
    rng = np.random.default_rng(seed); y = np.asarray(y); p = np.asarray(p)
    s = [fn(y[i], p[i]) for i in (_strat_idx(rng, y) for _ in range(n_boot))]
    return float(fn(y, p)), float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975))


def paired_bootstrap_delta(fn, y, p_a, p_b, n_boot=2000, seed=0):
    """fn(A) - fn(B) on identical resamples: estimate, 95% CI, approximate two-sided p."""
    rng = np.random.default_rng(seed); y = np.asarray(y); p_a = np.asarray(p_a); p_b = np.asarray(p_b)
    d = np.array([fn(y[i], p_a[i]) - fn(y[i], p_b[i]) for i in (_strat_idx(rng, y) for _ in range(n_boot))])
    return (float(fn(y, p_a) - fn(y, p_b)), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975)),
            float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))))


def wilson_ci(k, n, z=1.96):
    """95% interval for a proportion k/n (tier and subgroup prevalence)."""
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n; den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)
