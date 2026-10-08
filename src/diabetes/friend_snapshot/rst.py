"""rst.py - rough-set utilities. One verified implementation, imported everywhere."""
import numpy as np
import pandas as pd

EPS = 1e-12  # tolerance so float noise never flips a purity >= beta comparison


def gamma_vprs(df, B, target, beta=1.0):
    """Reference implementation (slow, obviously correct). beta=1.0 is classical Pawlak."""
    if not B:
        return 0.0
    g = df.groupby(list(B))[target].agg(["size", "sum"])
    p = g["sum"] / g["size"]
    purity = np.maximum(p, 1 - p)
    return g["size"][purity >= beta - EPS].sum() / len(df)


def encode(df, cols):
    """Integer-code each column as 0..k-1. Returns {col: (codes, k)}."""
    enc = {}
    for c in cols:
        codes, uniques = pd.factorize(df[c], sort=True)
        enc[c] = (codes.astype(np.int64), len(uniques))
    return enc


def granules(enc, B, n):
    """Compact granule id (0..G-1) for every row, under attribute subset B."""
    gid = np.zeros(n, dtype=np.int64)
    for a in B:
        codes, k = enc[a]
        _, gid = np.unique(gid * k + codes, return_inverse=True)
    return gid.ravel()


def class_stats(gid, y, beta=1.0):
    """Pooled and per-class beta-lower approximation statistics. y is a 0/1 float array."""
    size = np.bincount(gid)
    pos = np.bincount(gid, weights=y)
    p = pos / size
    in_pos = p >= beta - EPS          # granule belongs to beta-lower approx of D=1
    in_neg = (1 - p) >= beta - EPS    # granule belongs to beta-lower approx of D=0
    n, n_pos = len(y), y.sum()
    return {
        "gamma": size[in_pos | in_neg].sum() / n,
        "rows_pos_lower": size[in_pos].sum() / n,
        "rows_neg_lower": size[in_neg].sum() / n,
        "pos_captured": pos[in_pos].sum() / n_pos,   # share of real positives in certain-positive granules
        "n_granules": len(size),
        "singleton_share": float((size == 1).mean()),
    }


def gamma_fast(gid, y, beta=1.0):
    return class_stats(gid, y, beta)["gamma"]


def quickreduct(enc, y, attrs, beta=1.0, tol=1e-9):
    """Greedy forward selection. Ties go to the earlier attribute in `attrs`.
    Returns (reduct, gamma_reduct, gamma_full, path); path rows = (attr, gamma, n_tied)."""
    n = len(y)
    full = gamma_fast(granules(enc, attrs, n), y, beta)
    R, cur, path = [], 0.0, []
    gid_R = np.zeros(n, dtype=np.int64)
    while cur < full - tol:
        scores = {}
        for a in attrs:
            if a in R:
                continue
            codes, k = enc[a]
            _, gid = np.unique(gid_R * k + codes, return_inverse=True)
            scores[a] = (gamma_fast(gid.ravel(), y, beta), gid.ravel())
        best_g = max(s[0] for s in scores.values())
        if best_g <= cur + tol:
            break                          # nothing improves gamma: stop
        tied = [a for a in attrs if a in scores and scores[a][0] >= best_g - tol]
        best = tied[0]
        R.append(best)
        cur, gid_R = scores[best]
        path.append((best, cur, len(tied)))
    return R, cur, full, path


def gamma_null(enc, y, B, beta=1.0, n_perm=5, seed=0):
    """Gamma of subset B when labels are randomly permuted: the chance level of 'certainty'."""
    rng = np.random.default_rng(seed)
    gid = granules(enc, B, len(y))
    return np.array([gamma_fast(gid, rng.permutation(y), beta) for _ in range(n_perm)])


def heldout_logloss(gid, y, folds, a=20.0):
    """Mean out-of-fold log-loss of the granule model: each granule predicts its training-fold
    positive rate, shrunk toward the fold prevalence by `a` pseudo-counts (unseen granule = prior)."""
    G = int(gid.max()) + 1
    total = 0.0
    for k in range(int(folds.max()) + 1):
        fit, hold = folds != k, folds == k
        n = np.bincount(gid[fit], minlength=G)
        pos = np.bincount(gid[fit], weights=y[fit], minlength=G)
        p0 = y[fit].mean()
        ph = np.clip(((pos + a * p0) / (n + a))[gid[hold]], 1e-9, 1 - 1e-9)
        total -= np.sum(y[hold] * np.log(ph) + (1 - y[hold]) * np.log(1 - ph))
    return total / len(y)


def heldout_reduct(enc, y, attrs, a=20.0, n_folds=5, seed=0, tol=1e-6):
    """Greedy forward selection minimising held-out conditional entropy of the decision given
    the granules. Over-fine granulations lose out of sample, so the search stops by itself.
    Returns (reduct, path, gamma_h) with gamma_h = share of label uncertainty removed out of sample."""
    folds = np.random.default_rng(seed).permutation(len(y)) % n_folds
    gid_R = np.zeros(len(y), dtype=np.int64)
    prior = cur = heldout_logloss(gid_R, y, folds, a)
    R, path = [], []
    while True:
        best = None
        for c in attrs:
            if c in R:
                continue
            codes, k = enc[c]
            _, gid = np.unique(gid_R * k + codes, return_inverse=True)
            v = heldout_logloss(gid.ravel(), y, folds, a)
            if best is None or v < best[1]:
                best = (c, v, gid.ravel())
        if best is None or best[1] > cur - tol:
            break
        R.append(best[0]); cur, gid_R = best[1], best[2]
        path.append((best[0], round(1 - cur / prior, 4)))
    return R, path, 1 - cur / prior


# ---------- stability of a selection procedure ----------

def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 1.0


def mean_pairwise_jaccard(subsets):
    s = [set(x) for x in subsets]
    vals = [jaccard(s[i], s[j]) for i in range(len(s)) for j in range(i + 1, len(s))]
    return float(np.mean(vals)) if vals else 1.0


def nogueira_stability(subsets, all_attrs):
    """Nogueira et al. (2018) stability: 1 = identical subsets every run, about 0 = chance level.
    Handles subsets of different sizes."""
    M, d = len(subsets), len(all_attrs)
    Z = np.array([[a in set(s) for a in all_attrs] for s in subsets], dtype=float)
    p = Z.mean(axis=0)
    k_bar = Z.sum(axis=1).mean()
    denom = (k_bar / d) * (1 - k_bar / d)
    if M < 2 or denom == 0:
        return float("nan")
    s2 = M / (M - 1) * p * (1 - p)
    return float(1 - s2.mean() / denom)


# ---------- decision rules from granules ----------

def rule_stats(df, conds, target):
    """Support n, positive rate and lift of the rule 'IF every condition holds'."""
    m = np.ones(len(df), dtype=bool)
    for col, val in conds.items():
        m &= df[col].to_numpy() == val
    n = int(m.sum())
    if n == 0:
        return 0, float("nan"), float("nan")
    rate = float(df[target].to_numpy()[m].mean())
    return n, rate, rate / float(df[target].mean())


def simplify_rule(df, conds, target, ok):
    """Value reduction: drop conditions one at a time while ok(n, rate, lift) stays True,
    preferring the drop that keeps the largest support."""
    conds = dict(conds)
    while len(conds) > 1:
        best = None
        for col in list(conds):
            trial = {k: v for k, v in conds.items() if k != col}
            n, rate, lift = rule_stats(df, trial, target)
            if ok(n, rate, lift) and (best is None or n > best[1]):
                best = (col, n)
        if best is None:
            break
        del conds[best[0]]
    return conds


def extract_rules(df, attrs, target, min_support=300, hi_lift=2.0, lo_lift=0.5):
    """Granules of `attrs` that are clearly high-risk (lift >= hi_lift) or low-risk (lift <= lo_lift),
    each simplified, then de-duplicated. Returns a list of dicts sorted by lift."""
    g = df.groupby(list(attrs))[target].agg(n="size", pos="sum").reset_index()
    prev = float(df[target].mean())
    g["lift"] = g["pos"] / g["n"] / prev
    rules, seen = [], set()
    for _, row in g[g["n"] >= min_support].iterrows():
        if row["lift"] >= hi_lift:
            kind, ok = "high", (lambda n, r, l: n >= min_support and l >= hi_lift)
        elif row["lift"] <= lo_lift:
            kind, ok = "low", (lambda n, r, l: n >= min_support and l <= lo_lift)
        else:
            continue
        conds = simplify_rule(df, {a: int(row[a]) for a in attrs}, target, ok)
        key = (kind, tuple(sorted(conds.items())))
        if key in seen:
            continue
        seen.add(key)
        n, rate, lift = rule_stats(df, conds, target)
        rules.append({"kind": kind, "if": conds, "support": n, "rate": round(rate, 4), "lift": round(lift, 3)})
    return sorted(rules, key=lambda r: -r["lift"])
