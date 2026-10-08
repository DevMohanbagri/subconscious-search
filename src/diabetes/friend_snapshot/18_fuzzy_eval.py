"""18 - Evaluate the fuzzy risk tiers (Design A) and refine the rule base on validation.
Rules are checked on validation fold A and the refined base is confirmed on fold B."""
import json
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from fuzzy import MamdaniFIS, load_rules, modifiable_score, TIERS
from metrics import wilson_ci

v = pd.read_csv("results/preds/val_final.csv")
y, P = v.y_true.to_numpy(), v.p_cal.to_numpy()
X = {"P": P, "BMI": v.BMI.to_numpy(), "Age": v.Age.to_numpy(),
     "M": modifiable_score(v.BMI, v.PhysActivity, v.Fruits, v.Veggies),
     "HighBP": v.HighBP.to_numpy(), "HighChol": v.HighChol.to_numpy(),
     "HeartDiseaseorAttack": v.HeartDiseaseorAttack.to_numpy()}
fold_a, fold_b = next(StratifiedKFold(2, shuffle=True, random_state=0).split(P[:, None], y))


def evaluate(rules, idx, label):
    fis = MamdaniFIS(rules)
    score, W = fis.infer({k: a[idx] for k, a in X.items()})
    t, yy, pp = fis.tier(score), y[idx], P[idx]
    obs = [yy[t == i].mean() for i in range(4)]
    k = int(0.2 * len(yy))
    cap_f = yy[np.lexsort((-pp, -score))[:k]].sum() / yy.sum()
    cap_p = yy[np.argsort(-pp)[:k]].sum() / yy.sum()
    print(f"{label}: tier prevalence {np.round(obs, 3)} shares {np.round(np.bincount(t, minlength=4) / len(t), 3)} "
          f"monotone={bool(np.all(np.diff(obs) > 0))} | catch@20% fuzzy {cap_f:.3f} vs P {cap_p:.3f} | "
          f"PR-AUC fuzzy {average_precision_score(yy, score + 1e-6 * pp):.4f} vs P {average_precision_score(yy, pp):.4f}")
    return fis, score, W, t


def check_rules(rules, idx, min_n=200):
    """For each non-backbone rule: do the people it fires for (>= 0.5) differ from what the model
    expected for them (O/E = observed rate / mean P), in the direction the rule claims?"""
    fis = MamdaniFIS(rules)
    _, W = fis.infer({k: a[idx] for k, a in X.items()})
    rank = {t: i for i, t in enumerate(TIERS)}
    out = []
    for i, r in enumerate(rules):
        if r["source"] == "backbone":
            continue
        m = W[:, i] >= 0.5
        n = int(m.sum())
        if n == 0:
            out.append({"rule": r["id"], "n": 0, "keep": False, "why": "never fires"}); continue
        e = P[idx][m].mean(); lo, hi = wilson_ci(y[idx][m].sum(), n)
        backbone_tier = max(rank[rr["then"]] for rr in rules if rr["source"] == "backbone"
                            and any(c in r["if"] for c in rr["if"]))
        up = rank[r["then"]] > backbone_tier                   # escalation or de-escalation?
        ok = n >= min_n and ((lo / e > 1) if up else (hi / e < 1))
        out.append({"rule": r["id"], "direction": "up" if up else "down", "n": n,
                    "O/E": round(y[idx][m].mean() / e, 2), "ci": f"{lo / e:.2f}-{hi / e:.2f}", "keep": ok,
                    "why": "supported" if ok else ("too few people" if n < min_n else "data disagree")})
    return pd.DataFrame(out)


v1 = load_rules("fuzzy_rules.json")
evaluate(v1, np.arange(len(y)), "v1 all validation")
chk = check_rules(v1, fold_a)
print(chk.to_string(index=False))
keep = set(chk.loc[chk.keep, "rule"])
v2 = [r for r in v1 if r["source"] == "backbone" or r["id"] in keep]
json.dump([{**r, "if": [list(c) for c in r["if"]]} for r in v2], open("fuzzy_rules_v2.json", "w"), indent=1)
print("kept:", sorted(keep))
evaluate(v1, fold_b, "v1 fold B")
fis, score, W, t = evaluate(v2, fold_b, "v2 fold B")

rows = []
for i, name in enumerate(TIERS):
    m = t == i; lo, hi = wilson_ci(y[fold_b][m].sum(), m.sum())
    rows.append({"tier": name, "share": m.mean(), "observed": y[fold_b][m].mean(), "lo": lo, "hi": hi})
tb = pd.DataFrame(rows); tb.round(4).to_csv("results/fuzzy_tiers_foldB.csv", index=False)
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(tb.tier, tb.observed, yerr=[tb.observed - tb.lo, tb.hi - tb.observed], capsize=4)
for i, r in tb.iterrows():
    ax.text(i, r.hi + 0.01, f"{r.share:.0%} of people", ha="center", fontsize=8)
ax.axhline(y.mean(), ls=":", color="0.5"); ax.set_ylabel("Observed diabetes prevalence (validation fold B)")
fig.tight_layout(); fig.savefig("figures/fig_fuzzy_tiers.png", dpi=300)
busy = int(np.argmax((W >= 0.2).sum(axis=1)))
print("example trace:", fis.trace(W[busy], top=4))
