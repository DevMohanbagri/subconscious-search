"""07 - Does the binning scheme matter downstream? LR on one-hot bins, plus a paired bootstrap."""
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import average_precision_score
from data_io import load, features, log_result, TARGET
from discretise import fit_cuts, apply_cuts, BINNED
from metrics import summarize, paired_bootstrap_delta

df, tr, va, _ = load()
feats, yva, preds = features(df), df.loc[va, TARGET].to_numpy(), {}
for scheme in ["clinical", "equal_width", "supervised"]:
    cuts = fit_cuts(df.loc[tr], scheme)                         # learned on training rows only
    dtr, dva = apply_cuts(df.loc[tr], cuts), apply_cuts(df.loc[va], cuts)
    ct = ColumnTransformer([("bins", OneHotEncoder(handle_unknown="ignore"), BINNED)], remainder=StandardScaler())
    m = make_pipeline(ct, LogisticRegression(max_iter=3000)).fit(dtr[feats], dtr[TARGET])
    preds[scheme] = m.predict_proba(dva[feats])[:, 1]
    cut_txt = str({c: [round(float(x), 2) for x in v] for c, v in cuts.items()})
    log_result(script="07_discretise_compare.py", feature_set=f"all_21_{scheme}_bins", n_features=len(feats),
               model="logistic regression (one-hot bins)", seed=0, split="val", calibrated="native",
               **summarize(yva, preds[scheme]), notes=cut_txt)
    print(f"{scheme:12s} PR-AUC {average_precision_score(yva, preds[scheme]):.4f}  cuts {cut_txt}")
d, lo, hi, p = paired_bootstrap_delta(average_precision_score, yva, preds["clinical"], preds["supervised"], n_boot=1000)
print(f"clinical minus supervised: {d:+.4f} (95% CI {lo:+.4f} to {hi:+.4f}, p = {p:.2f})")
