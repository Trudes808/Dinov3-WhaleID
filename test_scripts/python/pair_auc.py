"""How much identity information does each feature source carry? Pair-level AUC of the
appearance score for same-individual vs different-individual pairs (labels used for scoring
only), overall and within the hard subsets: pairs inside the same group (interleaved in time)
and pairs from the competition group.

    .venv/bin/python test_scripts/python/pair_auc.py SCORES_NPZ LABELS_JSON [--min-cells N]
"""
import sys, json, numpy as np
from sklearn.metrics import roc_auc_score
z = dict(np.load(sys.argv[1])); stems = [str(s) for s in z["stems"]]
lab = json.loads(open(sys.argv[2]).read())["labels"]; gt = np.array([lab[s][0] for s in stems]); grp = np.array([g[:3] for g in gt])
n = len(stems); iu = np.triu_indices(n, 1)
same = (gt[:, None] == gt[None, :])[iu]; same_grp = (grp[:, None] == grp[None, :])[iu]; dt = z["dt_s"][iu]
sources = sorted({k.split("__")[0] for k in z if "__same" in k})
print(f"{'source':12s} {'hyp':6s} {'pairs':>7s} {'AUC all':>8s} {'AUC same group':>15s} {'AUC comp. group G02':>20s} {'AUC dt>10min':>13s} {'cells>=20':>10s}")
for src in sources:
    for hyp in ("same", "best"):
        a = z[f"{src}__same"][iu]
        if hyp == "best":
            f = z[f"{src}__flip"][iu]; a = np.where(np.isnan(a), f, np.where(np.isnan(f), a, np.maximum(a, f)))
        ok = np.isfinite(a)
        def auc(mask):
            m = ok & mask
            return roc_auc_score(same[m], a[m]) if m.sum() > 20 and same[m].any() and (~same[m]).any() else float("nan")
        comp = ((grp == "G02")[:, None] & (grp == "G02")[None, :])[iu]
        big = (z[f"{src}__n_same"][iu] >= 20)
        print(f"{src:12s} {hyp:6s} {ok.sum():7d} {auc(np.ones_like(ok)):8.3f} {auc(same_grp):15.3f} {auc(comp):20.3f} {auc(dt > 600):13.3f} {auc(big):10.3f}")
t = z["time_term"][iu]; ok = np.ones_like(same)
print(f"{'time term':12s} {'':6s} {len(t):7d} {roc_auc_score(same, t):8.3f} {roc_auc_score(same[same_grp], t[same_grp]):15.3f} {roc_auc_score(same[comp], t[comp]) if comp.sum() else float('nan'):20.3f}")
