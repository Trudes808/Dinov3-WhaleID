"""Does the where-to-look window score help the Photo Labeling page's pair ranking? Leave-one-day-out over the val days
(the page's logistic is fitted on the val days): the page's ten features against the same plus the window score
(max D_sal over the instance pairs of two stacks, and whether it exists). Measure = the page's own: the rank of the
true animal when the stacks of an encounter are sorted one by one (label_assist.simulate, no runs). Read-only."""
import sys, json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, '/home/sat3737/NOAA_whales/Dinov3-WhaleID')
from model_fine_tuning.cluster import label_assist as LA
D0 = Path('/home/sat3737/NOAA_whales/Dinov3-WhaleID/experiments/days_all'); VAL = ["2025-01-15", "2025-02-27", "2025-03-14"]
def window_matrix(day, D):
    ws = np.load(D0 / day / 'safe' / 'window_scores.npz'); kid = {str(k): i for i, k in enumerate(ws['keys'])}; S = ws['D_sal'].astype(np.float32); n = len(D['clusters']); W = np.full((n, n), np.nan, np.float32)
    idx = [[kid[k] for k in c['keys'] if k in kid] for c in D['clusters']]
    for i in range(n):
        for j in range(i + 1, n):
            if idx[i] and idx[j]:
                blk = S[np.ix_(idx[i], idx[j])]
                if np.isfinite(blk).any(): W[i, j] = W[j, i] = np.nanmax(blk)
    return W
data = {d: LA.load_stacks(D0 / d) for d in VAL}; F = {}; Fx = {}
for d, D in data.items():
    f = LA.pair_features(D); W = window_matrix(d, D); has = np.isfinite(W); F[d] = f; Fx[d] = np.concatenate([f, np.where(has, np.nan_to_num(W), 0.0)[..., None], has[..., None].astype(np.float32)], -1)
def training(D, feats):
    X, y = [], []
    for sc in LA.scopes(D).values():
        for a in range(len(sc)):
            for b in range(a + 1, len(sc)): X.append(feats[sc[a], sc[b]]); y.append(D['true'][sc[a]] == D['true'][sc[b]])
    return np.asarray(X, np.float32), np.asarray(y, bool)
def run(feats, label):
    tot = {}
    for held in VAL:
        X = np.vstack([training(data[d], feats[d])[0] for d in VAL if d != held]); y = np.concatenate([training(data[d], feats[d])[1] for d in VAL if d != held])
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(X, y)
        P = 1.0 / (1.0 + np.exp(-(feats[held] @ clf.coef_[0].astype(np.float32) + clf.intercept_[0]))); np.fill_diagonal(P, 0.0)
        t = LA.simulate(data[held], P, 1.5); tot[held] = t
    ex = sum(t['existing'] for t in tot.values())
    print(f"{label:44s} first choice {sum(t['first'] for t in tot.values()) / ex:.3f}  top three {sum(t['top3'] for t in tot.values()) / ex:.3f}  (decisions with an existing animal {ex}) | per day first choice {[round(tot[d]['first'] / max(tot[d]['existing'], 1), 3) for d in VAL]}")
print('fit settings of label_assist:'); import inspect; src = inspect.getsource(LA.main); i = src.find("if a.cmd == \"fit\""); print('   ', [l.strip()[:200] for l in src.splitlines() if 'LogisticRegression(' in l])
run(F, "the page's ten features (held-out val day)"); run(Fx, 'plus the where-to-look window score')
Fw = {d: np.concatenate([F[d][..., :2], Fx[d][..., 10:], F[d][..., 8:10]], -1) for d in VAL}; run(Fw, 'sequence P + window + gap (no student)')
