"""Second comparison: non-linear heads on the richer features (same day-grouped CV as vc_compare.py)."""
import pickle, sys, warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from vc_common import *
warnings.filterwarnings('ignore')
T = all_targets(); days = {d: load_day(d) for d in DAYS if split_of(d) != 'eval'}
keys = [k for d in days for k in days[d]['keys']]; kid = {k: i for i, k in enumerate(keys)}; day_of = np.array([d for d in days for _ in days[d]['keys']])
base = np.vstack([days[d]['X'] for d in days]); F = {}
for d in days:
    z = np.load(f'{HERE}/feats/{d}.npz'); zi = {str(k): i for i, k in enumerate(z['keys'])}; idx = [zi[k] for k in days[d]['keys']]
    for n in z.files:
        if n != 'keys': F.setdefault(n, []).append(unit(z[n][idx]))
F = {n: np.vstack(v) for n, v in F.items()}
Y = np.full((len(keys), 2), -9, np.int8); rnd = np.zeros(len(keys), bool)
for k, v in T.items():
    if k in kid: Y[kid[k]] = v['y']; rnd[kid[k]] = v['src'].get('round') == 'r0'
lab = (Y != -9).any(1); fit_days = np.array(sorted(set(day_of))); np.random.default_rng(0).shuffle(fit_days); folds = np.array_split(fit_days, 8)


def cv(X, C=4.0, dim=128, gamma='scale'):
    S = np.full((len(keys), 2), np.nan, np.float32)
    for f in folds:
        te = np.isin(day_of, f) & lab
        for j in range(2):
            m = ~np.isin(day_of, f) & (Y[:, j] >= 0); sc = StandardScaler().fit(X[m]); A = sc.transform(X[m]); B = sc.transform(X[te])
            p = PCA(dim, random_state=0).fit(A); S[te, j] = SVC(C=C, kernel='rbf', gamma=gamma, class_weight='balanced').fit(p.transform(A), Y[m, j]).decision_function(p.transform(B))
    return S


def report(name, S):
    out = [f'{name:44s}']
    for j in range(2):
        m = lab & (Y[:, j] >= 0); out.append(f'{roc_auc_score(Y[m, j], S[m, j]):.4f}')
    for sub in (lab, lab & rnd):
        m = sub & (Y >= 0).all(1); true = np.where(Y[m, 0] == 1, 0, np.where(Y[m, 1] == 1, 1, 2)); pred = np.where(S[m, 0] > 0, 0, np.where(S[m, 1] > 0, 1, 2)); cm = np.zeros((3, 3), int)
        for t, p in zip(true, pred): cm[t, p] += 1
        out.append(f'{np.trace(cm) / cm.sum():.3f} F{cm[0].tolist()} D{cm[1].tolist()} O{cm[2].tolist()}')
    print(' | '.join(out), flush=True)


V = {'sq+ar cls+mean': ['sq_cls', 'sq_mean', 'ar_cls', 'ar_mean'], 'sq cls+mean+max': ['sq_cls', 'sq_mean', 'sq_max'], 'sq+ar cls+mean+max': ['sq_cls', 'sq_mean', 'sq_max', 'ar_cls', 'ar_mean', 'ar_max'],
     'ar cls+mean+max': ['ar_cls', 'ar_mean', 'ar_max']}
res = {}
for name, cols in V.items():
    X = np.hstack([F[c] for c in cols])
    for C, dim in ((4.0, 128), (10.0, 128), (4.0, 256), (1.5, 128)):
        if name != 'sq+ar cls+mean' and (C, dim) != (4.0, 128): continue
        S = cv(X, C=C, dim=dim); res[(name, C, dim)] = S; report(f'{name} | svm C {C} pca {dim}', S)
X = np.hstack([base, F['sq_cls'], F['sq_mean'], F['ar_cls'], F['ar_mean']]); S = cv(X); res[('base+sq+ar', 4.0, 128)] = S; report('base224+sq+ar cls+mean | svm C 4 pca 128', S)
pickle.dump({'keys': keys, 'res': res, 'Y': Y, 'rnd': rnd, 'lab': lab}, open(f'{HERE}/compare2_scores.pkl', 'wb'))
