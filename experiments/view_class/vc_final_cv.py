"""The final recipe under day-grouped cross-validation, with and without synthetic shots:
    python vc_final_cv.py
Features: frozen DINOv3 ViT-B/16 cls + mean patch token of two crops per instance (448-px square; shape-keeping, long
side 672) = 3072 numbers; standardised, PCA 128, RBF SVM (C 4, balanced classes), one for the fluke, one for the fin.
Synthetic: a stratified subsample of the run04 + run05 shots, each domain centred on its own mean, total weight w x real."""
import sys, warnings, pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from vc_common import *
warnings.filterwarnings('ignore')
COLS = ('sq_cls', 'sq_mean', 'ar_cls', 'ar_mean')
T = all_targets(); days = {d: load_day(d, feats=False) for d in DAYS if split_of(d) != 'eval'}
keys = [k for d in days for k in days[d]['keys']]; kid = {k: i for i, k in enumerate(keys)}; day_of = np.array([d for d in days for _ in days[d]['keys']]); parts = []
for d in days:
    z = np.load(f'{HERE}/feats/{d}.npz'); zi = {str(k): i for i, k in enumerate(z['keys'])}; idx = [zi[k] for k in days[d]['keys']]
    parts.append(np.hstack([unit(z[n][idx]) for n in COLS]))
X = np.vstack(parts); z = np.load(f'{HERE}/feats/synth.npz'); Ys = z['y'].astype(np.int8); cat = z['cat']; Xs = np.hstack([unit(z[n]) for n in COLS])
Y = np.full((len(keys), 2), -9, np.int8); rnd = np.zeros(len(keys), bool)
for k, v in T.items():
    if k in kid: Y[kid[k]] = v['y']; rnd[kid[k]] = v['src'].get('round') == 'r0'
lab = (Y != -9).any(1); fit_days = np.array(sorted(set(day_of))); np.random.default_rng(0).shuffle(fit_days); folds = np.array_split(fit_days, 8)
mu_r, mu_s = X.mean(0), Xs.mean(0); rng = np.random.default_rng(0)
sub = np.concatenate([rng.choice(np.nonzero(cat == c)[0], min(int((cat == c).sum()), 300), replace=False) for c in sorted(set(cat.tolist()))])        # <= 300 per pose category
print(f'real {X.shape}, labelled {int(lab.sum())}; synthetic subsample {len(sub)} of {len(Xs)}')


def run(synth=0.0, centre=True, seed=0):
    S = np.full((len(keys), 2), np.nan, np.float32); A_r = X - mu_r if centre else X; A_s = (Xs - mu_s if centre else Xs)[sub]
    for f in folds:
        te = np.isin(day_of, f) & lab
        for j in range(2):
            m = ~np.isin(day_of, f) & (Y[:, j] >= 0); ms = Ys[sub, j] >= 0
            A, y, sw = A_r[m], Y[m, j], np.ones(int(m.sum()))
            if synth: A, y, sw = np.vstack([A, A_s[ms]]), np.concatenate([y, Ys[sub][ms, j]]), np.concatenate([sw, np.full(int(ms.sum()), synth * m.sum() / ms.sum())])
            sc = StandardScaler().fit(A_r[m]); p = PCA(128, random_state=0).fit(sc.transform(A_r[m]))
            clf = SVC(C=4.0, kernel='rbf', gamma='scale', class_weight='balanced').fit(p.transform(sc.transform(A)), y, sample_weight=sw)
            S[te, j] = clf.decision_function(p.transform(sc.transform(A_r[te])))
    return S


def report(name, S):
    out = [f'{name:40s}']
    for j in range(2):
        m = lab & (Y[:, j] >= 0); out.append(f'{roc_auc_score(Y[m, j], S[m, j]):.4f}')
    for subm in (lab, lab & rnd):
        m = subm & (Y >= 0).all(1); true = np.where(Y[m, 0] == 1, 0, np.where(Y[m, 1] == 1, 1, 2)); pred = np.where(S[m, 0] > 0, 0, np.where(S[m, 1] > 0, 1, 2)); cm = np.zeros((3, 3), int)
        for t, p in zip(true, pred): cm[t, p] += 1
        out.append(f'{np.trace(cm) / cm.sum():.3f} F{cm[0].tolist()} D{cm[1].tolist()} O{cm[2].tolist()}')
    print(' | '.join(out), flush=True)


res = {}
for name, kw in (('real only', dict(synth=0.0, centre=False)), ('real only, centred (same thing)', dict(synth=0.0)), ('real + synthetic 0.1', dict(synth=0.1)), ('real + synthetic 0.3', dict(synth=0.3)), ('real + synthetic 1.0', dict(synth=1.0)),
                 ('real + synthetic 0.3, not centred', dict(synth=0.3, centre=False))):
    res[name] = run(**kw); report(name, res[name])
pickle.dump({'keys': keys, 'res': res, 'Y': Y, 'rnd': rnd, 'lab': lab, 'day': day_of}, open(f'{HERE}/final_cv_scores.pkl', 'wb'))
