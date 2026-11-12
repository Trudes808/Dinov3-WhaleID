"""Does synthetic data help the view-class probes on REAL photos?   python vc_synth_test.py [feature set: base | sq]
Day-grouped 8-fold cross-validation on the real labels (train + val days); the synthetic shots (vc_synth.py: exact
labels from the render metadata) are only ever added to the training side. Variants:
  real only | synthetic only | real + synthetic at sample weight w | the same with each domain centred on its own mean
  (real mean from all real instances, labelled or not) | real + only the synthetic "neither" shots (pectoral fins,
  breaches, spy-hops, blows: the class with the fewest real labels)
and a learning curve: the same with only N real labels per fold."""
import sys, warnings, pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from vc_common import *
warnings.filterwarnings('ignore')
fs = sys.argv[1] if len(sys.argv) > 1 else 'base'

T = all_targets(); days = {d: load_day(d) for d in DAYS if split_of(d) != 'eval'}
keys = [k for d in days for k in days[d]['keys']]; kid = {k: i for i, k in enumerate(keys)}; day_of = np.array([d for d in days for _ in days[d]['keys']])
if fs == 'base':
    X = np.vstack([days[d]['X'] for d in days])
else:
    parts = []
    for d in days:
        z = np.load(f'{HERE}/feats/{d}.npz'); zi = {str(k): i for i, k in enumerate(z['keys'])}; idx = [zi[k] for k in days[d]['keys']]
        parts.append(np.hstack([unit(z[n][idx]) for n in ('sq_cls', 'sq_mean', 'sq_max')]))
    X = np.vstack(parts)
z = np.load(f'{HERE}/feats/synth.npz'); Ys = z['y'].astype(np.int8); cat = z['cat']
Xs = np.hstack([unit(z[n]) for n in (('b_cls', 'b_mean') if fs == 'base' else ('sq_cls', 'sq_mean', 'sq_max'))])
Y = np.full((len(keys), 2), -9, np.int8); rnd = np.zeros(len(keys), bool)
for k, v in T.items():
    if k in kid: Y[kid[k]] = v['y']; rnd[kid[k]] = v['src'].get('round') == 'r0'
lab = (Y != -9).any(1); fit_days = np.array(sorted(set(day_of))); np.random.default_rng(0).shuffle(fit_days); folds = np.array_split(fit_days, 8)
mu_r, mu_s = X.mean(0), Xs.mean(0)
print(f'features {fs}: real {X.shape}, labelled {int(lab.sum())}; synthetic {Xs.shape}; cosine between the two domain means {float(mu_r @ mu_s / np.linalg.norm(mu_r) / np.linalg.norm(mu_s)):.3f}')


def run(mode, w=0.3, centre=False, n_real=None, seed=0, neither_only=False):
    S = np.full((len(keys), 2), np.nan, np.float32); rng = np.random.default_rng(seed)
    A_r = X - mu_r if centre else X; A_s = Xs - mu_s if centre else Xs
    for f in folds:
        te = np.isin(day_of, f) & lab
        for j in range(2):
            m = np.nonzero(~np.isin(day_of, f) & (Y[:, j] >= 0))[0]
            if n_real is not None and len(m) > n_real: m = rng.choice(m, n_real, replace=False)
            ms = Ys[:, j] >= 0
            if neither_only: ms &= (Ys == 0).all(1)
            if mode == 'real': A, y, sw = A_r[m], Y[m, j], np.ones(len(m))
            elif mode == 'synth': A, y, sw = A_s[ms], Ys[ms, j], np.ones(int(ms.sum()))
            else: A, y, sw = np.vstack([A_r[m], A_s[ms]]), np.concatenate([Y[m, j], Ys[ms, j]]), np.concatenate([np.ones(len(m)), np.full(int(ms.sum()), w * len(m) / max(ms.sum(), 1))])
            if len(set(y.tolist())) < 2: continue
            sc = StandardScaler().fit(A_r[m] if mode != 'synth' and len(m) > 50 else A); clf = LogisticRegression(C=0.5 / A.shape[1] * 100, class_weight='balanced', max_iter=3000).fit(sc.transform(A), y, sample_weight=sw)
            S[te, j] = clf.decision_function(sc.transform(A_r[te]))
    return S


def report(name, S):
    out = [f'{name:46s}']
    for j in range(2):
        m = lab & (Y[:, j] >= 0) & np.isfinite(S[:, j]); out.append(f'{roc_auc_score(Y[m, j], S[m, j]):.4f}')
    for sub in (lab, lab & rnd):
        m = sub & (Y >= 0).all(1) & np.isfinite(S).all(1); true = np.where(Y[m, 0] == 1, 0, np.where(Y[m, 1] == 1, 1, 2)); pred = np.where(S[m, 0] > 0, 0, np.where(S[m, 1] > 0, 1, 2)); cm = np.zeros((3, 3), int)
        for t, p in zip(true, pred): cm[t, p] += 1
        out.append(f'{np.trace(cm) / cm.sum():.3f} F{cm[0].tolist()} D{cm[1].tolist()} O{cm[2].tolist()}')
    print(' | '.join(out), flush=True)


print('variant | AUC fluke | AUC dorsal | all sources: accuracy, rows true F / D / O -> predicted [F, D, O] | random sample r0: same')
report('real only', run('real'))
report('synthetic only', run('synth'))
report('synthetic only, domains centred', run('synth', centre=True))
for w in (0.1, 0.3, 1.0):
    report(f'real + synthetic, total synthetic weight {w} x real', run('both', w=w))
report('real + synthetic 0.3, domains centred', run('both', w=0.3, centre=True))
report('real + synthetic "neither" shots only, 0.3', run('both', w=0.3, neither_only=True))
print('\nlearning curve: N real labels per probe (mean of 3 draws)')
for n in (50, 100, 200, 400, 800):
    for name, kw in (('real only', dict(mode='real')), ('real + synthetic 0.3', dict(mode='both', w=0.3)), ('real + synthetic 0.3, centred', dict(mode='both', w=0.3, centre=True))):
        acc = []
        for seed in range(3):
            S = run(n_real=n, seed=seed, **kw); r = []
            for sub in (lab, lab & rnd):
                m = sub & (Y >= 0).all(1) & np.isfinite(S).all(1); true = np.where(Y[m, 0] == 1, 0, np.where(Y[m, 1] == 1, 1, 2)); pred = np.where(S[m, 0] > 0, 0, np.where(S[m, 1] > 0, 1, 2)); r.append(float((true == pred).mean()))
            acc.append(r)
        a = np.mean(acc, 0); print(f'   N {n:4d}  {name:32s} all sources {a[0]:.3f}  random sample {a[1]:.3f}', flush=True)
