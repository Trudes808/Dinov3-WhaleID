"""Exp 3: a view-specific similarity from the frozen view features (the ones the classifier already computes).
For the instances the classifier says show a fluke (or a dorsal fin): centre per day, PCA, then scale the directions
by how much one animal varies along them on the TRAIN days (within-class covariance normalisation; LDA as a variant).
Tested on the val days: stack pairs inside catalogue encounters where both stacks show the part, score = best cosine
between their instances, same animal against different animals (AUC), next to the student's part cosine of the page.
    python vm_exp3.py fluke|dorsal [test split]"""
import sys, warnings, pickle
import numpy as np
from sklearn.metrics import roc_auc_score
from vm_common import *
warnings.filterwarnings('ignore')
part = sys.argv[1]; test = sys.argv[2] if len(sys.argv) > 2 else 'val'; THR = 0.5


def gather(day):
    d = load_day(day); F = view_feats(day); pv = d.pf if part == 'fluke' else d.pd; rows = []; per_stack = {}
    for s in range(d.n):
        if d.kind[s] == 'extra': continue
        idx = [j for j, p in enumerate(pv[s]) if p >= THR]
        if not idx: continue
        per_stack[s] = (len(rows), len(idx)); rows += [F[d.keys[s][j]] for j in idx]
    X = np.array(rows, np.float32) if rows else np.zeros((0, 3072), np.float32)
    lab = np.concatenate([[f'{day}|{d.cat[s]}' if d.cat[s] else None] * n for s, (o, n) in per_stack.items()]) if per_stack else np.array([])
    stack = np.concatenate([[s] * n for s, (o, n) in per_stack.items()]) if per_stack else np.array([])
    return d, X, lab, stack, per_stack


tr = [gather(day) for day in TRAIN]; te = [gather(day) for day in {'val': VAL, 'eval': EVAL, 'train': TRAIN}[test]]
Xtr = np.vstack([x - x.mean(0) for _, x, _, _, _ in tr if len(x)]); ytr = np.concatenate([l for _, x, l, _, _ in tr if len(x)]); keep = np.array([l is not None for l in ytr]); Xtr, ytr = Xtr[keep], ytr[keep]
dtr = np.array([l.split('|')[0] for l in ytr])
print(f'{part}: {len(Xtr)} train instances of {len(set(ytr))} day-animals on {len(TRAIN)} days')


def fit(dim, mode, shrink=0.1, blocks=slice(None)):
    X = Xtr[:, blocks]; mu = X.mean(0); U, S, Vt = np.linalg.svd(X - mu, full_matrices=False); Pm = Vt[:dim].T / (S[:dim] / np.sqrt(len(X)))      # whitened PCA
    Z = (X - mu) @ Pm; W = np.zeros((dim, dim)); n = 0
    cls = {}
    for i, l in enumerate(ytr): cls.setdefault(l, []).append(i)
    means = {}
    for l, idx in cls.items():
        if len(idx) < 2: continue
        z = Z[idx]; m = z.mean(0); means[l] = (m, len(idx)); W += (z - m).T @ (z - m); n += len(idx)
    W = W / n; W = (1 - shrink) * W + shrink * np.eye(dim) * np.trace(W) / dim
    if mode == 'pca': T = np.eye(dim)
    else:
        ev, E = np.linalg.eigh(W); T = E @ np.diag(ev ** -0.5) @ E.T                                    # W^-1/2
        if mode == 'lda':                                                                              # then the directions that separate the animals of one day
            Bm = np.zeros((dim, dim)); 
            for day in set(dtr):
                ms = [(m @ T, k) for l, (m, k) in means.items() if l.startswith(day)]
                if len(ms) < 2: continue
                c = np.average([m for m, _ in ms], axis=0, weights=[k for _, k in ms])
                for m, k in ms: Bm += k * np.outer(m - c, m - c)
            ev2, E2 = np.linalg.eigh(Bm); T = T @ E2[:, ::-1][:, :max(8, dim // 4)]
    return dict(mu=mu, P=Pm @ T, blocks=blocks)


def project(M, X):
    Z = (X[:, M['blocks']] - X[:, M['blocks']].mean(0) - 0 * M['mu']) @ M['P'] if len(X) else np.zeros((0, M['P'].shape[1]))
    return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)


def evaluate(M, name, show=True):
    ys, new, old, cov = [], [], [], collections.Counter()
    for d, X, lab, stack, per in te:
        Z = project(M, X) if M is not None else None
        for a, b, same in scope_pairs(d):
            if a not in per or b not in per: continue
            cov['pairs where both show the part'] += 1; f_old = (d.F if part == 'fluke' else d.Dn)[a, b]; cov['with a student cosine'] += f_old > 0
            (oa, na), (ob, nb) = per[a], per[b]; s = float((Z[oa:oa + na] @ Z[ob:ob + nb].T).max()) if Z is not None else 0.0
            ys.append(same); new.append(s); old.append(f_old)
    ys, new, old = np.array(ys), np.array(new), np.array(old); m = old > 0
    if M is None:
        print(f"   {'student part cosine (the page, gated)':46s} pairs {int(m.sum()):6d} of {len(ys)} (the rest have no chart)  AUC {roc_auc_score(ys[m], old[m]):.3f}"); return
    r = np.argsort(np.argsort(new[m])) + np.argsort(np.argsort(old[m]))
    print(f"   {name:46s} all {len(ys)} pairs AUC {roc_auc_score(ys, new):.3f} | where the student has a cosine: {roc_auc_score(ys[m], new[m]):.3f} | rank mean of both there {roc_auc_score(ys[m], r):.3f} | pairs without a chart ({int((~m).sum())}): {roc_auc_score(ys[~m], new[~m]) if (~m).sum() > 30 and ys[~m].min() != ys[~m].max() else float('nan'):.3f}")
    return ys, new, old


print(f'test = {test} days; stack pairs inside encounters where both stacks show a {part}')
evaluate(None, '')
for dim in (64, 128, 256):
    evaluate(fit(dim, 'pca'), f'PCA {dim} + cosine (nothing learned)')
    for sh in (0.1, 0.5):
        evaluate(fit(dim, 'wccn', sh), f'PCA {dim} + within-animal whitening, shrink {sh}')
    evaluate(fit(dim, 'lda', 0.3), f'PCA {dim} + LDA (within day)')
for nm, bl in (('square crop only', slice(0, 1536)), ('shape crop only', slice(1536, 3072)), ('cls tokens only', np.r_[0:768, 1536:2304]), ('mean tokens only', np.r_[768:1536, 2304:3072])):
    evaluate(fit(128, 'wccn', 0.3, bl), f'PCA 128 + whitening 0.3, {nm}')
