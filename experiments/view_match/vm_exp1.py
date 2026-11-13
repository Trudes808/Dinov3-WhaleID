"""Exp 1: how informative are the page's part similarities, by what the view classifier says the two stacks show?
Stack pairs inside catalogue encounters, same animal against different animals (AUC), train + val days."""
import sys, warnings
import numpy as np
from sklearn.metrics import roc_auc_score
from vm_common import *
warnings.filterwarnings('ignore')
which = sys.argv[1] if len(sys.argv) > 1 else 'val'
days = {'val': VAL, 'train': TRAIN, 'eval': EVAL}[which]
rows = []
for day in days:
    g = load_day(day, 'inputs_gated'); u = load_day(day, 'inputs_ungated'); n = g.n
    hasF = np.array([np.nanmax(p) >= 0.5 if len(p) else False for p in g.pf]); hasD = np.array([np.nanmax(p) >= 0.5 if len(p) else False for p in g.pd])
    for a, b, same in scope_pairs(g):
        gap = max(0.0, max(g.t0[a], g.t0[b]) - min(g.t1[a], g.t1[b]))
        rows.append((day, same, u.F[a, b], g.F[a, b], u.Dn[a, b], g.Dn[a, b], g.B[a, b], g.P[a, b], hasF[a] and hasF[b], hasD[a] and hasD[b], gap, hasF[a] != hasF[b], g.Q[a, b]))
A = np.array([r[1:] for r in rows], float); y = A[:, 0]
def auc(m, col, name):
    m = m & (A[:, col] > 0)
    if m.sum() < 30 or y[m].min() == y[m].max(): print(f'   {name:62s} n {int(m.sum()):6d}  (too few)'); return
    print(f'   {name:62s} n {int(m.sum()):6d}  same {y[m].mean():.3f}  AUC {roc_auc_score(y[m], A[m, col]):.3f}')
print(f'{which}: {len(A)} stack pairs inside encounters, same animal {y.mean():.3f}')
allm = np.ones(len(A), bool); far = A[:, 9] > 60
for tag, m0 in (('all gaps', allm), ('gap > 60 s', far)):
    print(f' {tag}')
    auc(m0, 1, 'fluke cosine, every pair with two fluke charts (before)')
    auc(m0 & (A[:, 7] > 0), 1, 'fluke cosine, both stacks show a fluke (classifier)')
    auc(m0 & (A[:, 7] == 0), 1, 'fluke cosine, fluke chart but the classifier says no fluke')
    auc(m0, 2, 'fluke cosine as gated on the page')
    auc(m0, 3, 'dorsal cosine, every pair with two fin charts (before)')
    auc(m0 & (A[:, 8] > 0), 3, 'dorsal cosine, both stacks show a fin (classifier)')
    auc(m0 & (A[:, 8] == 0), 3, 'dorsal cosine, fin chart but the classifier says no fin')
    auc(m0, 4, 'dorsal cosine as gated on the page')
    auc(m0, 5, 'body cosine'); auc(m0, 6, 'link score')
# coverage: stacks the classifier says show the part, with / without a part embedding to compare
for day in days[:0]: pass
tot = collections.Counter()
for day in days:
    g = load_day(day, 'inputs_gated'); u = load_day(day, 'inputs_ungated')
    for i in range(g.n):
        if g.kind[i] == 'extra' or not len(g.pf[i]): continue
        f = np.nanmax(g.pf[i]) >= 0.5; dd = np.nanmax(g.pd[i]) >= 0.5
        tot['stacks'] += 1; tot['fluke stacks'] += f; tot['fluke stacks with a fluke embedding'] += f and (g.F[i] > 0).any() ; tot['fin stacks'] += dd; tot['fin stacks with a fin embedding'] += dd and (g.Dn[i] > 0).any()
        tot['stacks with a fluke chart but no fluke'] += (not f) and (u.F[i] > 0).any(); tot['stacks with a fin chart but no fin'] += (not dd) and (u.Dn[i] > 0).any()
print(dict(tot))
