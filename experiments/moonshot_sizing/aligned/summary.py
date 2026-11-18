"""One table over results/aligned_*.json: val + set-aside days at 16 days (mean of 20 reps), paired difference to BASE,
coverage, val days (21 days), same / opposite flanks (val)."""
import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent.parent / 'results'


def m(rows, k):
    return float(np.mean([r[k] for r in rows]))


def d(c, b, k):
    v = np.array([x[k] - y[k] for x, y in zip(c, b)])
    return float(v.mean()), int((v > 0).sum())


print(f"{'name':<36s} {'part':<6s} {'AUC':>6s} {'AP':>6s} {'P95%':>5s} {'P90%':>5s} | {'dAUC':>7s} up | {'dP90':>6s} up | {'BASE AUC':>8s} | cover | val AUC (BASE) | same fl | opp fl")
for f in sorted(R.glob('aligned_*.json')):
    j = json.load(open(f))
    kind = j.get('kind')
    if kind == 'features':
        c, b = j['held']["('cand', 16)"], j['held']["('base', 16)"]
        cov = f"{j['instances']}/{j['instances_total']} inst"
        v = j['val']; vc, vb = v['cand'], v['base']; fl = v['flank']['cand']
    elif kind in ('pairs', 'fusion'):
        c, b = j['held']['cand'], j['held']['base']
        cov = f"{j['pairs_scored']}/{j['pairs_total']}" if kind == 'pairs' else 'all (fill)'
        v = j['val']; vc, vb = v['cand'], v['base']; fl = v['flank']['cand']
    elif kind == 'fitted_pairs':
        c, b = j['held']['16']['cand'], j['held']['16']['base']
        cov = 'q-filtered'
        v = j['val']; vc, vb = v['cand'], v['base']; fl = v['flank']['cand']
    else:
        continue
    da, ua = d(c, b, 'auc'); d9, u9 = d(c, b, 'r90')
    sf = fl.get('same flank', {}).get('auc', float('nan')); of = fl.get('opposite flanks', {}).get('auc', float('nan'))
    print(f"{j['name']:<36s} {j['part']:<6s} {m(c, 'auc'):6.3f} {m(c, 'ap'):6.3f} {100 * m(c, 'r95'):5.1f} {100 * m(c, 'r90'):5.1f} | {da:+7.3f} {ua:2d} | {100 * d9:+6.1f} {u9:2d} | {m(b, 'auc'):8.3f} | {cov} | {vc['auc']:.3f} ({vb['auc']:.3f}) | {sf:.3f} | {of:.3f}")
