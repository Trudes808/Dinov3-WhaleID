"""pixels: one line per variant and part from ../results/pixels_*.json (16-day head, val + set-aside, mean of 20
repetitions; paired difference to BASE; val days with all 21; dorsal flanks)."""
import json
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent.parent / 'results'


def row(path):
    o = json.load(open(path))
    h = o['held']
    c, b = h["('cand', 16)"], h["('base', 16)"]
    m = lambda rows, k: float(np.mean([r[k] for r in rows]))
    d = np.array([x['auc'] - y['auc'] for x, y in zip(c, b)])
    d90 = np.array([x['r90'] - y['r90'] for x, y in zip(c, b)]) * 100
    g = lambda s, k='auc': float(np.mean([r[k] for r in h[f"('cand', {s})"]]))
    gb = lambda s, k='auc': float(np.mean([r[k] for r in h[f"('base', {s})"]]))
    v = o['val']
    fl = v['flank']['cand']
    s = (f"{o['name'][7:]:<16s} {o['part']:<6s} n {o['instances']:>4d}/{o['instances_total']:<4d} AUC {m(c, 'auc'):.3f} AP {m(c, 'ap'):.3f} "
         f"P95 {100 * m(c, 'r95'):4.1f} P90 {100 * m(c, 'r90'):4.1f} | dAUC {d.mean():+.3f} up {int((d > 0).sum()):2d}/20 dP90 {d90.mean():+.1f} "
         f"| 5->10 {g(10) - g(5):+.3f} 10->16 {g(16) - g(10):+.3f} (BASE {gb(10) - gb(5):+.3f} {gb(16) - gb(10):+.3f}) "
         f"| val AUC {v['cand']['auc']:.3f} (B {v['base']['auc']:.3f}) P90 {100 * v['cand']['r90']:.1f}")
    if fl:
        s += f" same {fl['same flank']['auc']:.3f} opp {fl['opposite flanks']['auc']:.3f}"
    s += f" | raw {m(h['raw'], 'auc'):.3f}"
    return s


if __name__ == '__main__':
    for p in sorted(R.glob('pixels_*.json')):
        if len(sys.argv) == 1 or any(a in p.name for a in sys.argv[1:]):
            print(row(p))
