"""One table from results/public_reid_*.json (bench.evaluate / run_eval.py fuse outputs)."""
import glob
import json
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent.parent / 'results'


def m(rows, k):
    return float(np.mean([r[k] for r in rows]))


def up(c, b, k):
    d = np.array([x[k] - y[k] for x, y in zip(c, b)])
    return float(d.mean()), int((d > 0).sum())


print(f"{'name':<44s} {'part':<6s} | raw AUC  AP   P95  P90 | head16 AUC   AP   P95   P90 | dAUC (up/20) dP90 (up/20) | val21 AUC BASE | AUC 5->10->16  BASE")
for f in sorted(glob.glob(str(R / 'public_reid_*.json'))):
    o = json.load(open(f))
    h = o['held']
    if o.get('kind') == 'fusion_curve':
        continue
    if o.get('kind') == 'features':
        c, b = h['(\'cand\', 16)'], h['(\'base\', 16)']
        raw = h['raw']
        rs = f"{m(raw, 'auc'):.3f} {m(raw, 'ap'):.3f} {100 * m(raw, 'r95'):4.1f} {100 * m(raw, 'r90'):4.1f}"
        lc = ' '.join(f"{m(h[str(('cand', s))], 'auc'):.3f}" for s in (5, 10, 16))
        lb = ' '.join(f"{m(h[str(('base', s))], 'auc'):.3f}" for s in (5, 10, 16))
    else:
        c, b = h['cand'], h['base']
        rs, lc, lb = '  -     -     -    -  ', '', ''
    da, ka = up(c, b, 'auc')
    d9, k9 = up(c, b, 'r90')
    v = o['val']
    print(f"{o['name']:<44s} {o['part']:<6s} | {rs} | {m(c, 'auc'):.3f} {m(c, 'ap'):.3f} {100 * m(c, 'r95'):4.1f} {100 * m(c, 'r90'):4.1f} | "
          f"{da:+.3f} ({ka:2d})  {100 * d9:+.1f} ({k9:2d}) | {v['cand']['auc']:.3f} {v['base']['auc']:.3f} P90 {100 * v['cand']['r90']:4.1f} {100 * v['base']['r90']:4.1f} | {lc} | {lb}")
