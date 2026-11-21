"""Diagnostic: does the drawing compare better when the two fins were seen from a similar angle? Pairs split by the
ratio of the two clusters' keypoint base ratios (front-base-to-back-base length / fin height, a proxy of the viewing
angle). Label-free scores, all 24 days pooled, HW-b6 on the same pairs.   python view_strata.py"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.argv = sys.argv[:1]
sys.path.insert(0, str(HERE.parent))
import bench  # noqa: E402
from check_hw import hw_feats, pair_scores  # noqa: E402
from size_levers import auc  # noqa: E402

CACHE = Path('/home/sat3737/NOAA_whales/moonshot_cache/sketch_fin')
ALL = bench.TRAIN + bench.VAL
out = {}
rows = {k: [] for k in ('dc', 'ncc', 'cls')}
for d in ALL:
    F = pickle.load(open(CACHE / 'pairs' / f'{d}.pkl', 'rb'))
    C = pickle.load(open(CACHE / 'clusters' / f'{d}.pkl', 'rb'))
    HW = pair_scores(d, hw_feats(d))
    for p, f in F.items():
        r = abs(np.log(C[p[0]]['base'] / C[p[1]]['base']))
        if f.get('dc') is not None:
            rows['dc'].append((r, -f['dc'], f['same'], HW[p]))
        if f.get('mark_ncc') is not None:
            rows['ncc'].append((r, f['mark_ncc'], f['same'], HW[p]))
        if f['cls_a'] is not None and f['cls_b'] is not None:
            rows['cls'].append((r, float(f['cls_a'] == f['cls_b']), f['same'], HW[p]))
for k, R in rows.items():
    R = np.array(R, float)
    for lo, hi in ((0, 0.15), (0.15, 0.4), (0.4, 9)):
        m = (R[:, 0] >= lo) & (R[:, 0] < hi)
        s, y, h = R[m, 1], R[m, 2] > 0, R[m, 3]
        res = dict(same=int(y.sum()), diff=int((~y).sum()), auc=auc(s[y], s[~y]), hw_auc=auc(h[y], h[~y]))
        if k == 'cls':
            res.update(same_eq=float(s[y].mean()), diff_eq=float(s[~y].mean()))
        out[f'{k}|{lo}-{hi}'] = res
        print(k, f'|log base ratio| {lo}-{hi}', res)
json.dump(out, open(HERE / 'view_strata.json', 'w'), indent=1)
