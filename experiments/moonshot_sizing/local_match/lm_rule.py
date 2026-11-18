"""Fixed, untrained linking rule: cluster pair linked when hom12 (homography inliers, 1.2 % threshold, best of 2 x 2
instance pairs) >= T. Thresholds were read off the val days (T dorsal 120, fluke 50); this script reports them on the
TRAIN days, which played no part in choosing them, and BASE's same-pairs-found at precision 0.99 / 1.0 for reference.
    python lm_rule.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lm_score as SC  # noqa: E402
from lm_lib import bench  # noqa: E402
from size_levers import at_precision  # noqa: E402

for part, nofb, ts in (('dorsal', False, (50, 80, 120, 200)), ('fluke', True, (30, 50, 80, 120))):
    for split, days in (('val', bench.VAL), ('train', bench.TRAIN)):
        S = SC.cluster_scores('aliked1024', part, days, 'plain', no_fallback=nofb)['hom12']
        rows = [(S[d][(a, b)], s) for d in days for a, b, s in bench.pairs(d, part) if (a, b) in S.get(d, {})]
        if not rows:
            continue
        v, y = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
        nd = len([d for d in days if S.get(d)])
        print(f'{part} {split} ({nd} days, {len(rows)} pairs, {int(y.sum())} same)')
        for t in ts:
            m = v >= t
            print(f'   hom12 >= {t:3d}: {int(m.sum()):4d} linked, {int(y[m].sum()):4d} right, {int(m.sum() - y[m].sum()):3d} wrong, recall {y[m].sum() / max(y.sum(), 1):.3f}')
        fit = [d for d in bench.TRAIN if d not in days] or bench.TRAIN
        B = bench.base_scores(part, days, fit if split == 'train' else bench.TRAIN) if split == 'val' else None
        if B is not None:
            pos = np.array([B[d][(a, b)] for d in days for a, b, s in bench.pairs(d, part) if s and (a, b) in S.get(d, {})])
            neg = np.array([B[d][(a, b)] for d in days for a, b, s in bench.pairs(d, part) if not s and (a, b) in S.get(d, {})])
            print(f'   BASE (head on 21 train days), same pairs: found at precision 0.99 {at_precision(pos, neg, 0.99)}, at 1.0 {at_precision(pos, neg, 1.0)}')
