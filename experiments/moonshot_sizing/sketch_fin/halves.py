"""Diagnostic: how repeatable is the drawing within seconds? Safe clusters with >= 4 frames on one side are split into two
halves (alternate frames), each half gets its own consensus drawing, and the two halves are compared like a pair.
Against: the bench's different-animal pairs (same group) of the same days. Upper bound for "same animal".

    python halves.py
"""
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np

import extract  # noqa: F401
import bench
import draw as DR
import score as SC
from size_levers import auc

HERE = Path(__file__).resolve().parent
DAYS = bench.VAL + ['2025-01-16', '2025-03-13', '2025-04-01']


def pack(cons, H, base):
    c = dict(top=cons['top'], trail=cons['trail'], marks=cons['marks'], seen_n=cons['seen_n'].astype(np.uint8), raw=cons['raw'].astype(np.uint8),
             whale=cons['whale'], n=cons['n'], H=H, base=base, side=True)
    return SC.clean(c)


def main():
    dc_h, ncc_h, miss_h, dc_d, ncc_d, miss_d = [], [], [], [], [], []
    for day in DAYS:
        keys, z, info = DR.load_day(day)
        idx = {k: i for i, k in enumerate(keys)}
        for c, ks in bench.clusters(day, 'dorsal').items():
            ks = [k for k in ks if k in idx]
            if len(ks) < 4:
                continue
            ii = [idx[k] for k in ks]
            used, fr, cons, side, ref = DR.cluster_drawing((z['rgb'][ii], z['mask'][ii], z['inside'][ii]), info, ks)
            fr = [f for f in fr if f.get('ncc', 1) >= DR.MIN_NCC]
            if len(fr) < 4:
                continue
            H = float(np.median([info[k]['H'] for k in used])); base = float(np.median([info[k]['base'] for k in used]))
            a, b = pack(DR.consensus(fr[0::2]), H, base), pack(DR.consensus(fr[1::2]), H, base)
            if a['well']:
                v = SC.contour_dist(a, b)
                if v is not None:
                    dc_h.append(v)
            n_, m_, _ = SC.marks_cmp(a, b)
            if n_ is not None:
                ncc_h.append(n_)
            miss_h.append(m_)
        P = pickle.load(open(SC.CACHE / 'pairs' / f'{day}.pkl', 'rb'))
        for p, f in P.items():
            if f['same']:
                continue
            if f.get('dc') is not None:
                dc_d.append(f['dc'])
            if f.get('mark_ncc') is not None:
                ncc_d.append(f['mark_ncc'])
            if f.get('sameside') and f.get('coseen', 0) > 0:
                miss_d.append(f['miss'])
    r = dict(n_halves_contour=len(dc_h), n_halves_marks=len(ncc_h),
             contour_auc_halves_vs_diff=auc(-np.array(dc_h), -np.array(dc_d)), marks_auc_halves_vs_diff=auc(np.array(ncc_h), np.array(ncc_d)),
             miss_zero_halves=float(np.mean(np.array(miss_h) == 0)), miss_zero_diff=float(np.mean(np.array(miss_d) == 0)),
             dc_median_halves=float(np.median(dc_h)), dc_median_diff=float(np.median(dc_d)),
             ncc_median_halves=float(np.median(ncc_h)), ncc_median_diff=float(np.median(ncc_d)))
    print(json.dumps(r, indent=1))
    json.dump(r, open(HERE / 'halves.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
