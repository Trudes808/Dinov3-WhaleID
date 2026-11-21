"""Diagnostic: is "a clear mark on one fin, nothing drawn there on the other" clean evidence of DIFFERENT when both
clusters are seen very well? Same side, both clusters >= 3 registered frames, fin >= 70 px, side view; a missing mark
must be >= AREA px, persistent in the consensus, and fall where the other cluster saw the skin in >= 3 frames with no
mark drawn within 10 px in any frame. Precision of "miss >= k" for DIFFERENT against the base rate on these pairs.
    python strict.py"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

import extract  # noqa: F401
import bench
import score as SC

HERE = Path(__file__).resolve().parent


def run(day):
    C = SC.day_clusters(day)
    rows = []
    for a, b, same in bench.pairs(day, 'dorsal'):
        if a not in C or b not in C:
            continue
        A, B = C[a], C[b]
        if A['side'] != B['side'] or min(A['n'], B['n']) < 3 or min(A['H'], B['H']) < 70 or min(A['base'], B['base']) < 1.2:
            continue
        r = [bool(same)]
        for area in (30, 60, 120):
            SC.MISS_AREA = area
            old = SC.marks_cmp.__defaults__
            # need >= 3 seeing frames: temporarily raise n so that need = 2 -> emulate with seen_n >= 3 by masking
            A2, B2 = dict(A), dict(B)
            A2['seen_n'] = np.where(A['seen_n'] >= 3, A['seen_n'], 0).astype(np.uint8)
            B2['seen_n'] = np.where(B['seen_n'] >= 3, B['seen_n'], 0).astype(np.uint8)
            ncc, miss, co = SC.marks_cmp(A2, B2)
            r.append(miss)
        rows.append(r)
    return rows


if __name__ == '__main__':
    cv2.setNumThreads(1)
    with Pool(7) as p:
        R = [r for rs in p.map(run, bench.TRAIN + bench.VAL) for r in rs]
    R = np.array(R, float)
    same = R[:, 0] > 0
    out = dict(pairs=len(R), same=int(same.sum()), diff=int((~same).sum()), base_rate_diff=float((~same).mean()))
    print(out)
    for j, area in enumerate((30, 60, 120), 1):
        for k in (1, 2, 3):
            m = R[:, j] >= k
            out[f'area{area}_miss>={k}'] = dict(n=int(m.sum()), precision_diff=float((~same[m]).mean()) if m.any() else None,
                                              share_of_diff=float((m & ~same).sum() / max(1, (~same).sum())))
            print(area, k, out[f'area{area}_miss>={k}'])
    json.dump(out, open(HERE / 'strict.json', 'w'), indent=1)
