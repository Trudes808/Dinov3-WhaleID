"""sketch_v2 C (2026-11-19): which marks hold across moments and light? For two drawings of the same side of the SAME
catalogue animal (two of its stacks, more than 60 s apart: different moments, often different light) the second is lined
up onto the first (fin_sketch.register_view: stretch along the body, scale, shift) and every consensus mark of the first
(drawn or left out as small) that falls where the second saw the skin is checked: does the second have a mark within
V_TOL px? Against the same check for two DIFFERENT animals of one group (bench.pairs: the chance of a hit). By mark size:
equivalent diameter in fin heights, and area in source pixels. 21 train + 3 val days only.
    python mark_size.py [SUB]      SUB = the drawings folder of the day (default sketch/v2)  -> mark_size[_<tag>].json"""
import collections
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'moonshot_sizing'))
sys.path.insert(0, str(HERE.parents[1]))
import bench  # noqa: E402
from model_fine_tuning.cluster import fin_sketch as FS  # noqa: E402

SUB = sys.argv[1] if len(sys.argv) > 1 else 'sketch/v2'
MIN_SCORE = float(sys.argv[2]) if len(sys.argv) > 2 else FS.V_MIN_SCORE      # 0 = every pair, to see how agreement depends on the lining-up score
D_BINS = (0, 0.04, 0.06, 0.08, 0.10, 0.13, 0.18, 0.25, 9)
SRC_BINS = (0, 6, 12, 25, 50, 100, 1e9)


def views(day):
    d = FS.day_dir(day) / SUB
    idx = json.loads((d / 'index.json').read_text())['clusters']
    out = {}
    for c in bench.clusters(day, 'dorsal'):
        for s, x in (idx.get(str(c)) or {}).items():
            out[(c, s)] = FS.load_view(d / x['npz'])
    return out


def check(a, b):
    """rows (diameter H, source px, hit) for a's marks against b lined up onto a."""
    M, sc = FS.register_view(a, b)
    if sc < MIN_SCORE:
        return []
    w = lambda x: cv2.warpAffine(x.astype(np.uint8), M, (FS.DW, FS.DH), flags=cv2.INTER_NEAREST) > 0
    bm, bs = w(b['marks'] | b['small']), w(b['seen'])
    k = np.ones((2 * FS.V_TOL + 1, 2 * FS.V_TOL + 1), np.uint8)
    near = cv2.dilate(bm.astype(np.uint8), k) > 0
    am = (a['marks'] | a['small']) & a['seen']
    n, lab, st, _ = cv2.connectedComponentsWithStats(am.astype(np.uint8), connectivity=8)
    H = a['meta']['H']
    rows = []
    for i in range(1, n):
        comp = lab == i
        if bs[comp].mean() < 0.8:
            continue
        area_h2 = st[i, cv2.CC_STAT_AREA] / FS.DPX ** 2
        rows.append((float(np.sqrt(4 * area_h2 / np.pi)), float(area_h2 * H * H), bool(near[comp].any()), bool(a['n'] >= 2), float(sc)))
    return rows


def run(day):
    V = views(day)
    Dd = bench.D(day)
    out = []
    for a, b, same in bench.pairs(day, 'dorsal'):
        for s in ('L', 'R'):
            if (a, s) in V and (b, s) in V:
                for x, y in ((a, b), (b, a)):
                    out += [(bool(same),) + r for r in check(V[(x, s)], V[(y, s)])]
    return day, out


if __name__ == '__main__':
    cv2.setNumThreads(1)
    with Pool(8) as p:
        R = [r for _, rs in p.map(run, bench.TRAIN + bench.VAL) for r in rs]
    R = np.array(R, float)
    same, dia, src, hit, multi, score = R[:, 0] > 0, R[:, 1], R[:, 2], R[:, 3] > 0, R[:, 4] > 0, R[:, 5]
    res = {'marks_same': int(same.sum()), 'marks_diff': int((~same).sum()), 'by_diameter': [], 'by_source_px': []}
    for name, v, bins in (('by_diameter', dia, D_BINS), ('by_source_px', src, SRC_BINS)):
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (v >= lo) & (v < hi)
            hs, hd = hit[m & same], hit[m & ~same]
            row = dict(lo=lo, hi=hi, n_same=int(len(hs)), n_diff=int(len(hd)), hit_same=round(float(hs.mean()), 3) if len(hs) else None,
                       hit_diff=round(float(hd.mean()), 3) if len(hd) else None)
            if len(hs) and len(hd):
                row['lift'] = round(row['hit_same'] - row['hit_diff'], 3)
            res[name].append(row)
            print(name, row)
    # the joint rule: diameter >= d and source px >= p
    res['rules'] = []
    for d in (0.0, 0.04, 0.06, 0.08, 0.10, 0.13):
        for p_ in (0, 6, 12, 25):
            m = (dia >= d) & (src >= p_)
            hs, hd = hit[m & same], hit[m & ~same]
            res['rules'].append(dict(d=d, p=p_, kept_same=int(len(hs)), hit_same=round(float(hs.mean()), 3), hit_diff=round(float(hd.mean()), 3),
                                     left_out_hit_same=round(float(hit[~m & same].mean()), 3) if (~m & same).any() else None))
    for r in res['rules']:
        print(r)
    # the lining-up score: marks of the drawn size, hit rate same vs different animal by score of the pair
    res['by_score'] = []
    big = dia >= FS.MARK_D
    for lo, hi in ((0, 0.3), (0.3, 0.4), (0.4, 0.45), (0.45, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.01)):
        m = big & (score >= lo) & (score < hi)
        hs, hd = hit[m & same], hit[m & ~same]
        row = dict(lo=lo, hi=hi, n_same=int(len(hs)), n_diff=int(len(hd)), hit_same=round(float(hs.mean()), 3) if len(hs) else None, hit_diff=round(float(hd.mean()), 3) if len(hd) else None)
        res['by_score'].append(row); print('score', row)
    tag = SUB.replace('sketch/', '').replace('/', '_') + ('' if MIN_SCORE == FS.V_MIN_SCORE else f'_min{MIN_SCORE:g}')
    json.dump(res, open(HERE / f'mark_size_{tag}.json', 'w'), indent=1)
