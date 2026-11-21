"""sketch_fin step 4: compare consensus drawings of two safe clusters (CPU, train + val days only, no eval day).

Drawing clean-up (no labels): mark components >= MARK_AREA px, the MARK_K largest kept; nicks re-detected on the
consensus trailing edge (rows y in [-0.85, -0.12] H, depth >= NICK_D2 H). "Well seen": side view (keypoint base ratio
>= BASE_MIN), fin height >= H_MIN px in the source.

Pair evidence (cluster a, cluster b):
  dc         contour distance, median |difference| (H) of the top edge (x in [-1.6, -0.3]) and the trailing edge, both
             fins well seen; compared across flanks too (canonical frame mirrored, head left)
  mark_ncc   same side only: b registered onto a (scale + shift on marks + outline), normalised correlation of the
             blurred mark maps over the place both saw
  miss       same side only: marks of one drawing (>= MISS_AREA px) that fall where the other cluster saw the skin in
             >= 2 frames (1 if it has one frame) and drew nothing within MISS_TOL px in any frame; count both ways
  nick_dis   nick on one trailing edge, the other's trailing edge seen there and smooth (|residual| < 0.01 H); both ways
  cls        shape class (k-means 9 on well-seen consensus outlines of the 21 train days, no labels)
"""
from __future__ import annotations

import functools
import json
import pickle
import sys

import cv2
import numpy as np
from sklearn.cluster import KMeans

import extract  # noqa: F401
import bench
import clusters as CL
import draw as DR
from extract import CACHE, PXH, X0, Y0

MARK_AREA, MARK_K = 20, 8
MISS_AREA, MISS_TOL = 30, 10
NICK_D2 = 0.035
BASE_MIN, H_MIN = 1.2, 40
ALL = bench.TRAIN + bench.VAL
TOPSEL = (DR.COLS_TOP >= (-1.6 - X0) * PXH) & (DR.COLS_TOP <= (-0.3 - X0) * PXH)
TRSEL = (DR.ROWS_TR >= (-0.85 - Y0) * PXH) & (DR.ROWS_TR <= (-0.12 - Y0) * PXH)


def clean(c):
    m = c['marks'].astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
    comps = [(st[i, cv2.CC_STAT_AREA], i) for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= MARK_AREA]
    comps = sorted(comps, reverse=True)[:MARK_K]
    keep = np.zeros(n, bool)
    for a, i in comps:
        keep[i] = True
    c['mk'] = keep[lab]
    c['comps'] = [(lab == i) for a, i in comps]
    c['well'] = c['base'] >= BASE_MIN and c['H'] >= H_MIN and c['n'] >= 1
    nk, r = DR.nicks(c['trail'], -1, depth=NICK_D2)
    c['nk'] = [i for i in nk if TRSEL[i]]
    c['res'] = r
    w = c['whale'].astype(np.uint8)
    band = (w - cv2.erode(w, np.ones((5, 5), np.uint8))).astype(np.float32)
    c['feat'] = cv2.GaussianBlur(c['mk'].astype(np.float32), (0, 0), 2.0) * 3 + cv2.GaussianBlur(band, (0, 0), 2.0)
    return c


@functools.lru_cache(None)
def day_clusters(day):
    return {k: clean(v) for k, v in CL.load(day).items()}


def descriptor(c):
    v = np.concatenate([c['top'][TOPSEL][::5] / PXH, c['trail'][TRSEL][::5] / PXH])
    return None if np.isnan(v).any() else v


def contour_dist(a, b):
    d1 = np.abs(a['top'][TOPSEL] - b['top'][TOPSEL]) / PXH
    d2 = np.abs(a['trail'][TRSEL] - b['trail'][TRSEL]) / PXH
    d = np.concatenate([d1[~np.isnan(d1)], d2[~np.isnan(d2)]])
    return float(np.median(d)) if len(d) >= 15 else None


def nick_dis(a, b):
    n = 0
    for x, y in ((a, b), (b, a)):
        for i in x['nk']:
            lo, hi = max(i - 3, 0), i + 4
            seg = y['trail'][lo:hi]
            if len(seg) and not np.isnan(seg).any() and np.abs(y['res'][lo:hi]).max() < 0.01:
                n += 1
    return n


def marks_cmp(a, b):
    """(ncc of marks after registration, missing-mark count) for two drawings of the same side."""
    M, _ = DR.register(a['feat'], b['feat'], fb_is_feat=True)
    TW, TH = extract.TW, extract.TH
    sb = cv2.warpAffine(b['seen_n'], M, (TW, TH), flags=cv2.INTER_NEAREST)
    rb = cv2.warpAffine(b['raw'], M, (TW, TH), flags=cv2.INTER_NEAREST)
    mb = cv2.warpAffine(b['mk'].astype(np.uint8), M, (TW, TH), flags=cv2.INTER_NEAREST) > 0
    co = (a['seen_n'] >= 1) & (sb >= 1)
    ga = cv2.GaussianBlur(a['mk'].astype(np.float32), (0, 0), 3.0) * co
    gb = cv2.GaussianBlur(mb.astype(np.float32), (0, 0), 3.0) * co
    den = np.sqrt((ga ** 2).sum() * (gb ** 2).sum())
    ncc = float((ga * gb).sum() / den) if den > 1e-6 else None
    k = np.ones((2 * MISS_TOL + 1, 2 * MISS_TOL + 1), np.uint8)
    miss = 0
    # a's marks absent in b
    need_b = 2 if b['n'] >= 2 else 1
    near_b = cv2.dilate((rb >= 1).astype(np.uint8), k) > 0
    for comp in a['comps']:
        if comp.sum() >= MISS_AREA and (sb[comp] >= need_b).mean() >= 0.8 and not near_b[comp].any():
            miss += 1
    # b's marks absent in a (in a's frame)
    need_a = 2 if a['n'] >= 2 else 1
    near_a = cv2.dilate((a['raw'] >= 1).astype(np.uint8), k) > 0
    for comp in b['comps']:
        cw = cv2.warpAffine(comp.astype(np.uint8), M, (TW, TH), flags=cv2.INTER_NEAREST) > 0
        if cw.sum() >= MISS_AREA and (a['seen_n'][cw] >= need_a).mean() >= 0.8 and not near_a[cw].any():
            miss += 1
    return ncc, miss, int(co.sum())


def fit_classes(k=9, seed=0):
    X = []
    for d in bench.TRAIN:
        for c in day_clusters(d).values():
            v = descriptor(c)
            if c['well'] and c['n'] >= 2 and v is not None:
                X.append(v)
    km = KMeans(k, n_init=10, random_state=seed).fit(np.array(X))
    return km, len(X)


def pair_features(day, km):
    C = day_clusters(day)
    cls = {}
    for c, v in C.items():
        dv = descriptor(v)
        cls[c] = int(km.predict(dv[None])[0]) if v['well'] and dv is not None else None
    out = {}
    for a, b, same in bench.pairs(day, 'dorsal'):
        if a not in C or b not in C:
            continue
        A, B = C[a], C[b]
        f = dict(same=bool(same), side_a=A['side'], side_b=B['side'], sameside=A['side'] == B['side'], na=A['n'], nb=B['n'],
                 well=A['well'] and B['well'], cls_a=cls[a], cls_b=cls[b])
        f['dc'] = contour_dist(A, B) if f['well'] else None
        f['nick_dis'] = nick_dis(A, B) if f['well'] else None
        f['nick_any'] = len(A['nk']) + len(B['nk'])
        if f['sameside']:
            f['mark_ncc'], f['miss'], f['coseen'] = marks_cmp(A, B)
        out[(a, b)] = f
    return out


def run_day(day):
    km = pickle.load(open(CACHE / 'kmeans9.pkl', 'rb'))
    out = CACHE / 'pairs' / f'{day}.pkl'
    if not out.exists():
        out.parent.mkdir(exist_ok=True)
        pickle.dump(pair_features(day, km), open(out, 'wb'))
    return day


if __name__ == '__main__':
    from multiprocessing import Pool
    cv2.setNumThreads(1)
    if not (CACHE / 'kmeans9.pkl').exists():
        km, n = fit_classes()
        pickle.dump(km, open(CACHE / 'kmeans9.pkl', 'wb'))
        print('k-means on', n, 'well-seen train clusters', flush=True)
    days = sorted(ALL, key=lambda d: -len(bench.pairs(d, 'dorsal')))
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 7) as pool:
        for d in pool.imap_unordered(run_day, days):
            print('done', d, flush=True)
    print('ALL DONE', flush=True)
