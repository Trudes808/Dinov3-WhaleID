"""sketch_fluke step 3: cluster drawings, drawing comparison, light-share class, exclusion ("different") evidence.

Nothing here is fitted on labels: the template (mean blade silhouette of clear light-underside views, build.py), the
gates and the class bins are fixed by hand before any scoring. Bench numbers: run.py.

Per instance (from build.py's cache):
  prec / rec   share of the seen mask inside the fluke template / share of the template that is seen ("seen well")
  side         'ventral' when the notch sits below the tip line in the upright photo (> 0.08 half-span: tail raised, seen
               from behind) or the blade is more than 25 % light (the upper surface is dark); 'dorsal' when the notch sits
               above the line (< -0.05: fluke low, seen from above) and the blade is not light; else unknown
  drawable     tips from the point model and on the mask (frame 'points'; the mask-extreme fallback mis-registers
               single-blade close-ups as whole flukes, see sheets), ok (span >= 250 RAW px, not cut by the frame, blade area >= 0.045 span^2) and prec >= 0.60 and rec >= 0.30 (template = mean silhouette > 0.15, dilated 3 px)
Per cluster and side: consensus drawing (draw.consensus); side 'any' = all drawable frames whatever the side.
"""
import functools
import json

import cv2
import numpy as np

import draw  # noqa: F401  (puts the bench folder on the path)
import bench
from prep import CACHE

T = cv2.dilate((np.load(CACHE / 'template.npy') > 0.15).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0   # loose blade template
PREC_MIN, REC_MIN = 0.60, 0.30
BINS = (0.05, 0.25, 0.5, 0.75)        # light share of the drawn blade -> class 1..5 (1 = dark, 5 = mostly white)
CS = 2                                # comparison at 128 x 64


@functools.lru_cache(4)
def inst(day):
    z = np.load(CACHE / 'inst' / f'{day}.npz')
    meta = json.load(open(CACHE / 'inst' / f'{day}.json'))
    seen, n = z['seen'], z['n'].astype(np.float32) / 200.0
    for i, r in enumerate(meta):
        s = seen[i]
        k = (s & T).sum()
        r['prec'] = float(k / max(s.sum(), 1)); r['rec'] = float(k / T.sum())
        r['light'] = float((n[i][s] > 0.5).mean()) if s.sum() else 0.0
        g = r.get('notch')
        g = np.nan if g is None else g
        if (np.isfinite(g) and g > 0.08) or r['light'] > 0.25:
            r['side'] = 'ventral'
        elif np.isfinite(g) and g < -0.05:
            r['side'] = 'dorsal'
        else:
            r['side'] = None
        r['drawable'] = bool(r['ok'] and r.get('frame') == 'points' and r['prec'] >= PREC_MIN and r['rec'] >= REC_MIN)
    return meta, seen, n, z['im']


@functools.lru_cache(4)
def cluster_drawings(day):
    """{cluster: {side: consensus}} for side in ventral / dorsal / any."""
    meta, seen, n, _ = inst(day)
    by = {}
    for i, r in enumerate(meta):
        if r['drawable'] and r['cluster'] is not None:
            by.setdefault(r['cluster'], []).append(i)
    out = {}
    for c, ix in by.items():
        d = {}
        for side in ('ventral', 'dorsal', 'any'):
            jj = [i for i in ix if side == 'any' or meta[i]['side'] == side]
            if not jj:
                continue
            rs = [dict(ok=True, seen=seen[i], n=n[i], key=meta[i]['key']) for i in jj]
            cons = draw.consensus(rs)
            cons['small'] = small(cons)
            cons['light'] = float((cons['n'][cons['seen']] > 0.5).mean()) if cons['seen'].sum() else 0.0
            cons['cls'] = int(np.searchsorted(BINS, cons['light'])) + 1
            cons['rec'] = float((cons['seen'] & T).sum() / T.sum())
            d[side] = cons
        out[c] = d
    return out


def small(cons):
    s = cv2.resize(cons['seen'].astype(np.uint8), (draw.W // CS, draw.H // CS), interpolation=cv2.INTER_NEAREST) > 0
    s = cv2.erode(s.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    v = cv2.resize(cons['n'], (draw.W // CS, draw.H // CS), interpolation=cv2.INTER_AREA)
    v = cv2.GaussianBlur(v, (0, 0), 1.0)
    st = cv2.resize(cons['stable'], (draw.W // CS, draw.H // CS), interpolation=cv2.INTER_AREA)
    return s, v, st


SHIFTS = [(dx, dy) for dx in (-4, -2, 0, 2, 4) for dy in (-2, 0, 2)]
MIN_COMMON = 0.15 * T.sum() / CS ** 2


def _shift(a, dx, dy):
    return np.roll(np.roll(a, dy, 0), dx, 1)


def compare(A, B):
    """(similarity, light-share difference on the common region, common area share, contradiction share) or None when they share too little."""
    sa, va, ta = A['small']; sb, vb, tb = B['small']
    best, bdiff, bcom, bcon = -2.0, None, 0, 0.0
    for dx, dy in SHIFTS:
        sb2, vb2 = _shift(sb, dx, dy), _shift(vb, dx, dy)
        c = sa & sb2
        k = c.sum()
        if k < MIN_COMMON:
            continue
        x, y = va[c], vb2[c]
        sx, sy = x.std(), y.std()
        r = float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy)) if sx > 1e-3 and sy > 1e-3 else 0.0
        if r > best:
            best, bcom = r, k
            bdiff = abs(float((x > 0.5).mean()) - float((y > 0.5).mean()))
            bcon = float((((x > 0.55) & (y < 0.3)) | ((y > 0.55) & (x < 0.3))).mean())   # clear light against clear dark
    if bdiff is None:
        return None
    return best, bdiff, bcom / (T.sum() / CS ** 2), bcon


@functools.lru_cache(4)
def day_scores(day):
    """{(a, b): {side: (sim, diff, common)}} for the bench pairs of the day."""
    D = cluster_drawings(day)
    out = {}
    for a, b, _ in bench.pairs(day, 'fluke'):
        if a not in D or b not in D:
            continue
        r = {}
        for side in ('ventral', 'dorsal', 'any'):
            if side in D[a] and side in D[b]:
                v = compare(D[a][side], D[b][side])
                if v is not None:
                    r[side] = v
        if r:
            out[(a, b)] = r
    return out
