"""sketch_fin step 2: the automatic "researcher's drawing" of one dorsal tile, and the consensus drawing of a safe cluster.

Per frame (canonical tile from extract.py, head left, back base (0, 0), tip (-0.25, -1), units = fin heights H):
  contour   top edge y(x) of the whale mask for x <= tip (leading edge + hump), trailing edge x(y) for rows between the
            tip and the back base; each mask edge moved to the strongest brightness step within +-4 px
  nicks     where the trailing edge (or the tip region) cuts in by >= NICK_D H against its own smoothed curve
  marks     skin pixels at least K_MARK robust deviations brighter than their local skin median (scars, rakes); clipped
            highlights are "not seen", not marks; a band along the outline is excluded (rim light)
  seen      skin pixels inside the photo, not clipped, away from the outline
Per safe cluster (frames registered by the keypoint frame, one side only = the majority side of the frames):
  a mark is kept where >= PERSIST of the frames that saw that place show a mark within TOL px (needs >= 2 frames);
  contour = median over frames per column / row; nicks re-detected on the median trailing edge;
  seen count map = number of frames that saw each place.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter

from extract import CACHE, PXH, TH, TW, X0, Y0

NICK_D, NICK_SM = 0.025, 4.0          # nick depth (H) against a curve smoothed with sigma 4 px
K_MARK, MIN_AREA = 2.8, 12            # mark threshold (robust sd) and min component area (px)
RIM = 5                               # px band along the outline excluded from marks
CLIP = 245                            # max(rgb) at or above = clipped highlight
MIN_NCC = 0.35                        # registration score below which a frame is left out of the consensus
TOL, PERSIST = 5, 0.5                 # consensus tolerance (px) and share of seeing frames that must agree

COLS_TOP = np.arange(0, int(round((-0.25 - X0) * PXH)) + 1)          # tile columns for x in [X0, -0.25]
ROWS_TR = np.arange(int(round((-0.97 - Y0) * PXH)), int(round((-0.08 - Y0) * PXH)) + 1)


def px(x, y):
    return (x - X0) * PXH, (y - Y0) * PXH


def whale_of(mask, inside):
    m = (mask > 127) & (inside > 0)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m > 0
    cx, cy = (int(v) for v in px(-0.45, -0.35))
    k = lab[cy, cx] if lab[cy, cx] > 0 else 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return lab == k


def contour(whale, inside, L):
    """top[y per column of COLS_TOP] and trail[x per row of ROWS_TR] in tile px (NaN = not seen)."""
    g = cv2.GaussianBlur(L, (0, 0), 1.2)
    gy, gx = np.diff(g, axis=0, prepend=g[:1]), np.diff(g, axis=1, prepend=g[:, :1])
    top = np.full(len(COLS_TOP), np.nan, np.float32)
    for j, c in enumerate(COLS_TOP):
        col = whale[:, c]
        if not col.any():
            continue
        y = int(np.argmax(col))
        if y <= 3 or not inside[y - 3, c]:
            continue
        lo, hi = max(y - 4, 1), min(y + 5, TH)
        y = lo + int(np.argmax(np.abs(gy[lo:hi, c])))
        top[j] = y
    trail = np.full(len(ROWS_TR), np.nan, np.float32)
    for i, r in enumerate(ROWS_TR):
        yr = r / PXH + Y0
        xref = int(round(px(-0.25 - 0.3 * (yr + 1.0), 0)[0]))
        row = whale[r]
        if not row[xref]:
            near = np.nonzero(row[max(xref - 12, 0):xref + 12])[0]
            if not len(near):
                continue
            xref = max(xref - 12, 0) + int(near[0])
        off = np.nonzero(~row[xref:])[0]
        if not len(off):
            continue
        x = xref + int(off[0])                               # first non-whale pixel right of the fin core
        if x >= TW - 3 or not inside[r, min(x + 2, TW - 1)] or x / PXH + X0 > 0.35:
            continue
        lo, hi = max(x - 4, 1), min(x + 5, TW)
        trail[i] = lo + int(np.argmax(np.abs(gx[r, lo:hi])))
    return top, trail


def nicks(prof, sign, sm=NICK_SM, depth=NICK_D):
    """indices of nick centres: profile cuts into the fin by >= depth H against its smoothed self. sign: +1 when a cut
    means a larger value (top edge, y down), -1 when smaller (trailing edge x)."""
    ok = ~np.isnan(prof)
    if ok.sum() < 10:
        return [], np.zeros_like(prof)
    idx = np.arange(len(prof))
    f = np.interp(idx, idx[ok], prof[ok])
    s = gaussian_filter1d(f, sm, mode='nearest')
    r = sign * (f - s) / PXH
    r[~ok] = 0
    out = []
    for i in range(1, len(r) - 1):
        if r[i] >= depth and r[i] >= r[i - 1] and r[i] >= r[i + 1] and ok[max(i - 3, 0):i + 4].all():
            if not out or i - out[-1] > 4:
                out.append(i)
    return out, r


def marks(rgb, whale, inside):
    """(mark bool map, seen bool map, contrast map) for one tile."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0]
    clip = rgb.max(2) >= CLIP
    core = cv2.erode(whale.astype(np.uint8), np.ones((2 * RIM + 1, 2 * RIM + 1), np.uint8)) > 0
    seen = core & ~cv2.dilate(clip.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & (inside > 0)
    if seen.sum() < 200:
        return np.zeros_like(seen), seen, np.zeros_like(L)
    fill = L.copy()
    fill[~whale] = np.median(L[seen])
    bg = median_filter(fill, size=17)
    d = L - bg
    v = d[seen]
    sd = max(1.4826 * np.median(np.abs(v - np.median(v))), 3.0)
    c = d / sd
    m = (c >= K_MARK) & seen
    n, lab_, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= MIN_AREA
    return keep[lab_], seen, c


def frame_drawing(rgb, mask, inside):
    whale = whale_of(mask, inside)
    L = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
    top, trail = contour(whale, inside > 0, L)
    m, seen, c = marks(rgb, whale, inside)
    return dict(top=top, trail=trail, marks=m, seen=seen, whale=whale, c=c, rgb=rgb)


SCALES = (0.9, 0.95, 1.0, 1.05, 1.1)
SHIFT = 24


def feat(fr):
    """image used to register two drawings: persistent-looking contrast + the outline band."""
    c = np.clip(fr['c'], 0, 6) * fr['seen']
    w = fr['whale'].astype(np.uint8)
    band = (w - cv2.erode(w, np.ones((5, 5), np.uint8))).astype(np.float32)
    return cv2.GaussianBlur(c.astype(np.float32), (0, 0), 1.5) + 2.0 * cv2.GaussianBlur(band, (0, 0), 2.0)


def register(fa, fb, scales=SCALES, shift=SHIFT, fb_is_feat=False):
    """best similarity (2x3, tile px) moving drawing b onto a: scale about the back base + translation; (M, ncc)."""
    A = fa if fb_is_feat else feat(fa)
    Bf = fb if fb_is_feat else feat(fb)
    Ap = cv2.copyMakeBorder(A, shift, shift, shift, shift, cv2.BORDER_CONSTANT, value=0)
    cx, cy = px(-0.5, -0.4)
    best = (-2, None)
    for s in scales:
        M = np.array([[s, 0, cx - s * cx], [0, s, cy - s * cy]], np.float32)
        Bw = cv2.warpAffine(Bf, M, (TW, TH))
        r = cv2.matchTemplate(Ap, Bw, cv2.TM_CCORR_NORMED)
        _, v, _, loc = cv2.minMaxLoc(r)
        if v > best[0]:
            M2 = M.copy()
            M2[0, 2] += loc[0] - shift
            M2[1, 2] += loc[1] - shift
            best = (v, M2)
    return best[1], float(best[0])


def warp_tile(rgb, mask, inside, M):
    return (cv2.warpAffine(rgb, M, (TW, TH), flags=cv2.INTER_LINEAR), cv2.warpAffine(mask, M, (TW, TH), flags=cv2.INTER_LINEAR),
            cv2.warpAffine(inside, M, (TW, TH), flags=cv2.INTER_NEAREST))


def cluster_drawing(tiles, info, keys, refine=True):
    """tiles = (rgb, mask, inside) arrays indexed like keys. -> (keys used, frame drawings registered, consensus, side,
    index of the reference frame). Only the majority side's frames are used."""
    import collections
    side = collections.Counter(info[k]['mirrored'] for k in keys).most_common(1)[0][0]
    use = [i for i, k in enumerate(keys) if info[k]['mirrored'] == side]
    fr = [frame_drawing(*(t[i] for t in tiles)) for i in use]
    ref = max(range(len(use)), key=lambda j: fr[j]['seen'].sum() * min(info[keys[use[j]]]['H'], 150))
    if refine and len(use) > 1:
        for j in range(len(use)):
            if j == ref:
                continue
            M, v = register(fr[ref], fr[j])
            fr[j] = frame_drawing(*warp_tile(*(t[use[j]] for t in tiles), M))
            fr[j]['ncc'] = v
    good = [f for f in fr if f.get('ncc', 1.0) >= MIN_NCC]       # a frame that does not register is left out
    return [keys[i] for i in use], fr, consensus(good), side, ref


def consensus(frames):
    """frames: list of frame_drawing dicts (one side). -> cluster drawing."""
    n = len(frames)
    top = np.nanmedian(np.stack([f['top'] for f in frames]), 0) if n else None
    trail = np.nanmedian(np.stack([f['trail'] for f in frames]), 0) if n else None
    seen_n = np.sum([f['seen'] for f in frames], 0)
    k = np.ones((2 * TOL + 1, 2 * TOL + 1), np.uint8)
    near = np.sum([cv2.dilate(f['marks'].astype(np.uint8), k).astype(bool) & f['seen'] for f in frames], 0)
    raw = np.sum([f['marks'] for f in frames], 0)
    if n >= 2:
        cm = (seen_n >= 2) & (near >= PERSIST * np.maximum(seen_n, 1)) & (raw >= 1)
        cm = cm & (near >= 2)
    else:
        cm = raw >= 1
    tipc = int(round((-0.7 - X0) * PXH))
    ntop = [i for i in nicks(top, +1)[0] if COLS_TOP[i] >= tipc] if top is not None else []
    ntr = nicks(trail, -1)[0] if trail is not None else []
    return dict(top=top, trail=trail, marks=cm, raw=raw, seen_n=seen_n, n=n, nick_top=ntop, nick_trail=ntr,
                whale=np.mean([f['whale'] for f in frames], 0) >= 0.5)


def smooth_nan(p, s):
    ok = ~np.isnan(p)
    if ok.sum() < 3:
        return p
    i = np.arange(len(p))
    f = gaussian_filter1d(np.interp(i, i[ok], p[ok]), s, mode='nearest')
    return np.where(ok, f, np.nan)


def render(dr, scale=1.0, faint=None):
    """drawing as an RGB image: paper white, unseen skin hatched grey, contour black, marks dark, nicks red."""
    img = np.full((TH, TW, 3), 255, np.uint8)
    wh = dr['whale']
    seen = (dr['seen_n'] >= (2 if dr['n'] >= 2 else 1)) if 'seen_n' in dr else dr['seen']
    hatch = ((np.add.outer(np.arange(TH), np.arange(TW)) // 3) % 4 == 0)
    img[wh & ~seen & hatch] = (200, 200, 200)
    img[wh & seen] = (245, 245, 238)
    if faint is not None:
        img[faint & ~dr['marks']] = (200, 200, 230)
    img[dr['marks']] = (40, 40, 40)
    pts_top = [(int(c), int(y)) for c, y in zip(COLS_TOP, smooth_nan(dr['top'], 3)) if not np.isnan(y)]
    pts_tr = [(int(x), int(r)) for r, x in zip(ROWS_TR, smooth_nan(dr['trail'], 1.5)) if not np.isnan(x)]
    for pts in (pts_top, pts_tr):
        for a, b in zip(pts[:-1], pts[1:]):
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) < 12:
                cv2.line(img, a, b, (0, 0, 0), 2, cv2.LINE_AA)
    for i in dr.get('nick_top', []):
        c, y = int(COLS_TOP[i]), dr['top'][i]
        if not np.isnan(y):
            cv2.line(img, (c, int(y) - 7), (c, int(y) + 7), (220, 0, 0), 3)
    for i in dr.get('nick_trail', []):
        r, x = int(ROWS_TR[i]), dr['trail'][i]
        if not np.isnan(x):
            cv2.line(img, (int(x) - 7, r), (int(x) + 7, r), (220, 0, 0), 3)
    return img


def load_day(day):
    z = np.load(CACHE / 'tiles' / f'{day}.npz')
    info = json.load(open(CACHE / 'tiles' / f'{day}_info.json'))
    return [str(k) for k in z['keys']], z, info
