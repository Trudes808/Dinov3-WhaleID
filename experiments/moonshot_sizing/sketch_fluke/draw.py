"""sketch_fluke step 2: per-instance canonical drawings and per-cluster consensus drawings (CPU, no fitting).

Canonical canvas (W x H = 256 x 128): the two tips (sorted by x in the blades-up chart, whale's right on the image right)
go to (-1, 0) and (1, 0) half-spans, y down, X in [-1.15, 1.15], Y in [-0.35, 0.80]; the notch lands near (0, 0.2-0.3).
Per instance:
  seen   whale mask (eroded 2 px so white water along the outline is dropped) inside the canvas, Y < 0.62 (blades)
  n      brightness normalised per photo: (L - p5 on the whale) / (p99.5 of the whole cut-out - p5): 0 = the darkest skin,
         1 = the brightest thing in the photo (spray, white pigment); exposure-free to first order
  white  n > 0.5 ("light"), the drawing's pigment layer; dark marks inside white = n < 0.3
  side   'ventral' / 'dorsal' / None: from the notch position relative to the tip line IN THE UPRIGHT PHOTO (notch below
         the line = tail raised and seen from behind = underside towards the camera; above = fluke low and seen from
         above) and the parts.py CSE flag 'underside' (both must agree; see sides())
  drawable  both tips valid and near the mask, span >= 250 RAW px, blade area >= AREA_MIN of span^2
Per cluster (consensus): the drawable instances of the cluster's majority side, per pixel the median n over the frames
that saw it, kept where at least half the frames (>= 1) saw it; 'stable' = the share of frames that agree with the median
class (light / dark) at that pixel.
"""
import functools
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import bench  # noqa: E402
from prep import CACHE, SHEETS  # noqa: E402

W, H = 256, 128
X0, X1, Y0, Y1 = -1.15, 1.15, -0.35, 0.80
PX = W / (X1 - X0)
YB = 0.62                    # blades only above this
AREA_MIN = 0.045             # blade area / span^2 (half-span units: area in (half-span)^2 / 4)
SPAN_MIN = 250               # RAW px
POINTS = json.load(open('/home/sat3737/NOAA_whales/moonshot_cache/aligned/fluke_points.json'))


def sim(src, dst):
    s0, s1 = complex(*src[0]), complex(*src[1]); d0, d1 = complex(*dst[0]), complex(*dst[1])
    a = (d1 - d0) / (s1 - s0 + 1e-9); b = d0 - a * s0
    return np.array([[a.real, -a.imag, b.real], [a.imag, a.real, b.imag]])


def to_px(X, Y):
    return ((X - X0) * PX, (Y - Y0) * PX)


@functools.lru_cache(4)
def load(day):
    z = np.load(CACHE / 'prep' / f'{day}.npz')
    return dict(im=z['im'], whale=z['whale'], fluke=z['fluke']), json.load(open(CACHE / 'prep' / f'{day}.json'))


def photo_notch(key):
    """Signed notch offset from the tip line in the upright 2560-px photo, in half-spans (+ = notch below the line)."""
    r = POINTS.get(key)
    if r is None:
        return np.nan
    L, R, N = (np.array(r[k], float) for k in ('L', 'R', 'N'))
    if L[0] > R[0]:
        L, R = R, L
    v = R - L; s = np.linalg.norm(v) / 2
    if s < 5:
        return np.nan
    nrm = np.array([-v[1], v[0]]) / (2 * s)          # points down in the image when the tip line is level
    return float((N - (L + R) / 2) @ nrm / s)


def instance(arr, m, i):
    """Canonical drawing of one instance: dict(seen, n, ok, why, side cues) or None-ish with ok False."""
    p = np.array(m['pts'], float)
    out = dict(ok=False, why='', key=m['key'], cluster=m['cluster'], animal=m['animal'], underside=m['underside'])
    out['notch'] = photo_notch(m['key'])
    whale = arr['whale'][i] > 127
    if whale.sum() < 50:
        out['why'] = 'no mask'; return out
    dt = cv2.distanceTransform((~whale).astype(np.uint8), cv2.DIST_L2, 3)
    a = b = None
    if np.isfinite(p[:2]).all():
        a, b = (p[0], p[1]) if p[0][0] <= p[1][0] else (p[1], p[0])
        span384 = np.linalg.norm(b - a)
        dtip = max(dt[int(q[1]), int(q[0])] if (0 <= q[0] < 384 and 0 <= q[1] < 384) else 1e9 for q in (a, b))
        out['tip_off'] = float(dtip / max(span384 / 2, 1))
        out['frame'] = 'points'
    if a is None or out['tip_off'] > 0.12 or span384 < 20:
        fm = arr['fluke'][i] > 127
        fm = fm if fm.sum() > 200 else whale
        ys, xs = np.nonzero(fm)
        lo, hi = np.percentile(xs, 0.5), np.percentile(xs, 99.5)
        a = np.array([lo, ys[xs <= lo].mean()]); b = np.array([hi, ys[xs >= hi].mean()])
        span384 = np.linalg.norm(b - a)
        out['tip_off'] = 0.0
        out['frame'] = 'mask'
        out['cut'] = bool(xs.min() <= 2 or xs.max() >= 381)
    out['span_raw'] = float(span384 * 4 / m['scale'])
    M = sim([a, b], [to_px(-1, 0), to_px(1, 0)])
    im = cv2.warpAffine(arr['im'][i], M, (W, H), flags=cv2.INTER_AREA, borderValue=0)
    wk = cv2.warpAffine(arr['whale'][i], M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0) > 127
    inside = cv2.warpAffine(np.full((384, 384), 255, np.uint8), M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0) > 0
    wk = cv2.erode(wk.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    yy = (np.arange(H) / PX + Y0)[:, None]
    seen = wk & inside & (yy < YB)
    Lc = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    Lall = cv2.cvtColor(arr['im'][i], cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    if seen.sum() < 200:
        out.update(seen=seen, n=np.zeros((H, W), np.float32), im=im, why='little seen'); return out
    p5 = np.percentile(Lc[seen], 5); top = np.percentile(Lall, 99.5)
    n = np.clip((Lc - p5) / max(top - p5, 20.0), 0, 1.2)
    area = seen.sum() / PX ** 2                         # in half-span^2
    out.update(seen=seen, n=n.astype(np.float32), im=im, M=M, area=float(area / 4.0), p5=float(p5), top=float(top))
    if out.get('cut'):
        out['why'] = 'cut by the frame'
    elif out['span_raw'] < SPAN_MIN:
        out['why'] = 'small'
    elif out['area'] < AREA_MIN:
        out['why'] = 'edge-on / mostly hidden'
    else:
        out['ok'] = True
    return out


def side_of(r):
    """'ventral' / 'dorsal' / None for one instance (geometry and the CSE flag must agree)."""
    g = r.get('notch', np.nan)
    if not np.isfinite(g):
        return None
    if g > 0.08 and r['underside']:
        return 'ventral'
    if g < -0.02 and not r['underside']:
        return 'dorsal'
    return None


def day_instances(day):
    arr, meta = load(day)
    return [instance(arr, m, i) for i, m in enumerate(meta)]


def consensus(rs):
    """One drawing from the drawable frames of a cluster (same side). Returns dict or None."""
    rs = [r for r in rs if r['ok']]
    if not rs:
        return None
    S = np.stack([r['seen'] for r in rs]).astype(np.float32)
    N = np.stack([r['n'] for r in rs])
    k = len(rs)
    cnt = S.sum(0)
    need = max(1, int(np.ceil(k / 2)))
    seen = cnt >= need
    Nm = np.where(S > 0, N, np.nan)
    with np.errstate(all='ignore'):
        import warnings
        warnings.simplefilter('ignore')
        med = np.nanmedian(Nm, axis=0) if k > 1 else N[0]
    med = np.where(seen, np.nan_to_num(med), 0).astype(np.float32)
    lite = (N > 0.5)
    agree = np.where(S > 0, lite == (med > 0.5)[None], False).sum(0) / np.maximum(cnt, 1)
    return dict(seen=seen, n=med, stable=agree.astype(np.float32), frames=k, keys=[r['key'] for r in rs], area=float(seen.sum() / PX ** 2 / 4))
