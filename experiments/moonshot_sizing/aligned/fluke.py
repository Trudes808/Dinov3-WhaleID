"""Fluke canonical frame from the two-peak decoded tips of the pose-side head B_both (tips named by their side in the
photo, as decode_tips 'image'), notch as a check.

Canonical units: half-span s = |R - L| / 2; L -> (-1, 0), R -> (1, 0), y down; when the notch is above the tip line the
frame is flipped top-bottom so the notch is always below (low flukes seen dorsal side up; the photo's left tip stays left).  Tile x in [-1.15, 1.15], y in [-0.35, 0.95].
    python fluke.py decode      writes CACHE/fluke_points.json from the probability maps (CPU)
"""
import functools
import json
import sys

import numpy as np

from common import CACHE, POSE, _sim

FX0, FX1, FY0, FY1 = -1.15, 1.15, -0.35, 0.95
TW, TH = 560, 320
PXS = TW / (FX1 - FX0)                       # tile px per half-span (243.5)


@functools.lru_cache(None)
def points():
    return json.load(open(CACHE / 'fluke_points.json'))


def fluke_frame(key):
    r = points().get(key)
    if r is None:
        return None
    L, R, N = (np.array(r[n], float) for n in ('L', 'R', 'N'))
    s = np.linalg.norm(R - L) / 2
    if s < 8:
        return None
    to_tile = lambda q: np.array([(q[0] - FX0) * PXS, (q[1] - FY0) * PXS])
    M = _sim([L, R], [to_tile([-1, 0]), to_tile([1, 0])])
    Nc = (M[:, :2] @ N + M[:, 2]) / PXS + np.array([FX0, FY0])
    turned = bool(Nc[1] < 0)
    if turned:                               # flip top-bottom (keeps the photo's left tip on the left)
        Rf = np.array([[1.0, 0, 0], [0, -1.0, 0]])
        M = _sim([Rf[:, :2] @ L, Rf[:, :2] @ R], [to_tile([-1, 0]), to_tile([1, 0])]) @ np.vstack([Rf, [0, 0, 1]])
        Nc = (M[:, :2] @ N + M[:, 2]) / PXS + np.array([FX0, FY0])
    return M, dict(S=float(s), turned=turned, Nc=Nc.tolist(), second=r['second'], conf=r['conf'])


def decode():
    sys.path.insert(0, '/home/sat3737/NOAA_whales/Dinov3-WhaleID')
    from model_fine_tuning.pose_side.decode_tips import decode_one
    out = {}
    for d in (CACHE / 'fluke_maps', POSE / 'maps_Bboth_val'):
        for f in sorted(d.glob('*.npz')):
            z = np.load(f); M = z['M']; Mi = np.linalg.inv(np.vstack([M, [0, 0, 1]]))[:2]
            pts, conf, model, image, second = decode_one(z['prob'])
            to = lambda p: [float(v) for v in Mi[:, :2] @ p + Mi[:, 2]]
            out[f.stem] = dict(L=to(image[0]), R=to(image[1]), N=to(pts['fluke_notch']), second=float(second),
                               conf=float(min(conf['fluke_tip_left'], conf['fluke_tip_right'], conf['fluke_notch'])))
    json.dump(out, open(CACHE / 'fluke_points.json', 'w'))
    print(len(out))


if __name__ == '__main__':
    if sys.argv[1] == 'decode':
        decode()
