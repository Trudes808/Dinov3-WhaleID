"""aligned (2026-11-18): canonical, position-by-position comparison of dorsal fins (fin + hump) and flukes.

Keypoints: the pose-side point model B_both (frozen DINOv3 + heatmap head, trained on run05 renders + the clicks of two
train days), already predicted for every instance of the 21 train + 3 val days (/var/tmp/sat3737-parts-cache/pose_side
pred_Bboth_train.json, pred2_Bboth_val.json; 2560-px image coordinates). No eval day is read (bench refuses them).

Dorsal canonical frame (units H = |tip - back base|, y down, head to the LEFT):
    back base -> (0, 0), tip -> TIP_C, similarity from these two points; mirrored when the front base lands right of
    the back base (the head side then sits right).  Tile x in [X0, X1] H, y in [Y0, Y1] H.
"""
from __future__ import annotations

import functools
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import bench  # noqa: E402

CACHE = Path('/home/sat3737/NOAA_whales/moonshot_cache/aligned')
SHEETS = Path('/tmp/claude-1072903897/-home-sat3737-NOAA-whales/06180765-15d8-433b-8ac1-13d344adb66e/scratchpad/aligned')
POSE = Path('/var/tmp/sat3737-parts-cache/pose_side')
DAYS = bench.DAYS

# dorsal tile
TIP_C = np.array([-0.25, -1.0])
X0, X1, Y0, Y1 = -3.4, 1.0, -1.3, 0.9
PXH = 128                       # tile pixels per fin height  -> 563 x 282, rounded to the patch grid below
TW, TH = 560, 288               # 35 x 18 patches of 16 px


@functools.lru_cache(None)
def keypoints():
    P, M = {}, {}
    for f in ('pred_Bboth_train', 'pred2_Bboth_val'):
        P.update(json.load(open(POSE / f'{f}.json')))
        M.update(json.load(open(POSE / f'{f}_meta.json'))['rows'])
    return P, M


@functools.lru_cache(None)
def photos(day):
    return json.load(open(DAYS / bench._ok(day) / 'safe' / 'photos.json'))


def head_side(day, key):
    return photos(day).get(key.rsplit('~', 1)[0], {}).get('head_side')


def _sim(src, dst):
    """2x3 similarity mapping src[0]->dst[0], src[1]->dst[1] (complex-number form)."""
    s0, s1 = complex(*src[0]), complex(*src[1]); d0, d1 = complex(*dst[0]), complex(*dst[1])
    a = (d1 - d0) / (s1 - s0 + 1e-9); b = d0 - a * s0
    return np.array([[a.real, -a.imag, b.real], [a.imag, a.real, b.imag]])


def dorsal_frame(key):
    """(M 2x3 image->tile px, info) or None.  info: H (px), mirrored, base ratio |F-B|/H, F in canonical units."""
    P, META = keypoints()
    p = P.get(key)
    if p is None:
        return None
    B, T, F = (np.array(p[n], float) for n in ('dorsal_fin_back', 'dorsal_fin_tip', 'dorsal_fin_front'))
    H = float(np.linalg.norm(T - B))
    if H < 4:
        return None
    to_tile = lambda q: np.array([(q[0] - X0) * PXH, (q[1] - Y0) * PXH])        # canonical units -> tile px
    M = _sim([B, T], [to_tile([0, 0]), to_tile(TIP_C)])
    Fc = (M[:, :2] @ F + M[:, 2]) / PXH + np.array([X0, Y0])
    mirrored = bool(Fc[0] > 0)
    if mirrored:                                                                    # mirror about the B-T line in canonical units
        M = _sim([B, T], [to_tile([0, 0]), to_tile(TIP_C)])
        # reflect image first: build similarity with reflection = sim on mirrored image coordinates
        R = np.array([[-1.0, 0, 0], [0, 1.0, 0]])                                  # x -> -x
        Bm, Tm = R[:, :2] @ B, R[:, :2] @ T
        Ms = _sim([Bm, Tm], [to_tile([0, 0]), to_tile(TIP_C)])
        M = Ms @ np.vstack([R, [0, 0, 1]])
        Fc = (M[:, :2] @ F + M[:, 2]) / PXH + np.array([X0, Y0])
    conf = META[key]['conf']
    return M, dict(H=H, mirrored=mirrored, base=float(np.linalg.norm(F - B) / H), Fc=Fc.tolist(),
                   conf=[conf['dorsal_fin_back'], conf['dorsal_fin_tip'], conf['dorsal_fin_front']])


@functools.lru_cache(maxsize=4)
def _img(day, stem):
    return cv2.cvtColor(cv2.imread(str(DAYS / day / 'images' / f'{stem}.jpg')), cv2.COLOR_BGR2RGB)


@functools.lru_cache(maxsize=4)
def _mask(day, stem):
    m = cv2.imread(str(DAYS / day / 'canon' / f'{stem}_mask.png'), cv2.IMREAD_GRAYSCALE)
    return None if m is None else (m > 0).astype(np.uint8) * 255


def warp(day, key, M, size=(TW, TH)):
    stem = key.rsplit('~', 1)[0]
    im, m = _img(day, stem), _mask(day, stem)
    big = abs(np.linalg.det(M[:, :2])) < 1                                         # downsampling -> area
    t = cv2.warpAffine(im, M, size, flags=cv2.INTER_AREA if big else cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    inside = cv2.warpAffine(np.full(im.shape[:2], 255, np.uint8), M, size, flags=cv2.INTER_NEAREST, borderValue=0)
    mt = cv2.warpAffine(m, M, size, flags=cv2.INTER_LINEAR, borderValue=0) if m is not None else np.zeros(size[::-1], np.uint8)
    return t, mt, inside
