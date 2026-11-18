"""A straight-horizon finder that treats real photos and renders alike.

The horizon is taken as the straight line (tilt within +-8 deg) across which brightness steps the same
way across the whole frame: the frame is cut into 16 column strips, and for every candidate line the
mean L* just above and just below it is compared in each strip. A line is accepted when the median step
is large against the frame's spread and at least 13 of the 16 strips agree in sign. Whale and ignore
pixels are left out. Above the line = sky, or land where there is land.

`validate` checks it on renders against the horizon from the render camera (geometry).
"""

from __future__ import annotations

import math
import warnings

import cv2
import numpy as np

WORK_W = 320
ANGLES = np.arange(-8.0, 8.01, 0.5)
STRIPS = 16
HALF = 5          # rows above / below compared
GAP = 1


def _strip_cums(img, ok, edges):
    v = np.where(ok, img, 0.0)
    rs = np.stack([v[:, a:c].sum(1) for a, c in zip(edges[:-1], edges[1:])], 1)
    return np.vstack([np.zeros((1, rs.shape[1])), np.cumsum(rs, 0)])


def detect(lab: np.ndarray, invalid: np.ndarray | None = None, whale_box=None, min_z: float = 0.35, min_agree: float = 0.8,
           max_tex_ratio: float = 0.75, min_dab: float = 8.0, strong_z: float = 0.9, max_land_b: float = -8.0, smooth_sky: float = 0.5):
    """lab: (H, W, 3) float CIELAB (or (H, W) L* only); invalid: pixels to leave out (whales); whale_box:
    main whale xyxy in these pixels. A candidate line must
      * step in L* the same way in >= 80 % of the 16 strips, by >= min_z x the frame's L* spread,
      * have a smoother band above than below (fine-texture ratio < max_tex_ratio: sky / distant land),
      * pass above the middle of the whale box at the whale's column (on renders 95 % of true horizons
        do); this rejects foam lines at the waterline,
      * change colour across it (median |da*| + |db*| >= min_dab: sea against sky or land), unless the
        L* step is very strong (>= strong_z), or it is brighter AND much smoother above (texture ratio
        < smooth_sky: hazy sky of the sea's hue) -- wave crests and foam lines keep the sea's hue,
      * be brighter above (sky), or, if darker above, not blue above (b* >= max_land_b: land).
    Returns (y per column or None, info dict)."""
    if lab.ndim == 2:
        lab = np.stack([lab, np.zeros_like(lab), np.zeros_like(lab)], -1)
    H, W = lab.shape[:2]
    s = WORK_W / W
    h = max(int(round(H * s)), 8)
    small = cv2.resize(lab.astype(np.float32), (WORK_W, h), interpolation=cv2.INTER_AREA)
    Ls = small[..., 0]
    tex = cv2.GaussianBlur(np.abs(Ls - cv2.GaussianBlur(Ls, (0, 0), 1.0)), (0, 0), 1.5)
    bad = np.zeros((h, WORK_W), bool)
    if invalid is not None and invalid.any():
        inv = cv2.resize(invalid.astype(np.uint8), (WORK_W, h), interpolation=cv2.INTER_NEAREST)
        bad = cv2.dilate(inv, np.ones((5, 5), np.uint8)) > 0
    spread = float(np.subtract(*np.percentile(Ls[~bad] if (~bad).sum() > 100 else Ls, [90, 10]))) + 2.0
    xs = np.arange(WORK_W, dtype=np.float32)
    best = (-1.0, None, None, 0.0, 0.0, None, None, None)
    edges = np.linspace(0, WORK_W, STRIPS + 1).astype(int)
    TH = 8
    if whale_box is not None:
        wx = 0.5 * (whale_box[0] + whale_box[2]) * s
        wy = (whale_box[3] - 0.5 * (whale_box[3] - whale_box[1])) * s
    ys = np.arange(HALF + GAP, h - HALF - GAP)
    for ang in ANGLES:
        t = math.tan(math.radians(ang))
        mapx = np.tile(xs[None, :], (h, 1))
        mapy = (np.arange(h, dtype=np.float32)[:, None] + (xs[None, :] - WORK_W / 2) * t).astype(np.float32)
        b = cv2.remap(bad.astype(np.float32), mapx, mapy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=1) > 0.5
        chans = [cv2.remap(small[..., c], mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan) for c in range(3)]
        tx = cv2.remap(tex, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
        ok = ~np.isnan(chans[0]) & ~b
        cn = _strip_cums(np.ones_like(Ls), ok, edges)
        a_n = cn[ys - GAP] - cn[ys - GAP - HALF]; b_n = cn[ys + GAP + HALF] - cn[ys + GAP]
        good = (a_n >= HALF * 3) & (b_n >= HALF * 3)
        diffs, above = [], []
        for img in chans:
            cs = _strip_cums(np.nan_to_num(img), ok, edges)
            a_s = cs[ys - GAP] - cs[ys - GAP - HALF]; b_s = cs[ys + GAP + HALF] - cs[ys + GAP]
            diffs.append(np.where(good, a_s / np.maximum(a_n, 1) - b_s / np.maximum(b_n, 1), np.nan))
            above.append(np.where(good, a_s / np.maximum(a_n, 1), np.nan))
        ct = _strip_cums(np.nan_to_num(tx), ok, edges)
        ya0 = np.clip(ys - GAP - 1 - TH, 0, h); ya1 = np.clip(ys - GAP - 1, 0, h)
        yb0 = np.clip(ys + GAP + 2, 0, h); yb1 = np.clip(ys + GAP + 2 + TH, 0, h)
        ta = (ct[ya1] - ct[ya0]).sum(1) / np.maximum((cn[ya1] - cn[ya0]).sum(1), 1)
        tb = (ct[yb1] - ct[yb0]).sum(1) / np.maximum((cn[yb1] - cn[yb0]).sum(1), 1)
        tratio = ta / np.maximum(tb, 1e-3)
        ngood = good.sum(1)
        d = diffs[0]
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            med = np.nanmedian(d, 1)
            agree = np.nansum(np.sign(d) == np.sign(med)[:, None], 1) / np.maximum(ngood, 1)
            dab = np.abs(np.nanmedian(diffs[1], 1)) + np.abs(np.nanmedian(diffs[2], 1))
            b_above = np.nanmedian(above[2], 1)
        z = np.nan_to_num(np.abs(med) / spread, nan=-1)
        dab = np.nan_to_num(dab, nan=0)
        keep = (ngood >= STRIPS * 0.6) & (agree >= min_agree) & (tratio < max_tex_ratio) & (z >= min_z) & ((dab >= min_dab) | (z >= strong_z) | ((med > 0) & (tratio < smooth_sky)))
        # brighter above = sky; darker above must look like land (not blue: b* above >= max_land_b)
        keep &= (med > 0) | (np.nan_to_num(b_above, nan=-99) >= max_land_b)
        if whale_box is not None:
            keep &= (ys + (wx - WORK_W / 2) * t) < wy
        score = np.where(keep, z + dab / 20.0, -1)
        i = int(np.argmax(score))
        if score[i] > best[0]:
            best = (float(score[i]), float(ys[i]), float(ang), float(agree[i]), float(med[i]), float(tratio[i]), float(z[i]), float(dab[i]))
    sc, y, ang, agree, med, tr, zz, db = best
    info = dict(score=round(sc, 3), z=None if zz is None else round(zz, 3), dab=None if db is None else round(db, 2), angle_deg=ang,
                agree=round(agree, 3), step=None if med is None else round(med, 2), tex_ratio=None if tr is None else round(tr, 3))
    if y is None or sc < 0:
        return None, info
    t = math.tan(math.radians(ang))
    cols = np.arange(W) * s
    yc = (y + (cols - WORK_W / 2) * t) / s
    info["y_centre"] = float(y / s)
    return yc.astype(np.float32), info


def render_truth(meta: dict, width: int, height: int, roll_sign: float = 1.0):
    """Horizon row per column from the render camera (sea-level plane at infinity), or None if the
    horizon is outside the frame. Needs the factory package on sys.path."""
    import sys
    sys.path.insert(0, "/home/sat3737/NOAA_whales/Whale-Image-Data-Factory")
    from factory.project import Camera
    cam = meta["camera"]
    rot = dict(cam["rotation"]); rot["roll"] = roll_sign * rot.get("roll", 0.0)
    c = Camera(cam["location"], rot, cam["focal_mm"], width, height)
    yaw = math.radians(cam["rotation"]["yaw"])
    phis = np.radians(np.linspace(-20, 20, 81))
    pts = np.stack([np.cos(yaw + phis), np.sin(yaw + phis), np.zeros_like(phis)], 1) * 1e10 + c.location
    uv, depth = c.project(pts)
    keep = depth > 0
    uv = uv[keep]
    A = np.polyfit(uv[:, 0], uv[:, 1], 1)
    y = np.polyval(A, np.arange(width)).astype(np.float32)
    if (y < 0).all() or (y > height).all():
        return None
    return y
