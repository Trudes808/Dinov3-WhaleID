"""local_match (2026-11-18): learned local features + geometric verification on the full-resolution part cut-outs.

Per instance that shows the part: the parts_up chart (1536 px, from the RAW; fins mirrored to one flank, never upside
down), keypoints only on whale pixels (whale mask dilated a little so that outline points are kept), ALIKED-n16 (or
DISK) descriptors, also on the horizontally mirrored chart for fins. Matching: LightGlue, batched with padding masks.
Scores per instance pair = inliers of a RANSAC similarity (scale / rotation bounded, the charts are pose-normalised),
inliers of a homography, raw match count, summed LightGlue confidence over the similarity inliers, inlier ratio and
the inlier spread (convex-hull area of inliers / whale area). Cluster pair = max over its instance pairs.

Caches under /home/sat3737/NOAA_whales/moonshot_cache/local_match/cache (nothing big in the repo).
"""
import functools
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import bench  # noqa: E402

CACHE = Path('/home/sat3737/NOAA_whales/moonshot_cache/local_match/cache')
SUFFIX = {'dorsal': 'fin', 'fluke': 'fluke'}


def chart_paths(day, key, part):
    d = bench.DAYS / day / 'parts_up'
    return d / 'charts' / f'{key}__{SUFFIX[part]}.jpg', d / f'{key}__{SUFFIX[part]}.npz'


@functools.lru_cache(None)
def meta(day, key, part):
    jp, zp = chart_paths(day, key, part)
    if not (jp.exists() and zp.exists()):
        return None
    z = np.load(zp, allow_pickle=True)
    return dict(pixels=int(z['pixels']), view_side=int(z['view_side']), mirrored=bool(z['mirrored']))


def box_crop(day, key, size, margin=0.15):
    """Fallback for an instance without a parts_up cut-out: square crop of its box in the 2560-px copy + canon mask."""
    img = cv2.cvtColor(cv2.imread(str(bench.image_path(day, key))), cv2.COLOR_BGR2RGB)
    mp = bench.DAYS / day / 'canon' / f"{key.rsplit('~', 1)[0]}_mask.png"
    m = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED) if mp.exists() else np.ones(img.shape[:2], np.uint8)
    x0, y0, x1, y1 = bench.box(day, key)
    c, h = ((x0 + x1) / 2, (y0 + y1) / 2), max(x1 - x0, y1 - y0) * (1 + 2 * margin) / 2
    s = size / (2 * h)
    M = np.array([[s, 0, size / 2 - s * c[0]], [0, s, size / 2 - s * c[1]]])
    im = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    wm = cv2.warpAffine((m > 0).astype(np.uint8), M, (size, size), flags=cv2.INTER_NEAREST) > 0
    bm = np.zeros_like(wm)
    bx = (np.array([x0, y0, x1, y1]) * s + [M[0, 2], M[1, 2], M[0, 2], M[1, 2]]).astype(int).clip(0, size)
    bm[bx[1]:bx[3], bx[0]:bx[2]] = True
    return im, wm & bm, wm & bm


def load_chart(day, key, part, size):
    jp, zp = chart_paths(day, key, part)
    if not jp.exists():
        return box_crop(day, key, size)
    img = cv2.cvtColor(cv2.imread(str(jp), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    z = np.load(zp, allow_pickle=True)
    whale = z['whale'].astype(np.uint8)
    pm = z['fin_band' if part == 'dorsal' else 'fluke'].astype(np.uint8)
    if img.shape[0] != size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        whale = cv2.resize(whale, (size, size), interpolation=cv2.INTER_NEAREST)
        pm = cv2.resize(pm, (size, size), interpolation=cv2.INTER_NEAREST)
    return img, whale > 0, pm > 0


def sharpness(img, mask):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    m = cv2.erode(mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    return float(lap[m].var()) if m.sum() > 50 else 0.0


def pick(day, part, keys, n, sharp):
    """Up to n instances of a cluster: largest part (pixels in the RAW crop), sharpness as the second criterion."""
    have = [k for k in keys if meta(day, k, part)]
    sc = {k: math.log(meta(day, k, part)['pixels'] + 1) + 0.5 * math.log(sharp.get(k, 1.0) + 1) for k in have}
    return sorted(have, key=lambda k: -sc[k])[:n]


# ---- batched LightGlue -----------------------------------------------------------------------------------------------
@torch.no_grad()
def lg_batch(lg, kp0, d0, v0, kp1, d1, v1, size):
    """kp (B, L, 2) pixel coords, d (B, L, D), v (B, L) valid. Returns list of (matches (M, 2), scores (M,))."""
    from kornia.feature.lightglue import filter_matches, normalize_keypoints
    s = torch.tensor([[float(size), float(size)]], device=kp0.device).expand(kp0.shape[0], 2)
    k0, k1 = normalize_keypoints(kp0, s), normalize_keypoints(kp1, s)
    m0, m1 = v0[:, None, :, None], v1[:, None, :, None]
    desc0, desc1 = lg.input_proj(d0), lg.input_proj(d1)
    e0, e1 = lg.posenc(k0), lg.posenc(k1)
    for i in range(lg.conf.n_layers):
        desc0, desc1 = lg.transformers[i](desc0, desc1, e0, e1, mask0=m0, mask1=m1)
    la = lg.log_assignment[lg.conf.n_layers - 1]
    md0, md1 = la.final_proj(desc0), la.final_proj(desc1)
    dd = md0.shape[-1]
    sim = torch.einsum('bmd,bnd->bmn', md0 / dd ** 0.25, md1 / dd ** 0.25).float()
    pair = v0[:, :, None] & v1[:, None, :]
    sim = sim.masked_fill(~pair, -1e4)                      # padded rows / columns take no softmax mass
    z0, z1 = la.matchability(desc0).float(), la.matchability(desc1).float()
    cert = torch.nn.functional.logsigmoid(z0) + torch.nn.functional.logsigmoid(z1).transpose(1, 2)
    s0 = torch.nn.functional.log_softmax(sim, 2)
    s1 = torch.nn.functional.log_softmax(sim, 1)
    L0, L1 = kp0.shape[1], kp1.shape[1]
    sc = sim.new_full((kp0.shape[0], L0 + 1, L1 + 1), 0)
    sc[:, :L0, :L1] = (s0 + s1 + cert).masked_fill(~pair, -1e4)
    sc[:, :-1, -1] = torch.nn.functional.logsigmoid(-z0.squeeze(-1))
    sc[:, -1, :-1] = torch.nn.functional.logsigmoid(-z1.squeeze(-1))
    mm0, _, ms0, _ = filter_matches(sc, lg.conf.filter_threshold)
    out = []
    for k in range(kp0.shape[0]):
        ok = (mm0[k] > -1) & v0[k]
        i0 = torch.where(ok)[0]
        out.append((torch.stack([i0, mm0[k][ok]], -1).cpu().numpy(), ms0[k][ok].float().cpu().numpy()))
    return out


# ---- geometry ----------------------------------------------------------------------------------------------------------
def verify(pa, pb, conf, area_a, thr, smin=0.5, smax=2.0, rmax=30.0):
    """Stats of one matched instance pair (pa, pb (M, 2) pixel coords)."""
    n = len(pa)
    out = dict(n=n, sim=0, simc=0.0, hom=0, aff=0, spread=0.0)
    if n < 4:
        return out, None
    M, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=thr, maxIters=2000, confidence=0.995)
    if M is not None:
        s = math.hypot(M[0, 0], M[1, 0])
        r = abs(math.degrees(math.atan2(M[1, 0], M[0, 0])))
        if smin <= s <= smax and r <= rmax:
            inl = inl.ravel().astype(bool)
            out['sim'] = int(inl.sum())
            out['simc'] = float(conf[inl].sum())
            if inl.sum() >= 3:
                out['spread'] = float(cv2.contourArea(cv2.convexHull(pa[inl].astype(np.float32)))) / max(area_a, 1.0)
        else:
            inl = None
    else:
        inl = None
    Ma, ia = cv2.estimateAffine2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=thr, maxIters=2000, confidence=0.995)
    if Ma is not None:
        out['aff'] = int(ia.sum())
    if n >= 5:
        H, ih = cv2.findHomography(pa, pb, cv2.USAC_MAGSAC, thr, maxIters=2000, confidence=0.995)
        if H is not None:
            out['hom'] = int(ih.sum())
    return out, inl
