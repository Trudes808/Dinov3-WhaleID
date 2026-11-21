"""Expanded triangle: the dorsal part (fin + hump) from the three fin points and a whale mask.

The triangle (tip T, front base F, back base B) fixes WHERE the dorsal part is and how it is turned; the mask then
gives its outline. Frame of one fin: "along" a = position on the base line from B (0) to F (1); "up" v = height above
the base line over the tip's height h (base line 0, tip 1). h_eff = max(h, base / 3) so that a nearly flat triangle
(foreshortened fin, three points close to a line) still gets a usable zone.

    E = (triangle AND mask)  OR  the pieces of (mask AND zone) that touch (triangle AND mask)
    zone = a in [-M_ALONG, 1 + M_ALONG]  AND  v >= -M_BELOW  AND  distance to the triangle <= D_NEAR x h_eff

M_ALONG: how far the part may reach beyond the two bases (fraction of the base); M_BELOW: how far under the base
line (fraction of h_eff); D_NEAR: how far outside the straight sides (fraction of h_eff), which takes in a curved
leading edge, a hooked tip and the hump's arc and stops the fill from running along another whale or the own fluke.
Parameters fixed on renders (experiments/parts_tri_expand/README.md); nothing here reads a photo.
"""
from __future__ import annotations

import cv2
import numpy as np

PARAMS = {"m_along": 0.10, "m_below": 0.15, "d_near": 0.35}     # overwritten by the frozen values below if present
FROZEN = None


def frame(T, F, B):
    T, F, B = (np.asarray(p, float) for p in (T, F, B))
    u = F - B
    base = float(np.hypot(*u))
    if base < 2:
        return None
    u = u / base
    w = (T - B) - ((T - B) @ u) * u
    h = float(np.hypot(*w))
    if h < 0.5:
        # three points on a line: take the side of the normal that points away from nothing in particular (image up)
        n = np.array([-u[1], u[0]])
        if n[1] > 0:
            n = -n
        h = 0.5
    else:
        n = w / h
    return {"u": u, "n": n, "base": base, "h": h, "h_eff": max(h, base / 3.0), "B": B, "F": F, "T": T}


def expand(T, F, B, mask: np.ndarray, m_along=None, m_below=None, d_near=None, return_parts=False):
    """E (bool, mask's shape) for one fin. mask: bool whale mask in the same pixel frame as the points."""
    p = dict(PARAMS)
    if FROZEN:
        p.update(FROZEN)
    m_along = p["m_along"] if m_along is None else m_along
    m_below = p["m_below"] if m_below is None else m_below
    d_near = p["d_near"] if d_near is None else d_near
    H, W = mask.shape
    fr = frame(T, F, B)
    out = np.zeros((H, W), bool)
    tri_full = np.zeros((H, W), bool)
    if fr is None:
        return (out, tri_full) if return_parts else out
    he = fr["h_eff"]
    pts = np.stack([fr["T"], fr["F"], fr["B"]])
    reach = (d_near + 0.5) * he + m_along * fr["base"] + 4
    x0, y0 = int(max(0, np.floor(pts[:, 0].min() - reach))), int(max(0, np.floor(pts[:, 1].min() - reach)))
    x1, y1 = int(min(W, np.ceil(pts[:, 0].max() + reach) + 1)), int(min(H, np.ceil(pts[:, 1].max() + reach) + 1))
    if x1 <= x0 or y1 <= y0:
        return (out, tri_full) if return_parts else out
    tri = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(tri, [np.round(pts - [x0, y0]).astype(np.int32)], 1)
    m = mask[y0:y1, x0:x1].astype(bool)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    Q = np.stack([xx, yy], -1).astype(np.float64) - fr["B"]
    al = Q @ fr["u"] / fr["base"]
    up = Q @ fr["n"] / he                       # in h_eff units below the line; only its sign / depth is used
    dist = cv2.distanceTransform((1 - tri).astype(np.uint8), cv2.DIST_L2, 5)
    zone = (al >= -m_along) & (al <= 1 + m_along) & (up >= -m_below) & (dist <= d_near * he)
    seed = (tri > 0) & m
    cand = (m & zone) | seed
    n_c, lab = cv2.connectedComponents(cand.astype(np.uint8), connectivity=8)
    ids = np.unique(lab[seed])
    ids = ids[ids > 0]
    e = np.isin(lab, ids) if len(ids) else seed
    out[y0:y1, x0:x1] = e
    tri_full[y0:y1, x0:x1] = tri > 0
    return (out, tri_full) if return_parts else out


def triangle(T, F, B, shape) -> np.ndarray:
    t = np.zeros(shape, np.uint8)
    cv2.fillPoly(t, [np.round(np.stack([T, F, B])).astype(np.int32)], 1)
    return t > 0
