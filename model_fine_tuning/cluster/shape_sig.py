"""Shape signatures: the dorsal fin outline and the fluke trailing edge, scored against the day's population.

The researchers' first cues are the shape of the dorsal fin and of the fluke's trailing edge (notches, nicks). For
every fin chart the isolated fin silhouette (cluster/parts.py) gives the outline from the leading base over the tip to
the trailing base (cluster/finshape.outline_descriptor: 64 points, normalised by the fin height); for every fluke chart
the blades' lower boundary sampled over 64 columns of the span (peduncle up, right tip on the right) gives the trailing
edge profile. Per day, each point of the descriptor has a population median and spread; a point's uniqueness is its
robust z-score against the day's other whales (0..1 after scaling by the day's 95th percentile), so a notch that no
other whale has lights up. Two shapes are compared by the uniqueness-weighted mean point distance; tracklet pairs take
the best chart pair. Used by cluster/signature.py (--shape): outlines with uniqueness for the web app and shape link
scores next to the token signature.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .finshape import outline_descriptor

N_PTS = 64


def fin_outline(npz: Path, min_height: int = 40) -> tuple[np.ndarray, np.ndarray] | None:
    """(points in chart pixels (64, 2), descriptor (128,)) of the isolated fin, or None when the fin is too small."""
    z = np.load(npz)
    sil = z["silhouette"]
    if sil.sum() < 300:
        return None
    r = outline_descriptor(sil, N_PTS)
    if r is None:
        return None
    norm, _, stats = r
    if stats["height"] < min_height:
        return None
    pts = norm.reshape(-1, 2) * stats["height"] + np.array(stats["tip"])
    return pts.astype(np.float32), norm


def fluke_edge(npz: Path, min_span: int = 80) -> tuple[np.ndarray, np.ndarray] | None:
    """(points in chart pixels (64, 2), descriptor (64,)) of the fluke's trailing (lower) edge over the span."""
    z = np.load(npz)
    m = z["fluke"]
    ys, xs = np.nonzero(m)
    if len(xs) < 300:
        return None
    x0, x1 = xs.min(), xs.max()
    if x1 - x0 < min_span:
        return None
    cols = np.linspace(x0, x1, N_PTS)
    pts = []
    for c in cols:
        col = m[:, int(round(c))]
        yy = np.nonzero(col)[0]
        pts.append([c, yy.max() if len(yy) else np.nan])
    pts = np.array(pts, np.float32)
    if np.isnan(pts[:, 1]).mean() > 0.2:
        return None
    pts[:, 1] = np.interp(cols, cols[np.isfinite(pts[:, 1])], pts[np.isfinite(pts[:, 1]), 1])
    span = float(x1 - x0)
    base = np.polyval(np.polyfit(np.linspace(-1, 1, N_PTS), pts[:, 1], 2), np.linspace(-1, 1, N_PTS))     # the smooth sweep of the edge
    desc = ((pts[:, 1] - base) / span).astype(np.float32)                                                   # notches = residual from the sweep
    return pts, desc


def uniqueness(D: np.ndarray, groups: np.ndarray, n_pts: int) -> np.ndarray:
    """Per descriptor and per outline point, the robust z of that point against the other whales (other groups):
    population median / MAD over all descriptors (one whale's many charts barely move a median), scaled so the day's
    95th percentile is 1. D (n, k * n_pts) with k coordinates per point."""
    med = np.median(D, 0); mad = np.median(np.abs(D - med), 0) * 1.4826 + 1e-4
    z = np.abs(D - med) / mad
    k = D.shape[1] // n_pts
    zp = np.sqrt((z.reshape(len(D), n_pts, k) ** 2).sum(-1))                      # (n, n_pts)
    hi = np.percentile(zp, 95) if len(zp) else 1.0
    return np.clip(zp / max(hi, 1e-6), 0, 1).astype(np.float32)


def shape_distance(Da: np.ndarray, Db: np.ndarray, Ua: np.ndarray, Ub: np.ndarray, n_pts: int, floor: float = 0.2) -> float:
    k = Da.shape[0] // n_pts
    d = np.linalg.norm((Da - Db).reshape(n_pts, k), axis=1)
    w = 0.5 * (Ua + Ub) + floor
    return float((w * d).sum() / w.sum())


def compute(parts_dir: Path, charts: list[tuple[int, str, str]], tracklet_of_row: dict) -> dict:
    """charts: (row, key, part) for the fin and fluke charts. Returns per row {"pts", "u"} and per part the descriptor
    bank {"rows", "D", "U"} for the link scores."""
    out = {"rows": {}, "banks": {}}
    for part, fn in (("fin", fin_outline), ("fluke", fluke_edge)):
        rows, pts_l, D = [], [], []
        for row, key, pt in charts:
            if pt != part:
                continue
            r = fn(parts_dir / f"{key}__{part}.npz")
            if r is None:
                continue
            rows.append(row); pts_l.append(r[0]); D.append(r[1])
        if len(rows) < 5:
            continue
        D = np.stack(D); groups = np.array([tracklet_of_row.get(r, -1) for r in rows])
        U = uniqueness(D, groups, N_PTS)
        for r, p, u in zip(rows, pts_l, U):
            out["rows"][r] = {"part": part, "pts": p, "u": u}
        out["banks"][part] = {"rows": np.array(rows), "D": D, "U": U}
    return out


def tracklet_scores(shape: dict, part_rows: dict, n_tracks: int) -> dict:
    """Per part, (T, T) shape score (negative uniqueness-weighted distance of the best chart pair), NaN when a side has
    no usable shape; plus the mean of the per-part z-scores where available ("shape")."""
    out = {}
    for part, bank in shape["banks"].items():
        pos = {int(r): i for i, r in enumerate(bank["rows"])}
        S = np.full((n_tracks, n_tracks), np.nan, np.float32)
        tr_rows = {t: [pos[r] for r in part_rows.get((t, part), []) if r in pos] for t in range(n_tracks)}
        for a in range(n_tracks):
            if not tr_rows[a]:
                continue
            for b in range(a + 1, n_tracks):
                if not tr_rows[b]:
                    continue
                best = min(shape_distance(bank["D"][i], bank["D"][j], bank["U"][i], bank["U"][j], N_PTS) for i in tr_rows[a] for j in tr_rows[b])
                S[a, b] = S[b, a] = -best
        out[part] = S
    if out:
        Z = []
        for part, S in out.items():
            v = S[np.isfinite(S)]
            Z.append((S - v.mean()) / (v.std() + 1e-9) if len(v) > 1 else S)
        out["shape"] = np.nanmean(np.stack(Z), 0) if len(Z) > 1 else Z[0]
    return out
