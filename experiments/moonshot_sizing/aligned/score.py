"""Position-by-position comparison of canonical tiles (CPU).

Instance-pair scores per day (N x N), then cluster-pair score = max over the two clusters' instance pairs (as the bench
does for embeddings).  Variants (name -> function of the day's cache):
  pos       mean over positions of cos(token_a[p], token_b[p]), weighted by whale share in both cells (block 11)
  posc      same after subtracting the day's mean token at each position (the "every fin looks like this here" part)
  posc_nb   posc with a 3x3 neighbourhood tolerance (best neighbour in b for each cell of a, symmetrised)
  posc_top  posc, mean of the best quarter of the positions
  pos8c     posc on block 8, 2x2-pooled cells
  outline   minus the mean absolute difference of the fin + hump top-edge profiles (mirror-invariant by construction),
            best over small shifts
"""
from __future__ import annotations

import functools
import json

import cv2
import numpy as np

from common import CACHE, PXH, X0, Y0, bench

QUAL = dict(H=30, base=1.5)


@functools.lru_cache(maxsize=3)
def load(part, day):
    z = np.load(CACHE / part / f'{day}.npz')
    info = json.load(open(CACHE / part / f'{day}_info.json'))
    return dict(keys=list(z['keys']), z=z, info=info)


def cell_weights(mask, inside, gh, gw):
    """whale share per token cell x share inside the photo."""
    m = np.stack([cv2.resize(x.astype(np.float32) / 255, (gw, gh), interpolation=cv2.INTER_AREA) for x in mask])
    return m * inside.astype(np.float32)


def quality(part, day, qual=QUAL):
    L = load(part, day)
    if qual is None:
        return np.ones(len(L['keys']), bool)
    if part == 'fluke':
        return fluke_quality(day)
    return np.array([L['info'][k]['H'] >= qual['H'] and L['info'][k]['base'] >= qual['base'] for k in L['keys']])


@functools.lru_cache(maxsize=6)
def tokens(part, day, layer='tok11', center=True, wmin=0.3):
    L = load(part, day)
    T = L['z'][layer].astype(np.float32)                     # N gh gw D
    N, gh, gw, D = T.shape
    ins = L['z']['inside'].astype(np.float32)
    if ins.shape[1:] != (gh, gw):
        ins = np.stack([cv2.resize(x, (gw, gh), interpolation=cv2.INTER_AREA) for x in ins])
    W = cell_weights(L['z']['mask'], ins, gh, gw)            # N gh gw
    T = T.reshape(N, gh * gw, D); W = W.reshape(N, gh * gw)
    if center:
        w = (W > wmin).astype(np.float32)
        mu = (T * w[..., None]).sum(0) / np.maximum(w.sum(0), 1)[:, None]
        T = T - mu[None]
    T /= np.maximum(np.linalg.norm(T, axis=2, keepdims=True), 1e-6)
    W = np.where(W > wmin, W, 0.0)
    return T, W, (gh, gw)


def pos_scores(part, day, layer='tok11', center=True, mode='mean', nb=False):
    T, W, (gh, gw) = tokens(part, day, layer, center)
    N, P, D = T.shape
    if mode == 'mean' and not nb:
        num = np.zeros((N, N), np.float32); den = np.zeros((N, N), np.float32)
        for p in range(P):
            w = W[:, p]
            idx = np.nonzero(w)[0]
            if len(idx) < 2:
                continue
            G = T[idx, p] @ T[idx, p].T
            ww = np.outer(w[idx], w[idx])
            num[np.ix_(idx, idx)] += G * ww; den[np.ix_(idx, idx)] += ww
        return np.where(den > 0.5, num / np.maximum(den, 1e-6), np.nan), den
    if nb:                                                    # global shift tolerance: best of the 9 shifts of +-1 cell
        Tg = T.reshape(N, gh, gw, D); Wg = W.reshape(N, gh, gw)
        best = np.full((N, N), -np.inf, np.float32); bden = np.zeros((N, N), np.float32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                num = np.zeros((N, N), np.float32); den = np.zeros((N, N), np.float32)
                for y in range(max(0, -dy), min(gh, gh - dy)):
                    for x in range(max(0, -dx), min(gw, gw - dx)):
                        wa, wb = Wg[:, y, x], Wg[:, y + dy, x + dx]
                        ia, ib = np.nonzero(wa)[0], np.nonzero(wb)[0]
                        if not len(ia) or not len(ib):
                            continue
                        G = Tg[ia, y, x] @ Tg[ib, y + dy, x + dx].T
                        ww = np.outer(wa[ia], wb[ib])
                        num[np.ix_(ia, ib)] += G * ww; den[np.ix_(ia, ib)] += ww
                s = np.where(den > 0.5, num / np.maximum(den, 1e-6), -np.inf)
                upd = s > best
                best = np.where(upd, s, best); bden = np.where(upd, den, bden)
        best = np.maximum(best, best.T)
        return np.where(np.isfinite(best), best, np.nan), np.maximum(bden, bden.T)
    out = np.full((N, N), np.nan, np.float32); den = np.zeros((N, N), np.float32)
    for i in range(N):                                        # top quarter of the positions
        wi = W[i] > 0
        if wi.sum() < 3:
            continue
        c = np.einsum('pd,npd->np', T[i][wi], T[:, wi], optimize=True)
        ok = W[:, wi] > 0
        k = np.maximum(ok.sum(1) // 4, 1)
        s = np.where(ok, c, -np.inf); s = -np.sort(-s, 1)
        cs = np.cumsum(np.where(np.isfinite(s), s, 0), 1)
        out[i] = cs[np.arange(N), k - 1] / k
        den[i] = ok.sum(1)
    out = np.where(den >= 3, out, np.nan)
    return out, den


# ---- outline ---------------------------------------------------------------------------------------------------------
XS = np.arange(-2.9, 0.81, 0.05)                              # columns, fin heights


@functools.lru_cache(maxsize=3)
def profiles(day, refine=True):
    """top edge y (fin heights) per column XS for every dorsal tile; NaN where unknown."""
    L = load('dorsal', day)
    M, G = L['z']['mask'], L['z']['grey']
    out = np.full((len(M), len(XS)), np.nan, np.float32)
    cols = np.round((XS - X0) * PXH).astype(int)
    for i in range(len(M)):
        m = M[i] > 127
        g = cv2.GaussianBlur(G[i].astype(np.float32), (0, 0), 1.5)
        gy = np.diff(g, axis=0)                                # dark whale under bright water/sky -> negative step
        for j, c in enumerate(cols):
            col = m[:, max(c - 1, 0):c + 2].any(1)
            if not col.any():
                continue
            y = int(np.argmax(col))
            if y <= 2:                                          # touches the tile top: unknown
                continue
            if refine:
                lo, hi = max(y - 8, 0), min(y + 8, len(gy))
                seg = gy[lo:hi, c]
                if len(seg) and seg.min() < -4:
                    y = lo + int(np.argmin(seg)) + 1
            out[i, j] = y / PXH + Y0
    return out


def outline_scores(day, shift=0.15, step=0.05, refine=True):
    Pf = profiles(day, refine)
    N = len(Pf)
    S = np.full((N, N), np.nan, np.float32); den = np.zeros((N, N), np.float32)
    ok = ~np.isnan(Pf)
    ns = int(round(shift / step))
    for i in range(N):
        if ok[i].sum() < 10:
            continue
        best = np.full(N, np.inf); bn = np.zeros(N)
        for dx in range(-ns, ns + 1):                          # horizontal shift in columns
            a = np.roll(Pf[i], dx); va = np.roll(ok[i], dx)
            if dx > 0:
                va[:dx] = False
            elif dx < 0:
                va[dx:] = False
            v = va[None] & ok
            n = v.sum(1)
            d = np.where(v, Pf - a[None], 0)
            mu = d.sum(1) / np.maximum(n, 1)                    # best vertical shift (clipped)
            mu = np.clip(mu, -shift, shift)
            e = np.where(v, np.abs(d - mu[:, None]), 0).sum(1) / np.maximum(n, 1)
            e = np.where(n >= 10, e, np.inf)
            better = e < best
            best = np.where(better, e, best); bn = np.where(better, n, bn)
        S[i] = -best; den[i] = bn
    S = np.where(np.isfinite(S), S, np.nan)
    S = np.fmax(S, S.T)
    return S, np.maximum(den, den.T)


# ---- cluster pairs ---------------------------------------------------------------------------------------------------
def cluster_pairs(part, day, S, keep=None, agg='max'):
    """S N x N instance scores (NaN = not comparable) -> {(a, b): score} for the bench's pairs."""
    L = load(part, day)
    idx = {k: i for i, k in enumerate(L['keys'])}
    keep = np.ones(len(L['keys']), bool) if keep is None else keep
    cl = {c: [idx[k] for k in ks if k in idx and keep[idx[k]]] for c, ks in bench.clusters(day, part).items()}
    out = {}
    for a, b, _ in bench.pairs(day, part):
        ia, ib = cl.get(a, []), cl.get(b, [])
        if not ia or not ib:
            continue
        v = S[np.ix_(ia, ib)].ravel(); v = v[~np.isnan(v)]
        if len(v):
            out[(a, b)] = float(v.max() if agg == 'max' else np.sort(v)[-3:].mean())
    return out


# ---- fluke trailing edge ---------------------------------------------------------------------------------------------
XF = np.arange(-1.0, 1.001, 0.04)                            # columns, half-spans


@functools.lru_cache(maxsize=3)
def fluke_profiles(day, refine=True):
    """top edge (= trailing edge, notch side up after the frame) per column XF, in half-spans; NaN where unknown."""
    import fluke
    L = load('fluke', day)
    M, G = L['z']['mask'], L['z']['grey']
    out = np.full((len(M), len(XF)), np.nan, np.float32)
    cols = np.round((XF - fluke.FX0) * fluke.PXS).astype(int).clip(0, fluke.TW - 1)
    for i in range(len(M)):
        m = M[i] > 127
        g = cv2.GaussianBlur(G[i].astype(np.float32), (0, 0), 1.5)
        gy = np.abs(np.diff(g, axis=0))                         # fluke may be lighter or darker than the water
        for j, c in enumerate(cols):
            col = m[:, max(c - 1, 0):c + 2].any(1)
            if not col.any():
                continue
            y = int(np.argmax(col))
            if y <= 2:
                continue
            if refine:
                lo, hi = max(y - 8, 0), min(y + 8, len(gy))
                seg = gy[lo:hi, c]
                if len(seg) and seg.max() > 4:
                    y = lo + int(np.argmax(seg)) + 1
            out[i, j] = y / fluke.PXS + fluke.FY0
    return out


def profile_scores(Pf, shift_cols=3, vshift=0.15, nmin=10):
    """-mean |profile_a - profile_b| after the best small horizontal / vertical shift; symmetric (best of both)."""
    N = len(Pf)
    S = np.full((N, N), np.nan, np.float32); den = np.zeros((N, N), np.float32)
    ok = ~np.isnan(Pf)
    for i in range(N):
        if ok[i].sum() < nmin:
            continue
        best = np.full(N, np.inf); bn = np.zeros(N)
        for dx in range(-shift_cols, shift_cols + 1):
            a = np.roll(Pf[i], dx); va = np.roll(ok[i], dx)
            if dx > 0:
                va[:dx] = False
            elif dx < 0:
                va[dx:] = False
            v = va[None] & ok
            n = v.sum(1)
            d = np.where(v, Pf - a[None], 0)
            mu = np.clip(d.sum(1) / np.maximum(n, 1), -vshift, vshift)
            e = np.where(v, np.abs(d - mu[:, None]), 0).sum(1) / np.maximum(n, 1)
            e = np.where(n >= nmin, e, np.inf)
            better = e < best
            best = np.where(better, e, best); bn = np.where(better, n, bn)
        S[i] = np.where(np.isfinite(best), -best, np.nan); den[i] = bn
    return np.fmax(S, S.T), np.maximum(den, den.T)


def fluke_quality(day, second=0.3, nx=0.35, ny=(0.0, 0.5), smin=40):
    L = load('fluke', day)
    out = []
    for k in L['keys']:
        i = L['info'][k]
        out.append(i['second'] >= second and abs(i['Nc'][0]) <= nx and ny[0] <= i['Nc'][1] <= ny[1] and i['S'] >= smin)
    return np.array(out, bool)


FT = np.array([-2.7, 0.0])                                   # template front base (canonical units)


@functools.lru_cache(maxsize=3)
def profiles_affine(day, ft=tuple(FT)):
    """the top-edge profile re-expressed in the 3-point affine frame (back base, tip fixed; front base -> ft): removes
    the foreshortening along the body axis that a two-point similarity keeps."""
    L = load('dorsal', day); Pf = profiles(day)
    T = np.array([-0.25, -1.0]); out = np.full_like(Pf, np.nan)
    for i, k in enumerate(L['keys']):
        Fc = np.array(L['info'][k]['Fc'])
        src = np.stack([T, Fc], 1); dst = np.stack([T, np.array(ft)], 1)
        if abs(np.linalg.det(src)) < 1e-3:
            continue
        A = dst @ np.linalg.inv(src)
        ok = np.isfinite(Pf[i])
        if ok.sum() < 5:
            continue
        p = A @ np.stack([XS[ok], Pf[i][ok]])
        o = np.argsort(p[0]); x, y = p[0][o], p[1][o]
        inside = (XS >= x[0]) & (XS <= x[-1])
        out[i, inside] = np.interp(XS[inside], x, y)
    return out
