"""Aligned grid vector per instance (label-free), for bench.evaluate: the canonical tile's tokens pooled to cells,
per-cell whale-weighted, reduced by one PCA fitted on whale tokens of all train + val days (no labels), day-centred per
cell by the bench's centring of the whole vector; cells without whale = 0.  The bench's head (whitened PCA 128 +
within-animal whitening + LDA, fitted on train days only) then learns which cells and directions matter.

    python grid.py PART LAYER POOL NPCA [--bench]
"""
import sys

import cv2
import numpy as np

import score
from common import bench


def pooled(part, day, layer, pool):
    T, W, (gh, gw) = score.tokens(part, day, layer, center=False)
    N, P, D = T.shape
    T = T.reshape(N, gh, gw, D) * W.reshape(N, gh, gw, 1)
    W = W.reshape(N, gh, gw)
    hh, ww = gh // pool, gw // pool
    T = T[:, :hh * pool, :ww * pool].reshape(N, hh, pool, ww, pool, D).sum((2, 4))
    W = W[:, :hh * pool, :ww * pool].reshape(N, hh, pool, ww, pool).sum((2, 4))
    T = T / np.maximum(W[..., None], 1e-6)
    return T, W / (pool * pool)


def build(part, layer='tok11', pool=2, npca=24, wmin=0.3, seed=0):
    days = bench.TRAIN + bench.VAL
    rng = np.random.default_rng(seed)
    samp = []
    cache = {}
    for d in days:
        T, W = pooled(part, d, layer, pool)
        cache[d] = (T, W)
        v = T[W > wmin]
        samp.append(v[rng.choice(len(v), min(len(v), 3000), replace=False)])
    S = np.concatenate(samp)
    mu = S.mean(0)
    _, _, Vt = np.linalg.svd(S - mu, full_matrices=False)
    Pm = Vt[:npca].T
    X = {}
    for d in days:
        T, W = cache[d]
        Z = (T - mu) @ Pm                                    # N h w npca
        Z = np.where((W > wmin)[..., None], Z, 0.0)
        keys = score.load(part, d)['keys']
        X[d] = {k: Z[i].ravel().astype(np.float32) for i, k in enumerate(keys)}
    return X


if __name__ == '__main__':
    part, layer, pool, npca = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    X = build(part, layer, pool, npca)
    if '--bench' in sys.argv:
        bench.evaluate(f'aligned_grid_{layer}_p{pool}_{npca}', part, X)


def outline_vectors(qual=None):
    """fin + hump top-edge profile per instance (fin heights), unknown columns filled with the day's column median."""
    X = {}
    for d in bench.TRAIN + bench.VAL:
        Pf = score.profiles(d).copy()
        med = np.nanmedian(Pf, 0); med = np.where(np.isfinite(med), med, 0)
        Pf = np.where(np.isfinite(Pf), Pf, med[None])
        keys = score.load('dorsal', d)['keys']
        keep = score.quality('dorsal', d, qual) if qual else np.ones(len(keys), bool)
        X[d] = {k: Pf[i].astype(np.float32) for i, k in enumerate(keys) if keep[i]}
    return X
