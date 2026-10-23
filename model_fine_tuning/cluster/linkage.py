"""Average-linkage agglomeration on a similarity matrix with cannot-link constraints.

Same rule as cluster.py's sklearn call (merge the pair of clusters with the highest mean similarity while it is at
least `threshold`), but a merge is refused when any member of one cluster is cannot-linked with any member of the
other (two whales in one photo, cluster/tracklets.py). Without constraints it reproduces the sklearn result up to
tie-breaking. n up to a few thousand nodes: each step is one vectorised argmax over the matrix.
"""
from __future__ import annotations

import numpy as np


def constrained_average_linkage(score: np.ndarray, threshold: float, cannot: np.ndarray | None = None) -> np.ndarray:
    n = len(score)
    S = np.array(score, dtype=np.float64); S = 0.5 * (S + S.T)
    S[~np.isfinite(S)] = -np.inf
    np.fill_diagonal(S, -np.inf)
    CL = np.array(cannot, dtype=bool) if cannot is not None else np.zeros((n, n), bool)
    CL = CL | CL.T
    active = np.ones(n, bool); size = np.ones(n); labels = np.arange(n)
    while True:
        M = np.where(CL | ~active[:, None] | ~active[None, :], -np.inf, S)
        flat = int(np.argmax(M)); i, j = divmod(flat, n)
        if not np.isfinite(M[i, j]) or M[i, j] < threshold:
            break
        row = (S[i] * size[i] + S[j] * size[j]) / (size[i] + size[j])       # average linkage over member pairs
        S[i, :] = row; S[:, i] = row; S[i, i] = -np.inf
        CL[i, :] |= CL[j, :]; CL[:, i] = CL[i, :]; CL[i, i] = False
        active[j] = False; size[i] += size[j]
        labels[labels == j] = i
        S[j, :] = -np.inf; S[:, j] = -np.inf
    _, out = np.unique(labels, return_inverse=True)
    return out.astype(np.int64)
