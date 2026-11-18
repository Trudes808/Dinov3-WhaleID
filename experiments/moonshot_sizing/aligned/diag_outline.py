"""Instance-level check of outline profile variants (val + two train days): same cluster / same animal other cluster
against other animals of the group.  Label use = scoring only."""
import numpy as np

import common  # noqa: F401
import bench
import score
from size_levers import auc

DAYS = bench.VAL + ['2025-01-02', '2025-03-18']


def labels(d):
    L = score.load('dorsal', d); k2c = bench._key2c(d); Dd = bench.D(d)
    cl = np.array([k2c[k] if k2c.get(k) is not None else -1 for k in L['keys']])
    an = np.array([Dd.maj.get(c) if c >= 0 else None for c in cl], object)
    return cl, an, score.quality('dorsal', d)


def run(tag, fn):
    pc, ps, ng = [], [], []
    for d in DAYS:
        S = fn(d)[0]; cl, an, q = labels(d)
        iu = np.triu_indices(len(cl), 1)
        for i, j in zip(*iu):
            if not (q[i] and q[j]) or np.isnan(S[i, j]) or an[i] is None or an[j] is None:
                continue
            if cl[i] == cl[j]:
                pc.append(S[i, j])
            elif an[i] == an[j]:
                ps.append(S[i, j])
            elif bench.group_of(an[i]) == bench.group_of(an[j]):
                ng.append(S[i, j])
    print(f'{tag:<40s} same cluster {auc(np.array(pc), np.array(ng)):.3f}   same animal other cluster {auc(np.array(ps), np.array(ng)):.3f}  ({len(ps)} / {len(ng)})', flush=True)


if __name__ == '__main__':
    X = score.XS
    for lo, hi in ():
        sel = (X >= lo - 1e-6) & (X <= hi + 1e-6)
        for sh in (3, 1):
            run(f'cols {lo}..{hi} shift {sh}', lambda d: score.profile_scores(score.profiles(d)[:, sel], shift_cols=sh, vshift=0.15, nmin=min(10, sel.sum() // 2)))
    run('affine frame, all cols, shift 3', lambda d: score.profile_scores(score.profiles_affine(d), shift_cols=3))
    run('affine frame, all cols, shift 1', lambda d: score.profile_scores(score.profiles_affine(d), shift_cols=1))
