"""Cluster-pair scores from the instance-pair stats of lm_match.py, and their bench numbers.

    python lm_score.py VARIANT PART screen          val days, bench.compare against BASE fitted on all 21 train days
    python lm_score.py VARIANT PART full SCORE      bench.evaluate_pairs (20 repetitions) for one score definition
"""
import os
import sys
import zlib
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lm_lib as LM  # noqa: E402
from lm_lib import bench  # noqa: E402


def load(variant, day, part):
    fp = LM.CACHE / variant / f'{day}__{part}__pairs.npz'
    if not fp.exists():
        return None
    z = np.load(fp)
    names = [str(n) for n in z['names']]
    F = torch.load(LM.CACHE / variant / f'{day}__{part}.pt', weights_only=False)
    keys = [str(k) for k in z['keys']]
    nkp = np.array([len(F['feats'][k]['plain']['kp']) for k in keys], float)
    return z['jobs'], z['stats'], names, nkp


def inst_scores(J, S, names, nkp):
    """{score name: per-row value}."""
    c = {n: S[:, i] for i, n in enumerate(names)}
    mn = np.minimum(nkp[J[:, 2]], nkp[J[:, 3]])
    out = {
        'matches': c['n@t12'],
        'sim6': c['sim@t6'], 'sim12': c['sim@t12'], 'aff12': c['aff@t12'], 'hom12': c['hom@t12'],
        'simconf12': c['simc@t12'], 'simconf6': c['simc@t6'],
        'ratio12': c['sim@t12'] / np.maximum(mn, 1),
        'spread12': c['spread@t12'],
        'sim12xspread': np.log1p(c['sim@t12']) * np.sqrt(c['spread@t12']),
    }
    return out


def load_geo(variant, day, part):
    fp = LM.CACHE / variant / f'{day}__{part}__geo.npz'
    if not fp.exists():
        return None
    z = np.load(fp)
    return z['jobs'], z['stats'], [str(n) for n in z['names']], None


def cluster_scores(variant, part, days, mirror='max', geo=False, no_fallback=False, npick=2):
    """{score: {day: {(a, b): value}}}; mirror = plain | max | mirror (dorsal only); geo = the lm_geo.py stats."""
    R = defaultdict(dict)
    for d in days:
        L = (load_geo if geo else load)(variant, d, part)
        if L is None:
            continue
        J, S, names, nkp = L
        if no_fallback and part == 'fluke':
            F = torch.load(LM.CACHE / variant / f'{d}__{part}.pt', weights_only=False)['feats']
            keys = [str(k) for k in np.load(LM.CACHE / variant / f'{d}__{part}__pairs.npz')['keys']]
            fb = np.array([bool(F[k]['meta'].get('fallback')) for k in keys])
            keep = ~(fb[J[:, 2]] | fb[J[:, 3]])
            J, S = J[keep], S[keep]
        if npick:
            Fp = torch.load(LM.CACHE / variant / f'{d}__{part}.pt', weights_only=False)['picked']
            rank = {k: i for v in Fp.values() for i, k in enumerate(v)}
            keys = [str(k) for k in np.load(LM.CACHE / variant / f'{d}__{part}__pairs.npz')['keys']]
            rk = np.array([rank.get(k, 99) for k in keys])
            keep = (rk[J[:, 2]] < npick) & (rk[J[:, 3]] < npick)
            J, S = J[keep], S[keep]
        sel = np.ones(len(J), bool)
        if part == 'dorsal' and mirror != 'max':
            sel = J[:, 4] == (1 if mirror == 'mirror' else 0)
        sc = {n: S[sel][:, i] for i, n in enumerate(names)} if geo else inst_scores(J[sel], S[sel], names, nkp)
        Js = J[sel]
        for name, v in sc.items():
            best = {}
            for (a, b), x in zip(map(tuple, Js[:, :2].tolist()), v.tolist()):
                if x > best.get((a, b), -1e9):
                    best[(a, b)] = x
            # random (deterministic) tie-break: the bench's AP and found-at-precision order tied scores positives first
            R[name][d] = {p: v + 1e-4 * (zlib.crc32(f'{d}|{p}'.encode()) % 10007) / 10007 for p, v in best.items()}
    return R


def screen(variant, part, geo=False, no_fallback=False):
    days = [d for d in bench.VAL if (LM.CACHE / variant / f'{d}__{part}__pairs.npz').exists()]
    for mirror in (('plain', 'max') if part == 'dorsal' else ('plain',)):
        R = cluster_scores(variant, part, days, mirror, geo, no_fallback)
        for name, S in R.items():
            r = bench.compare(f'local_match_{variant}_{name}_{mirror}', part, S, days, bench.TRAIN, quiet=True, save=False)
            c, b = r['cand'], r['base']
            fl = r['flank']['cand']
            fls = '  '.join(f"{k[:8]} {v['auc']:.3f}" for k, v in sorted(fl.items()) if k in ('same flank', 'opposite flanks'))
            print(f"{variant:11s} {part:6s} {mirror:6s} {name:13s} AUC {c['auc']:.3f} AP {c['ap']:.3f} P95 {100*c['r95']:5.1f} P90 {100*c['r90']:5.1f}"
                  f" | BASE {b['auc']:.3f} {b['ap']:.3f} {100*b['r95']:5.1f} {100*b['r90']:5.1f} | {r['pairs_scored']}/{r['pairs_total']} {fls}", flush=True)


if __name__ == '__main__':
    variant, part, mode = sys.argv[1:4]
    if mode == 'screen':
        screen(variant, part, 'geo' in sys.argv, 'nofb' in sys.argv)
    else:
        score, mirror = sys.argv[4], (sys.argv[5] if len(sys.argv) > 5 else 'max')
        R = cluster_scores(variant, part, bench.TRAIN + bench.VAL, mirror, 'geo' in sys.argv, 'nofb' in sys.argv)
        bench.evaluate_pairs(f'local_match_{variant}_{score}_{mirror}' if part == 'dorsal' else f'local_match_{variant}_{score}' + ('_nofb' if 'nofb' in sys.argv else ''), part, R[score])
