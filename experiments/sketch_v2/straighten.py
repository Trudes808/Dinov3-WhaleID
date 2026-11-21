"""sketch_v2 B (2026-11-19): does straightening (un-foreshortening along the body) make the outlines of the SAME animal
seen at different angles more alike, and keep different animals apart? 21 train + 3 val days only (bench refuses eval).

Drawings: DAY/sketch/v2/<cluster>__<L|R>.npz (fin_sketch v2, the picture frame: base line level, square fin height 1,
tip at (0, -1); outline polylines, head left). Pairs: bench.pairs(day, 'dorsal') (clusters of one catalogue group, more
than 60 s apart). Outline distance (fin heights): median |difference| of the top edge v(u), u in [-2.0, -0.1] (hump and
leading edge), and of the trailing edge u(v), v in [-0.85, -0.12]; at least 15 points both saw. Same side when both
clusters have it, else across sides (head-left frame). Straightening: u -> u * s per drawing, s from that drawing alone:
  prof_k   s = clip(profile, 0.35, 1) ** -k            (the photo pose model's |nose right|; k = 0.5, 1)
  base     s = clip(b_ref / bperp, 1, 2.5)             (b_ref = median base ratio of the stacks with profile >= 0.9 on
                                                         the 21 train days: "every fin is as long-based as a side-on one")
  label    s = base only where the view label says angled
  pairfit  diagnostic, not a straightening: the best relative stretch of the pair in [0.6, 1.6] (a free parameter
           fitted on the pair itself, so different animals get closer too)
Writes sketch_v2/straighten.json.      python straighten.py"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'moonshot_sizing'))
sys.path.insert(0, str(HERE.parents[1]))
import bench  # noqa: E402
from size_levers import auc  # noqa: E402
from model_fine_tuning.cluster import fin_sketch as FS  # noqa: E402

UG = np.arange(-2.0, -0.1 + 1e-9, 0.02)
VG = np.arange(-0.85, -0.12 + 1e-9, 0.02)
SUB = 'sketch/v2'


def curves(z):
    top = z['top'] / FS.DPX + np.array([FS.DU0, FS.DV0]); tr = z['trail'] / FS.DPX + np.array([FS.DU0, FS.DV0])
    return top, tr


def sample(poly, along, grid, gap=0.1):
    """poly (N, 2) [u, v]; along 0 = v(u) on a u grid, 1 = u(v) on a v grid; NaN where not seen."""
    ok = ~np.isnan(poly).any(1)
    p = poly[ok]
    out = np.full(len(grid), np.nan)
    if len(p) < 5:
        return out
    x, y = p[:, along], p[:, 1 - along]
    o = np.argsort(x); x, y = x[o], y[o]
    v = np.interp(grid, x, y, left=np.nan, right=np.nan)
    j = np.clip(np.searchsorted(x, grid), 1, len(x) - 1)
    v[(x[j] - x[j - 1]) > gap] = np.nan
    return v


def dist(A, B, sa=1.0, sb=1.0):
    ta = sample(A['top'] * [sa, 1], 0, UG); tb = sample(B['top'] * [sb, 1], 0, UG)
    ra = sample(A['trail'] * [sa, 1], 1, VG); rb = sample(B['trail'] * [sb, 1], 1, VG)
    d = np.concatenate([np.abs(ta - tb), np.abs(ra - rb)])
    d = d[~np.isnan(d)]
    return float(np.median(d)) if len(d) >= 15 else None


def levelled_frame(F):
    """the first v2 picture transform (2026-11-19 morning, since dropped): each drawing levelled on its own base line
    (turn clipped to 30 degrees), square fin height 1, tip at (0, -1)."""
    F = np.asarray(F, float); L = float(np.linalg.norm(F)); a = -F / L
    phi = float(np.clip(np.degrees(np.arctan2(a[1], a[0])), -30, 30)); r = np.radians(-phi)
    R = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]]); Tr = R @ FS.TIP_C; hp = max(float(-Tr[1]), 0.3)
    A = R / hp; t = -Tr / hp + np.array([0.0, -1.0])
    M = np.zeros((2, 3)); M[:, :2] = A / FS.PXH * FS.DPX; M[:, 2] = (A @ np.array([-4.6, -2.0]) + t - np.array([FS.DU0, FS.DV0])) * FS.DPX
    return M


def load(day):
    d = FS.day_dir(day) / SUB
    doc = json.loads((d / 'index.json').read_text()); idx = doc['clusters']; fixed = 'PHI0' in doc.get('picture', {})
    out = {}
    for c in bench.clusters(day, 'dorsal'):
        e = idx.get(str(c))
        if not e:
            continue
        out[c] = {}
        for s, x in e.items():
            z = np.load(d / x['npz'])
            top, tr = curves(z)
            # the same outline in the v1 frame (back base -> tip anchored, not levelled): undo the picture transform
            phi = np.radians(x['level_deg']); a = np.array([np.cos(phi), np.sin(phi)])
            L = x['bperp'] * abs(a[0] * FS.TIP_C[1] - a[1] * FS.TIP_C[0])
            M = FS.view_frame()[0] if fixed else levelled_frame(-L * a); Mi = np.linalg.inv(np.vstack([M, [0, 0, 1]]))[:2]
            bt = lambda q: ((Mi[:, :2] @ z_[q].T).T + Mi[:, 2]) / FS.PXH + np.array([FS.X0, FS.Y0]) + np.array([0.25, 0.0])
            z_ = {'top': z['top'], 'trail': z['trail']}
            out[c][s + '_bt'] = dict(top=bt('top'), trail=bt('trail'))
            out[c][s] = dict(top=top, trail=tr, prof=x['profile'] if x['profile'] is not None else np.nan, bperp=x['bperp'], view=x['view'],
                             n=x['n'], H=x['H'], grade=x['grade'])
    return out


def main():
    days = bench.TRAIN + bench.VAL
    C = {d: load(d) for d in days}
    bref = float(np.median([x['bperp'] for d in bench.TRAIN for v in C[d].values() for x in v.values() if 'prof' in x and x['prof'] >= 0.9]))
    S = {
        'none': lambda x: 1.0,
        'prof_0.5': lambda x: float(np.clip(x['prof'] if np.isfinite(x['prof']) else 1.0, 0.35, 1)) ** -0.5,
        'prof_1': lambda x: float(np.clip(x['prof'] if np.isfinite(x['prof']) else 1.0, 0.35, 1)) ** -1.0,
        'base': lambda x: float(np.clip(bref / max(x['bperp'], 1e-3), 1, 2.5)),
        'label': lambda x: float(np.clip(bref / max(x['bperp'], 1e-3), 1, 2.5)) if x['view'] == 'angled' else 1.0,
    }
    rows = []
    for d in days:
        for a, b, same in bench.pairs(d, 'dorsal'):
            if a not in C[d] or b not in C[d]:
                continue
            com = [s for s in ('L', 'R') if s in C[d][a] and s in C[d][b]]
            fa = [s for s in ('L', 'R') if s in C[d][a]]; fb = [s for s in ('L', 'R') if s in C[d][b]]
            sa_, sb_ = (com[0], com[0]) if com else (fa[0], fb[0])
            A, B = C[d][a][sa_], C[d][b][sb_]
            r = dict(day=d, same=bool(same), sameside=bool(com), angle=abs(np.log(A['bperp'] / B['bperp'])),
                     dprof=abs(A['prof'] - B['prof']) if np.isfinite(A['prof']) and np.isfinite(B['prof']) else np.nan,
                     well=min(A['n'], B['n']) >= 2 and min(A['H'], B['H']) >= 50)
            for k, f in S.items():
                r[k] = dist(A, B, f(A), f(B))
            Ab, Bb = C[d][a][sa_ + '_bt'], C[d][b][sb_ + '_bt']
            r['v1_frame'] = dist(Ab, Bb)
            for k in ('prof_0.5', 'prof_1', 'base'):
                r['v1_' + k] = dist(Ab, Bb, S[k](A), S[k](B))
            fit = [dist(A, B, 1.0, s) for s in np.exp(np.linspace(np.log(0.6), np.log(1.6), 11))]
            fit = [v for v in fit if v is not None]
            r['pairfit'] = min(fit) if fit else None
            rows.append(r)
    out = {'b_ref': bref, 'pairs': len(rows)}
    def summ(sel, tag):
        res = {}
        for k in list(S) + ['pairfit', 'v1_frame', 'v1_prof_0.5', 'v1_prof_1', 'v1_base']:
            ok = [r for r in sel if r[k] is not None and r['none'] is not None]
            s = np.array([r[k] for r in ok if r['same']]); df = np.array([r[k] for r in ok if not r['same']])
            if len(s) < 5 or len(df) < 5:
                continue
            res[k] = dict(same=len(s), diff=len(df), same_median=round(float(np.median(s)), 4), diff_median=round(float(np.median(df)), 4),
                          auc=round(float(auc(-s, -df)), 4))
        out[tag] = res
        print(tag, json.dumps(res))
    summ(rows, 'all')
    summ([r for r in rows if r['well']], 'well_seen')
    summ([r for r in rows if r['angle'] >= 0.4], 'different_angles_base')
    summ([r for r in rows if np.isfinite(r['dprof']) and r['dprof'] >= 0.3], 'different_angles_profile')
    summ([r for r in rows if r['angle'] < 0.15], 'similar_angles_base')
    summ([r for r in rows if r['day'] in bench.VAL], 'val_days')
    # paired change for same-animal pairs: share of pairs whose distance shrinks
    for k in ['prof_0.5', 'prof_1', 'base', 'label', 'v1_prof_0.5', 'v1_prof_1', 'v1_base']:
        base_k = 'v1_frame' if k.startswith('v1_') else 'none'
        ch = [r[k] - r[base_k] for r in rows if r['same'] and r[k] is not None and r[base_k] is not None and r['angle'] >= 0.4]
        out.setdefault('same_diffangle_change', {})[k] = dict(n=len(ch), median=round(float(np.median(ch)), 4), share_closer=round(float(np.mean(np.array(ch) < 0)), 3))
    print(json.dumps(out.get('same_diffangle_change')))
    json.dump(out, open(HERE / 'straighten.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
