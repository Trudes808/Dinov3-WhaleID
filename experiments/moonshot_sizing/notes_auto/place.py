"""Layer 1: each safe cluster to a noted group (or none = passer-by) from its time against the noted windows.

Rule: the windows that contain the cluster's median photo time (the smallest one if several); else the nearest window
if it is at most M minutes away, else none. M is chosen on the train days other than the scored day, maximising labelled clusters placed right (passers-by right when placed nowhere).
Options: frame cues from the sighting comments ('starts / ends on frame N of C<k>' moves that camera's window edge to
the time of that frame); aliases ('same as / continuation of group N' when the sighting adds no whale rows: its photos
go to group N).

    python place.py   -> aggregate numbers (repo log); per-day rows only to the cache
"""
import collections
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data  # noqa: E402

MS = (0, 2, 5, 10, 15, 20, 30, 45, 60, 90, 120, 1e9)


def windows(rec, cues=True, alias=True):
    """{camera or None: {group: (start, end)}} in seconds of the day; target group per sighting after aliasing."""
    G, W = rec['log']['groups'], rec['log']['whales']
    base = {}
    for k, g in G.items():
        if not np.isfinite(g['start']):
            continue
        e = g['end'] if np.isfinite(g['end']) and g['end'] > g['start'] else g['start'] + 3600
        base[k] = (g['start'], e)
    target = {k: k for k in G}
    if alias:
        for k, g in G.items():
            j = g['alias']
            if j in G and j != k and len(W.get(k, {})) <= len(W.get(j, {})):
                target[k] = j
    per_cam = {None: dict(base)}
    if cues:
        ftime = collections.defaultdict(dict)                     # camera -> frame -> time of day
        for c in rec['clusters'].values():
            for (cam, fr), p in zip(c['frames'], c['photos']):
                if cam and fr is not None:
                    ftime[cam][fr] = c['tmid']
        for k, g in G.items():
            for kind, cam, fr in g['cues']:
                if cam not in ftime or k not in base:
                    continue
                frs = np.array(sorted(ftime[cam]))
                cand = frs[frs >= fr] if kind == 'start' else frs[frs <= fr]       # first frame of the group / its last frame
                if not len(cand):
                    continue
                near = cand.min() if kind == 'start' else cand.max()
                if abs(near - fr) > 30:
                    continue
                t = ftime[cam][near]
                w = per_cam.setdefault(cam, dict(base))
                s, e = w[k]
                before = [j for j in w if j != k and w[j][0] < s]
                after = [j for j in w if j != k and w[j][0] > s]
                if kind == 'start':
                    w[k] = (t, max(e, t + 1))
                    if before:                                       # the camera's earlier frames stay with the group before
                        j = max(before, key=lambda j: w[j][0])
                        w[j] = (w[j][0], max(w[j][1], t - 1))
                else:
                    w[k] = (min(s, t - 1), t)
                    if after:
                        j = min(after, key=lambda j: w[j][0])
                        w[j] = (max(w[j][0], t + 1), max(w[j][1], t + 2))
    return per_cam, target


def place(rec, M, cues=True, alias=True):
    """{cluster: dict(g=group or None, inside=bool, n_in=int, dist=minutes outside)}"""
    per_cam, target = windows(rec, cues, alias)
    out = {}
    for k, c in rec['clusters'].items():
        w = per_cam.get(c['cam'], per_cam[None])
        t = c['tmid']
        inside = [g for g, (s, e) in w.items() if s <= t <= e]
        if inside:
            g = min(inside, key=lambda g: w[g][1] - w[g][0])
            out[k] = dict(g=target[g], inside=True, n_in=len(inside), dist=0.0)
            continue
        if not w:
            out[k] = dict(g=None, inside=False, n_in=0, dist=np.inf)
            continue
        dist = {g: min(abs(t - s), abs(t - e)) / 60 for g, (s, e) in w.items()}
        g = min(dist, key=dist.get)
        out[k] = dict(g=target[g] if dist[g] <= M else None, inside=False, n_in=0, dist=dist[g])
    return out


def right_group(c, g):
    """Is placement g right for cluster c? Passer-by right when g is None."""
    t = c['truth']
    return (g is None) if t[0] == 'O' else (g == t[1])


def tally(rec, P):
    n = collections.Counter()
    for k, c in rec['clusters'].items():
        if c['truth'] is None:
            continue
        ok = right_group(c, P[k]['g'])
        kind = 'passer' if c['truth'][0] == 'O' else 'group'
        n[f'{kind} clusters'] += 1
        n[f'{kind} photos'] += c['size']
        n[f'{kind} clusters right'] += ok
        n[f'{kind} photos right'] += ok * c['size']
        if kind == 'passer':
            n['passer clusters absorbed'] += not ok
        else:
            n['group clusters to none'] += P[k]['g'] is None
            n['group clusters to another group'] += P[k]['g'] is not None and not ok
    return n


def choose_M(recs, held, cues=True, alias=True):
    best, bm = -1, None
    for M in MS:
        s = 0
        for r in recs:
            if r['day'] == held or r['day'] not in data.bench.TRAIN:     # fitted on train days only, scored day held out
                continue
            t = tally(r, place(r, M, cues, alias))
            s += t['group clusters right'] + t['passer clusters right']
        if s > best:
            best, bm = s, M
    return bm


def run(recs, cues=True, alias=True):
    """Placement for every day with M chosen on the other days. -> {day: (M, placement)}"""
    return {r['day']: (M := choose_M(recs, r['day'], cues, alias), place(r, M, cues, alias)) for r in recs}


def summary(recs, res, days):
    t = collections.Counter()
    for r in recs:
        if r['day'] in days:
            t.update(tally(r, res[r['day']][1]))
    gc, pc = t['group clusters'], t['passer clusters']
    return (f"group clusters right {t['group clusters right']}/{gc} ({100 * t['group clusters right'] / max(1, gc):.1f} %), photos "
            f"{t['group photos right']}/{t['group photos']} ({100 * t['group photos right'] / max(1, t['group photos']):.1f} %); to none {t['group clusters to none']}, "
            f"to another group {t['group clusters to another group']} | passers-by caught {t['passer clusters right']}/{pc} clusters "
            f"({t['passer photos right']}/{t['passer photos']} photos), absorbed {t['passer clusters absorbed']}")


if __name__ == '__main__':
    recs = [data.load(d) for d in data.days()]
    priv = open(data.CACHE / 'place_days.log', 'w')
    for name, kw in (('time windows only', dict(cues=False, alias=False)), ('+ aliases', dict(cues=False, alias=True)),
                     ('+ aliases + frame cues', dict(cues=True, alias=True))):
        res = run(recs, **kw)
        Ms = collections.Counter(res[d][0] for d in res)
        print(f'== {name}; M chosen with the day held out: {dict(Ms)}')
        print(f'   all {len(recs)} days: ' + summary(recs, res, data.days()))
        print(f'   val days:     ' + summary(recs, res, data.bench.VAL))
        for r in recs:
            print(name, r['day'], summary(recs, res, [r['day']]), file=priv)
