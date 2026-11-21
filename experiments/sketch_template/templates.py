"""sketch_template A (2026-11-19): a standard outline (back and hump ahead of the fin, leading edge, trailing edge, the
back behind it; the side profile, head left, in the fin_sketch picture frame) for each dorsal shape the trip log notes
("height form", e.g. "moderate hook"), from the SIDE-ON drawings (view label "side") of the animals noted with that
shape on the 21 train + 3 val days. bench refuses the eval days.

  per animal   the median of its side-on drawings' curves (left and right sides pooled: both are drawn head left)
  per shape    the median over its animals, where at least MIN_AT animals saw the place; DIRECT when at least N_DIRECT
               animals have a side-on drawing, else FROM PARTS: the height's curve + the form's curve - the curve of all
               (each a median over all animals of that height / that form)
  all          the median of all animals ("generic", the outline when no shape is noted; used only in the measures)

How well do the noted shapes show in the outlines? Every animal's side-on outline against the templates built WITHOUT
its day (leave one day out): rank of its own noted shape among all shapes and among the shapes noted in its group, and
the same for the height alone and the form alone.

    python templates.py      -> CACHE/templates.npz (+ templates.json), CACHE/animal_curves.npz (per-animal: cache only),
                                sketch_template/templates_fit.json (aggregate numbers only)
"""
import collections
import json
import re

import numpy as np

import common as C
import outline as O

N_DIRECT = 8
MIN_AT = 3
MIN_SHARE = 0.4
PARTS = ('top', 'trail', 'back')


def collect():
    """rows: one per drawing of a catalogue animal with a noted shape."""
    rows = []
    for d in C.DAYS:
        N = C.noted(d)
        for (c, s), v in C.drawings(d).items():
            a = v['meta']['animal']
            sh = N.get(a)
            if not sh:
                continue
            m = v['meta']
            rows.append(dict(day=d, cl=c, side=s, animal=a, shape=sh, height=sh.split()[0], form=sh.split()[1], label=m.get('label', ''),
                             n=m['n'], H=m['H'], bperp=m['bperp'], grade=m['grade'], cur=O.curves(v)))
    return rows


def animal_curves(rows):
    """{animal: dict(shape, day, curves)} from its side-on drawings."""
    by = collections.defaultdict(list)
    for r in rows:
        if r['label'] == 'side' and C.FS.pad_edges(r['cur'])['ok']:      # a fin outline (keypoints right, edges seen)
            by[r['animal']].append(r)
    out = {}
    for a, rs in by.items():
        out[a] = dict(shape=rs[0]['shape'], height=rs[0]['height'], form=rs[0]['form'], day=rs[0]['day'], k=len(rs),
                      group=re.search(r'_(G\d+)A', a).group(1) if re.search(r'_(G\d+)A', a) else None,
                      cur={p: O.nanmed([r['cur'][p] for r in rs]) for p in PARTS})
    return out


def med(curs, min_n=MIN_AT):
    """the median curve where at least min_n animals and at least MIN_SHARE of them saw the place."""
    m = max(min_n, int(np.ceil(MIN_SHARE * len(curs))))
    return {p: O.nanmed([c[p] for c in curs], m) for p in PARTS}


def build(AC, exclude_day=None):
    """{shape: dict(cur, how, n)} for every noted shape (plus heights, forms and 'all')."""
    A = {a: x for a, x in AC.items() if x['day'] != exclude_day}
    allc = med([x['cur'] for x in A.values()])
    byh = {h: med([x['cur'] for x in A.values() if x['height'] == h]) for h in C.HEIGHTS}
    byf = {f: med([x['cur'] for x in A.values() if x['form'] == f]) for f in C.FORMS}
    T = {}
    for h in C.HEIGHTS:
        for f in C.FORMS:
            sh = f'{h} {f}'
            mem = [x['cur'] for x in A.values() if x['shape'] == sh]
            if len(mem) >= N_DIRECT:
                T[sh] = dict(cur=med(mem), how='direct', n=len(mem))
            else:
                T[sh] = dict(cur={p: byh[h][p] + byf[f][p] - allc[p] for p in PARTS}, how='parts', n=len(mem))
    T['all'] = dict(cur=allc, how='all', n=len(A))
    return T


def finish(cur):
    """a template's curves made drawable: smoothed, small gaps filled (inside the range seen)."""
    out = {}
    for p, g, s in (('top', O.UG, 2.0), ('trail', O.VG, 2.0), ('back', O.BG, 2.0)):
        x = cur[p].copy(); ok = ~np.isnan(x)
        if ok.sum() >= 2:
            i0, i1 = np.nonzero(ok)[0][[0, -1]]
            x[i0:i1 + 1] = O.fill_nan(x, g)[i0:i1 + 1]
        out[p] = O.smooth(x, s)
    return out


def fit_test(AC):
    shapes = [f'{h} {f}' for h in C.HEIGHTS for f in C.FORMS]
    res = collections.defaultdict(list)
    per_day = {d: build(AC, exclude_day=d) for d in sorted({x['day'] for x in AC.values()})}
    group_shapes = collections.defaultdict(set)
    for d in C.DAYS:
        for a, sh in C.noted(d).items():
            m = re.search(r'_(G\d+)A', a)
            if m:
                group_shapes[(d, m.group(1))].add(sh)
    for a, x in AC.items():
        T = per_day[x['day']]
        ds = {s: O.dist(x['cur'], T[s]['cur']) for s in shapes}
        if ds[x['shape']] is None:
            continue
        ds = {s: v for s, v in ds.items() if v is not None}
        order = sorted(ds, key=ds.get)
        res['rank_all'].append(order.index(x['shape']) + 1); res['n_all'].append(len(order))
        # heights / forms alone: the best template of each height / form
        for part, vals in (('height', C.HEIGHTS), ('form', C.FORMS)):
            best = {v: min([ds[s] for s in ds if s.split()[0 if part == 'height' else 1] == v] or [np.inf]) for v in vals}
            o = sorted(best, key=best.get)
            res[f'top1_{part}'].append(o[0] == x[part])
        gs = sorted(s for s in group_shapes[(x['day'], x['group'])] if s in ds)
        if len(set(gs)) >= 2:
            res['top1_group'].append(min(gs, key=ds.get) == x['shape']); res['chance_group'].append(1 / len(gs))
        own = ds[x['shape']]
        res['own'].append(own); res['other_med'].append(float(np.median([v for s, v in ds.items() if s != x['shape']])))
        dall = O.dist(x['cur'], T['all']['cur'])
        if dall is not None:
            res['own_minus_generic'].append(own - dall)
    r = res
    out = dict(animals=len(r['rank_all']),
               own_rank1_all=round(float(np.mean(np.array(r['rank_all']) == 1)), 3), chance_all=round(float(np.mean(1 / np.array(r['n_all']))), 3),
               own_rank_median=float(np.median(r['rank_all'])), n_shapes=int(np.median(r['n_all'])),
               own_in_top3=round(float(np.mean(np.array(r['rank_all']) <= 3)), 3),
               top1_height=round(float(np.mean(r['top1_height'])), 3), chance_height=round(1 / 3, 3),
               top1_form=round(float(np.mean(r['top1_form'])), 3), chance_form=round(1 / 6, 3),
               group_clusters=len(r['top1_group']), top1_group=round(float(np.mean(r['top1_group'])), 3), chance_group=round(float(np.mean(r['chance_group'])), 3),
               own_dist_median=round(float(np.median(r['own'])), 4), other_dist_median=round(float(np.median(r['other_med'])), 4),
               own_closer_than_generic=round(float(np.mean(np.array(r['own_minus_generic']) < 0)), 3))
    # a fairer chance for "height": the share of the most common height
    hs = collections.Counter(x['height'] for x in AC.values()); fs = collections.Counter(x['form'] for x in AC.values())
    out['majority_height'] = round(hs.most_common(1)[0][1] / sum(hs.values()), 3); out['majority_form'] = round(fs.most_common(1)[0][1] / sum(fs.values()), 3)
    return out


def distinct(T):
    shapes = [s for s in T if s != 'all']
    M = {a: {b: O.dist(T[a]['cur'], T[b]['cur']) for b in shapes} for a in shapes}
    return M


def main():
    rows = collect()
    AC = animal_curves(rows)
    T = build(AC)
    fin = {s: finish(t['cur']) for s, t in T.items()}
    # within-shape spread: an animal's distance to its own (all-days) template, against the templates' distances
    spread = collections.defaultdict(list)
    for a, x in AC.items():
        dd = O.dist(x['cur'], T[x['shape']]['cur'])
        if dd is not None:
            spread[x['shape']].append(dd)
    M = distinct(T)
    shapes = [s for s in T if s != 'all']
    pair_d = [M[a][b] for i, a in enumerate(shapes) for b in shapes[i + 1:] if M[a][b] is not None]
    near = {a: min(((M[a][b], b) for b in shapes if b != a and M[a][b] is not None), default=(None, None)) for a in shapes}
    fit = fit_test(AC)
    summary = dict(made=C.FS._now(), drawings=len(rows), side_on_drawings=sum(r['label'] == 'side' for r in rows), animals_side_on=len(AC),
                   n_direct=N_DIRECT, min_at=MIN_AT,
                   templates={s: dict(how=T[s]['how'], animals=T[s]['n'], spread_median=round(float(np.median(spread[s])), 4) if spread[s] else None,
                                      nearest=near[s][1] if s in near else None, nearest_dist=round(near[s][0], 4) if s in near and near[s][0] is not None else None)
                              for s in T if s != 'all'},
                   template_pair_dist=dict(median=round(float(np.median(pair_d)), 4), min=round(float(np.min(pair_d)), 4), max=round(float(np.max(pair_d)), 4)),
                   animal_to_own_template_median=round(float(np.median([v for x in spread.values() for v in x])), 4),
                   fit=fit)
    C.CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(C.CACHE / 'templates.npz', shapes=np.array(list(fin)), UG=O.UG, VG=O.VG, BG=O.BG,
                        **{f'{s}|{p}': fin[s][p] for s in fin for p in PARTS}, **{f'raw|{s}|{p}': T[s]['cur'][p] for s in T for p in PARTS})
    (C.CACHE / 'templates.json').write_text(json.dumps({s: dict(how=T[s]['how'], animals=T[s]['n']) for s in T}, indent=1))
    np.savez_compressed(C.CACHE / 'animal_curves.npz', **{f'{a}|{p}': x['cur'][p] for a, x in AC.items() for p in PARTS},
                        meta=json.dumps({a: dict(shape=x['shape'], day=x['day'], k=x['k']) for a, x in AC.items()}))
    (C.HERE / 'templates_fit.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
