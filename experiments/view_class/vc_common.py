"""View class of a whale instance (fluke / dorsal fin / other): shared loading for the experiment scripts.

An instance is one mask component of a photo (cluster/parts.py), keyed <stem>~k. Labels by eye in view_labels.json:
{instance key: {"day": day, "code": code, "round": name}} with code built from
  F / f   the tail fluke is out of the water (clear / marginal: tiny, mostly under water, only an edge or a tip)
  D / d   the dorsal fin is visible above the water (clear / marginal: tiny, in spray, cut by the frame, a bump at this size)
  O       a whale, but neither (head, blow, back without the fin, pectoral fin, tail stock, under water, breach)
  X       not a whale or nothing recognisable (splash only, glare, boat, people)
so "DF" = both clear, "Df" = clear fin and a marginal fluke. The marginal cases are kept apart from the clear ones in
training (no loss) and in the reports.
"""
import json, os
import numpy as np

R = '/home/sat3737/NOAA_whales/Dinov3-WhaleID'; D = f'{R}/experiments/days_all'; HERE = f'{R}/experiments/view_class'
SPLIT = json.load(open(f'{R}/model_fine_tuning/splits/days.json'))
EVAL, VAL = SPLIT['eval'], SPLIT['val']
DAYS = sorted(d for d in os.listdir(D) if os.path.exists(f'{D}/{d}/safe/clusters.json'))
TRAIN = [d for d in DAYS if d not in EVAL and d not in VAL]
LABELS = f'{HERE}/view_labels.json'


def split_of(day):
    return 'eval' if day in EVAL else 'val' if day in VAL else 'train'


def unit(a):
    a = a.astype(np.float32); return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-6)


def load_day(day, feats=True):
    """Instances of a day that are in a safe cluster (the non-whale filter already took the rest out)."""
    P = json.load(open(f'{D}/{day}/safe/photos.json')); C = json.load(open(f'{D}/{day}/safe/clusters.json'))['clusters']
    cl = {k: c['id'] for c in C for k in c['keys']}; meta = {}
    for stem, p in P.items():
        for j, it in enumerate(p['instances']):
            if it['key'] not in cl:
                continue
            x0, y0, x1, y1 = it['bbox']
            meta[it['key']] = dict(day=day, stem=stem, focal=p.get('focal_mm') or np.nan, w=x1 - x0, h=y1 - y0, rank=j, n=len(p['instances']), bbox=it['bbox'], cl=cl[it['key']],
                                   parts=it.get('parts', []), t=p.get('time'), cam=p.get('camera'))
    out = dict(day=day, keys=sorted(meta), meta=meta)
    if feats:
        z = np.load(f'{D}/{day}/safe/inst_feat.npz'); kid = {str(k): i for i, k in enumerate(z['keys'])}; idx = [kid[k] for k in out['keys']]
        out['X'] = np.hstack([unit(z['cls'][idx]), unit(z['mean'][idx])])
    return out


def load_labels():
    return json.load(open(LABELS)) if os.path.exists(LABELS) else {}


def save_labels(L):
    tmp = LABELS + '.tmp'; json.dump(L, open(tmp, 'w'), indent=0, sort_keys=True); os.replace(tmp, LABELS)


def targets(code):
    """code -> (fluke, dorsal) each 1 clear / 0 absent / -1 marginal (ignored in training); None for X."""
    f = 1 if 'F' in code else -1 if 'f' in code else 0
    d = 1 if 'D' in code else -1 if 'd' in code else 0
    return f, d


def all_targets(report=False):
    """Every labelled instance: key -> {'day', 'y': [fluke, dorsal] (1 / 0 / -1 marginal, no loss), 'src': {...}}.
    Sources, strongest first: the researcher's clicks and the catalogue's fluke collection say a part IS there (1);
    the by-eye code (view_labels.json) gives both parts incl. marginal and goes first where it exists; where the
    researcher clicked one part and not the other, the other counts as absent; the "no keypoints" tick = no dorsal fin
    and a marginal fluke (see below); a whale with only other points clicked says nothing."""
    L = load_labels(); O = json.load(open(f'{HERE}/other_labels.json')); T = {}; conflicts = []
    for k, v in L.items():
        if v['code'] == 'X':
            continue
        T[k] = {'day': v['day'], 'y': list(targets(v['code'])), 'src': {'eye': v['code'], 'round': v['round']}}
    for k, v in O['researcher'].items():
        e = T.setdefault(k, {'day': v['day'], 'y': [None, None], 'src': {}}); e['src']['researcher'] = [v['fluke'], v['dorsal'], v['none'], v['seen']]
        clicked = v['fluke'] or v['dorsal']
        for j, part in enumerate(('fluke', 'dorsal')):
            if v[part]:
                if e['y'][j] == 0: conflicts.append((k, part, 'eye 0, researcher clicked'))
                e['y'][j] = 1
            elif e['y'][j] is None:
                # the other part was clicked and this one was not: absent. Only other points (blowhole, spine end) clicked:
                # says nothing. The "no keypoints" tick: no dorsal fin; for the fluke it is often a blade or an edge in
                # spray that has no clickable tip or notch -> marginal (no loss).
                e['y'][j] = 0 if clicked else (-1 if j == 0 else 0) if v['none'] else None
            elif e['y'][j] == 1 and clicked and 'eye' in e['src']:
                conflicts.append((k, part, 'eye 1, researcher clicked only the other part'))
    for stem, v in O['catalog_fluke'].items():
        if len(v['instances']) != 1:
            continue
        k = v['instances'][0]; e = T.setdefault(k, {'day': v['day'], 'y': [None, None], 'src': {}}); e['src']['catalog'] = 1
        if e['y'][0] == 0: conflicts.append((k, 'fluke', 'eye / researcher 0, in the catalogue fluke collection'))
        e['y'][0] = 1
    for e in T.values():
        e['y'] = [-9 if v is None else v for v in e['y']]           # -9 = this source says nothing about the part
    if report:
        return T, conflicts
    return T
