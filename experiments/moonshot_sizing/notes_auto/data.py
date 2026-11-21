"""Trip-log assignment experiment (2026-11-19): per-day inputs, built once and cached.

Reads the private trip-log export through the reader in moonshot_cache/notes (tables.py). Anything that quotes the log
(per-day groups, whales, attributes) is written ONLY to moonshot_cache/notes_auto/; this file holds code, no log content.
Days: the 21 train + 3 val days (bench.py), minus days without a log. The four eval days are never read.

    python data.py          -> moonshot_cache/notes_auto/day/<day>.pkl
"""
import collections
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, '/home/sat3737/NOAA_whales/moonshot_cache/notes')
import bench  # noqa: E402
from size_levers import DAYS, Day  # noqa: E402

CACHE = Path('/home/sat3737/NOAA_whales/moonshot_cache/notes_auto')
HW = Path('/home/sat3737/NOAA_whales/moonshot_cache/public_reid/feat/hw_b6/box')
EVAL = {'2024-12-18', '2024-12-26', '2025-02-19', '2025-03-21'}
ALL24 = bench.TRAIN + bench.VAL
assert not EVAL & set(ALL24)

FORMS = ('triangle', 'hook', 'round', 'square', 'point', 'pyramid')
HEIGHTS = ('low', 'moderate', 'high')


def _ok(day):
    if day in EVAL:
        raise ValueError('eval day')
    return day


def offset(day):
    """photos.json time = local wall time read as US Central; the catalogue's capture times confirm 6 h / 5 h (DST from 2025-03-09)."""
    return 21600 if day < '2025-03-09' else 18000


def tod(day, t):
    return (t - offset(day)) % 86400


def cam_frame(photo):
    """('C1', 6488) from a photo id like '<date>_C1_<name>__<file ending in 4 digits>'."""
    m = re.match(r'[^_]+_(C\d+)_', photo)
    f = re.search(r'(\d{4})$', photo)
    return (m.group(1) if m else None), (int(f.group(1)) if f else None)


# ---- the log ------------------------------------------------------------------------------------------------------------
def _norm_shape(v):
    if not isinstance(v, str):
        return None, None
    v = v.strip().lower().replace('miderate', 'moderate').replace('squared', 'square')
    h = next((x for x in HEIGHTS if x in v.split()), None)
    f = next((x for x in FORMS if x in v.split()), None)
    return h, f


def _role(nick, ind, age):
    nk = nick.strip().lower() if isinstance(nick, str) else ''
    ro = ind.strip().lower() if isinstance(ind, str) else ''
    ag = age.strip().lower() if isinstance(age, str) else ''
    if 'calf' in nk or ro == 'calf' or ag == 'calf':
        return 'calf'
    if 'mom' in nk or 'mother' in nk or ro == 'mother':
        return 'mother'
    if 'escort' in nk or 'escort' in ro or nk in ('primary', 'secondary'):
        return 'escort'
    if 'challenger' in nk or 'challenger' in ro:
        return 'challenger'
    if 'nuclear' in nk or ro == 'nuclear':
        return 'nuclear'
    return None


def _coverage(v):
    if not isinstance(v, str) or not v.strip():
        return None
    s = set()
    for t in v.split('\n'):
        t = t.strip().lower()
        if t.startswith('fluke'):
            s.add('fluke')
        elif t.startswith('lft. dorsal'):
            s.add('L')
        elif t.startswith('rgt. dorsal'):
            s.add('R')
        elif 'peduncle' in t:
            s.add('ped')
    return s


def _frame_cues(text):
    """[(kind, camera, frame)] from a free-text comment: 'start' / 'end' / None for each 'frame NNNN' with a camera C<n> next to it."""
    out = []
    if not isinstance(text, str):
        return out
    for m in re.finditer(r'frame\s+(?:number\s+)?(\d{3,4})', text, re.I):
        after = text[m.end():m.end() + 12]
        before = text[max(0, m.start() - 40):m.start()]
        c = re.search(r'(C\d)', after) or (list(re.finditer(r'(C\d)', before)) or [None])[-1]
        if not c:
            continue
        b = before.lower()
        kind = 'start' if 'start' in b else 'end' if re.search(r'\bend', b) else None
        out.append((kind, c.group(1), int(m.group(1))))
    return out


_T = None


def _tables():
    global _T
    if _T is None:
        from tables import tables
        _T = tables()
    return _T


def log_day(day):
    from tables import xsec
    T = _tables()
    S, A = T['sighting'], T['animals']
    s = S[(S['day'] == _ok(day)) & S['Common name'].fillna('').str.contains('Humpback')]
    groups, whales, anchors = {}, collections.defaultdict(dict), []
    for _, r in s.iterrows():
        k = int(r['Sighting#'])
        a0, a1 = xsec(r['Start time']), xsec(r['End time'])
        com = r['Sightings comment'] if isinstance(r['Sightings comment'], str) else ''
        alias = re.search(r'(?:same as|continuation of)\s+group\s+(\d+)', com, re.I)
        size = r['Actual Gr size'] if pd.notna(r['Actual Gr size']) else r['Gr size avg']
        groups[k] = dict(start=a0, end=a1, type=str(r['Group type']).strip() if pd.notna(r['Group type']) else None,
                         size=float(size) if pd.notna(size) else None, alias=int(alias.group(1)) if alias else None,
                         cues=[c for c in _frame_cues(com) if c[0]])
        for _, w in A[A['Sighting_ID'] == r['sight_id']].iterrows():
            if pd.isna(w['Whale #']):
                continue
            n = int(w['Whale #'])
            h, f = _norm_shape(w['Dorsal fin'])
            fl = str(w['Fluke #']).strip() if pd.notna(w['Fluke #']) else ''
            wp = str(w['White_pec']).strip()[:1].upper() if pd.notna(w['White_pec']) else ''
            sex = w['Sex'] if w['Sex'] in ('Male', 'Female') else None
            whales[k][n] = dict(height=h, form=f, fluke=int(fl) if fl in '12345' and fl else None, wp={'Y': 1, 'N': 0}.get(wp),
                                sex=sex, role=_role(w['Nickname'], w['Ind_role'], w['Age class']), cov=_coverage(w['Photo coverage']))
            for txt in (w['Nickname'], w['Animal comment']):
                for kind, cam, fr in _frame_cues(txt):
                    anchors.append((k, n, cam, fr))
    return dict(groups=groups, whales=dict(whales), anchors=sorted(set(anchors)))


# ---- the photos -----------------------------------------------------------------------------------------------------
def truth_of(a):
    """('G', 3, 2) for '..._G03A02'; ('O', None, None) for a passer-by; None without a catalogue animal."""
    if not a:
        return None
    m = re.search(r'_G(\d+)A(\d+)', a)
    return ('G', int(m.group(1)), int(m.group(2))) if m else ('O', None, None)


def clusters_day(day):
    D = Day(_ok(day))
    d = DAYS / day / 'safe'
    ph = json.load(open(d / 'photos.json'))
    vc = json.load(open(d / 'view_class.json'))
    V, BOX = vc['views'], vc['box']
    vf = np.load(d / 'view_feat.npz')
    vkeys = [str(k) for k in vf['keys']]
    X = vf['X'].astype(np.float32).reshape(len(vkeys), 4, 768)
    X = (X / np.maximum(np.linalg.norm(X, axis=2, keepdims=True), 1e-6)).reshape(len(vkeys), -1)
    vidx = {k: i for i, k in enumerate(vkeys)}
    z = np.load(HW / f'{day}.npz')
    H = z['X'].astype(np.float32)
    H /= np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-9)
    hidx = {str(k): i for i, k in enumerate(z['keys'])}
    n_inst = {p: len(v.get('instances') or []) for p, v in ph.items()}
    inst = {}
    for p, v in ph.items():
        for it in v.get('instances') or []:
            inst[it['key']] = it
    photo_of = collections.defaultdict(set)
    for k, c in D.C.items():
        for key in c['keys']:
            photo_of[key.rsplit('~', 1)[0]].add(k)
    out = {}
    for k, c in D.C.items():
        keys = c['keys']
        ptimes = [ph[p]['time'] for p in c['photos'] if p in ph]
        cams = collections.Counter(cam_frame(p)[0] for p in c['photos'])
        fins, flukes, flank = [], [], collections.Counter()
        fin_h = []
        rel, app, multi = [], [], 0
        light = []
        for key in keys:
            v = V.get(key)
            p = key.rsplit('~', 1)[0]
            P = ph.get(p, {})
            it = inst.get(key)
            if v and v[1] >= 0.5:
                if key in hidx:
                    fins.append(hidx[key])
                if P.get('head_side') in (1, -1) and (P.get('cue_confidence') or 0) >= 0.5:
                    flank[P['head_side']] += 1
                b = BOX.get(key)
                if b:
                    fin_h.append(max(b[2] - b[0], b[3] - b[1]))
            if v and v[0] >= 0.5 and key in vidx:
                flukes.append(vidx[key])
                if it and it.get('lab_whale'):
                    light.append([it['lab_whale'][0] - (it.get('lab_water') or [0])[0], it['lab_whale'][0], it.get('lum_std') or 0, it.get('contrast') or 0])
            if it:
                a = float(it.get('area') or 0)
                if P.get('focal_mm') and a > 0:
                    app.append(0.5 * np.log(a) - np.log(P['focal_mm']))
                others = [float(o.get('area') or 0) for o in P.get('instances') or [] if o['key'] != key]
                if others and a > 0:
                    multi += 1
                    rel.append(np.log(a / max(max(others), 1.0)))
        nb = [vidx[key] for key in keys if key in vidx]
        out[k] = dict(
            id=k, maj=D.maj[k], truth=truth_of(D.maj[k]), size=len(c['photos']), photos=list(c['photos']),
            t0=tod(day, c['start']), t1=tod(day, c['end']), tmid=tod(day, float(np.median(ptimes)) if ptimes else (c['start'] + c['end']) / 2),
            cam=cams.most_common(1)[0][0] if cams else None, frames=[cam_frame(p) for p in c['photos']],
            fluke=bool(flukes), fin=bool(fins) or D.fin[k], flankL=flank[-1] > 0, flankR=flank[1] > 0,
            pfluke=max([V[key][0] for key in keys if key in V] or [0]), pfin=max([V[key][1] for key in keys if key in V] or [0]),
            hw=H[fins].mean(0) if fins else None, hw_inst=H[fins] if fins else None, fin_px=float(max(fin_h)) if fin_h else 0.0,
            base_fluke=X[flukes].mean(0) if flukes else None, base_fluke_keys=[vkeys[i] for i in flukes],
            base_all=X[nb].mean(0) if nb else None, light=np.mean(light, 0) if light else None,
            frac_multi=multi / max(1, len(keys)), rel=(float(np.median(rel)), float(np.min(rel)), float(np.max(rel))) if rel else None,
            app=float(np.median(app)) if app else None,
            co=sorted({o for key in keys for o in photo_of[key.rsplit('~', 1)[0]] if o != k}))
    return out


def links_day(day):
    L = json.load(open(DAYS / _ok(day) / 'safe' / 'link_scores.json'))['pairs']
    return [(r['a'], r['b'], r['gap'], r['tier'], r.get('lp')) for r in L]


def build(day):
    lg = log_day(day)
    if not lg['groups']:
        return None
    rec = dict(day=day, log=lg, clusters=clusters_day(day), links=links_day(day))
    (CACHE / 'day').mkdir(parents=True, exist_ok=True)
    pickle.dump(rec, open(CACHE / 'day' / f'{day}.pkl', 'wb'))
    return rec


def load(day):
    return pickle.load(open(CACHE / 'day' / f'{_ok(day)}.pkl', 'rb'))


def days():
    """The train + val days that have a log (cached by build)."""
    return [d for d in ALL24 if (CACHE / 'day' / f'{d}.pkl').exists()]


if __name__ == '__main__':
    for d in ALL24:
        r = build(d)
        if r is None:
            print(d, 'no log: skipped')
            continue
        C = r['clusters']
        print(d, 'groups', len(r['log']['groups']), 'whales', sum(len(v) for v in r['log']['whales'].values()), 'clusters', len(C),
              'labelled', sum(c['truth'] is not None for c in C.values()), 'frame cues', sum(len(g['cues']) for g in r['log']['groups'].values()),
              'aliases', sum(g['alias'] is not None for g in r['log']['groups'].values()), 'anchors', len(r['log']['anchors']))
