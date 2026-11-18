"""Sizing the levers for "cluster a new day" (2026-11-17). Read-only: cached safe clusters, view classes and catalogue
labels; part B also GETs the Labeling page's stacks and pair matrix from the running page (no POST, nothing written).

Score = the catalogue's list of animals per group. An animal is WHOLE when all its safe clusters sit in one piece,
CLEAN when that piece holds no other animal's cluster. A cluster's animal = the majority catalogue animal of its
photos; clusters without a catalogue animal are left out everywhere.

Part A  oracle rows: the catalogue joins every same-animal pair of clusters that meets a condition (a gap, a shared
        view). What a perfect short-gap linker / a perfect same-view matcher could reach. No mixing by construction.
Part B  today's evidence: the page's P(same whale) per stack pair, merged by average linkage inside each catalogue
        group (groups given), either down to the catalogue's count of animals or up to a threshold.

Days: val and train only. No eval day is read.   python size_levers.py [A] [B] [C] [D]
"""
import collections
import itertools
import json
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DAYS = REPO / 'experiments/days_all'
SPLIT = json.load(open(REPO / 'model_fine_tuning/splits/days.json'))


def major(c):
    e = sorted((c.get('catalog') or {}).items(), key=lambda kv: -kv[1])
    return e[0][0] if e else None


def group_of(a):
    m = re.search(r'_(G\d+)A\d+', a)
    return m.group(1) if m else 'O'


class Day:
    def __init__(s, day):
        s.day = day
        d = DAYS / day / 'safe'
        z = json.load(open(d / 'clusters.json'))
        s.C = {c['id']: c for c in z['clusters']}
        s.maj = {k: major(c) for k, c in s.C.items()}
        s.ids = [k for k in s.C if s.maj[k]]
        s.animals = sorted(set(s.maj[k] for k in s.ids))
        vf = d / 'view_class.json'
        views = json.load(open(vf))['views'] if vf.exists() else {}
        ph = json.load(open(d / 'photos.json'))
        s.fluke, s.fin, s.flank = {}, {}, {}
        for k, c in s.C.items():
            fl = fi = False
            fk = set()
            for key in c['keys']:
                v = views.get(key)
                if not v:
                    continue
                fl |= v[0] >= 0.5
                if v[1] >= 0.5:
                    fi = True
                    p = ph.get(key.rsplit('~', 1)[0]) or {}
                    if p.get('head_side') in (1, -1) and (p.get('cue_confidence') or 0) >= 0.5:
                        fk.add(p['head_side'])
            s.fluke[k], s.fin[k], s.flank[k] = fl, fi, fk
        s.has_views = bool(views)

    def gap(s, a, b):
        A, B = s.C[a], s.C[b]
        return max(0.0, max(A['start'], B['start']) - min(A['end'], B['end']))

    def size(s, k):
        return len(s.C[k]['photos'])


def components(ids, pairs):
    par = {k: k for k in ids}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for a, b in pairs:
        par[find(a)] = find(b)
    comp = collections.defaultdict(list)
    for k in ids:
        comp[find(k)].append(k)
    return list(comp.values())


def score(D, comps):
    """List-level counts for one day. comps = lists of labelled cluster ids."""
    where = collections.defaultdict(set)            # animal -> pieces holding one of its clusters
    content = []                                    # piece -> Counter(animal -> photos)
    for i, cm in enumerate(comps):
        cnt = collections.Counter()
        for k in cm:
            cnt[D.maj[k]] += D.size(k)
            where[D.maj[k]].add(i)
        content.append(cnt)
    found = {c.most_common(1)[0][0] for c in content}
    clean = {a for a, ps in where.items() if len(ps) == 1 and len(content[next(iter(ps))]) == 1}
    groups = collections.defaultdict(list)
    for a in where:
        groups[group_of(a)].append(a)
    return dict(animals=len(where), pieces=len(comps), per_animal=float(np.mean([len(v) for v in where.values()])),
                whole=sum(len(v) == 1 for v in where.values()), clean=len(clean), found=len(found),
                mixed=sum(len(c) > 1 for c in content), groups=len(groups),
                groups_right=sum(all(a in clean for a in g) for g in groups.values()),
                per_group={g: (len(v), sum(a in clean for a in v)) for g, v in groups.items()})


def total(rows):
    t = collections.Counter()
    for r in rows:
        for k in ('animals', 'pieces', 'whole', 'clean', 'found', 'mixed', 'groups', 'groups_right'):
            t[k] += r[k]
    t['per_animal'] = float(np.sum([r['per_animal'] * r['animals'] for r in rows]) / max(1, t['animals']))
    return t


def line(name, t):
    return (f"{name:<58s} animals {t['animals']:4d}  list {t['pieces']:5d}  pieces/animal {t['per_animal']:5.2f}  "
            f"whole+clean {t['clean']:4d} ({100 * t['clean'] / max(1, t['animals']):3.0f} %)  in the list by majority {t['found']:4d}  mixed pieces {t['mixed']:3d}  "
            f"groups right {t['groups_right']:3d} / {t['groups']:3d}")


# ---- part A: oracle rows --------------------------------------------------------------------------------------------
def cond_gap(w):
    return lambda D, a, b: D.gap(a, b) <= w


def c_fluke(D, a, b):
    return D.fluke[a] and D.fluke[b]


def c_fin_any(D, a, b):
    return D.fin[a] and D.fin[b]


def c_fin_flank(D, a, b):
    return D.fin[a] and D.fin[b] and bool(D.flank[a] & D.flank[b])


def any_of(*cs):
    return lambda D, a, b: any(c(D, a, b) for c in cs)


ROWS = [
    ('safe clusters as they are', None),
    ('perfect linking within 1.5 s', cond_gap(1.5)),
    ('perfect linking within 60 s', cond_gap(60)),
    ('perfect linking within 15 min', cond_gap(900)),
    ('perfect fluke matching alone (both show a fluke)', c_fluke),
    ('perfect dorsal matching alone, same flank', c_fin_flank),
    ('perfect dorsal matching alone, any flank', c_fin_any),
    ('60 s + fluke', any_of(cond_gap(60), c_fluke)),
    ('60 s + fluke + dorsal same flank', any_of(cond_gap(60), c_fluke, c_fin_flank)),
    ('60 s + fluke + dorsal any flank', any_of(cond_gap(60), c_fluke, c_fin_any)),
    ('15 min + fluke + dorsal any flank', any_of(cond_gap(900), c_fluke, c_fin_any)),
    ('60 s + dorsal same flank (no fluke matching)', any_of(cond_gap(60), c_fin_flank)),
    ('60 s + dorsal any flank (no fluke matching)', any_of(cond_gap(60), c_fin_any)),
    ('60 s + fluke + dorsal same or unknown flank', any_of(cond_gap(60), c_fluke, lambda D, a, b: D.fin[a] and D.fin[b] and (bool(D.flank[a] & D.flank[b]) or not D.flank[a] or not D.flank[b]))),
    ('1.5 s + fluke + dorsal any flank (no short-gap linker)', any_of(cond_gap(1.5), c_fluke, c_fin_any)),
]


def oracle_comps(D, cond, ids=None):
    ids = D.ids if ids is None else ids
    if cond is None:
        return [[k] for k in ids]
    pairs = [(a, b) for a, b in itertools.combinations(ids, 2) if D.maj[a] == D.maj[b] and cond(D, a, b)]
    return components(ids, pairs)


def part_a(days, label):
    Ds = [Day(d) for d in days]
    Ds = [D for D in Ds if D.has_views and D.ids]
    print(f'\n== part A, {label}: {len(Ds)} days, {sum(len(D.ids) for D in Ds)} labelled safe clusters ==')
    for name, cond in ROWS:
        print(line(name, total([score(D, oracle_comps(D, cond)) for D in Ds])))
    # what is left after 60 s + fluke + dorsal (same flank): the pieces of animals that are still split
    left = collections.Counter()
    n_split = 0
    for D in Ds:
        comps = oracle_comps(D, ROWS[8][1])
        per = collections.defaultdict(list)
        for cm in comps:
            per[D.maj[cm[0]]].append(cm)
        for a, cms in per.items():
            if len(cms) < 2:
                continue
            n_split += 1
            for cm in cms:
                fl, fi = any(D.fluke[k] for k in cm), any(D.fin[k] for k in cm)
                left['fluke + fin' if fl and fi else 'fluke only' if fl else 'fin only' if fi else 'neither'] += 1
    print(f'   after "60 s + fluke + dorsal same flank": {n_split} animals still split; their pieces by what they show: {dict(left)}')
    # leaving out the pieces that show neither a fluke nor a fin (after perfect 60-s linking)
    for name, cond in (('60 s, pieces without fluke or fin left out', cond_gap(60)),
                       ('60 s + fluke + dorsal same flank, such pieces left out', ROWS[8][1]),
                       ('60 s + fluke + dorsal any flank, such pieces left out', ROWS[9][1])):
        rows, lost = [], 0
        for D in Ds:
            base = oracle_comps(D, cond_gap(60))
            keep = [k for cm in base if any(D.fluke[k] or D.fin[k] for k in cm) for k in cm]
            lost += len(D.animals) - len({D.maj[k] for k in keep})
            if keep:
                rows.append(score(D, oracle_comps(D, cond, keep)))
        print(line(name, total(rows)) + f'  animals lost {lost}')
    views = collections.Counter()
    for D in Ds:
        for k in D.ids:
            views['fluke + fin' if D.fluke[k] and D.fin[k] else 'fluke' if D.fluke[k] else 'fin' if D.fin[k] else 'neither'] += 1
    print(f'   safe clusters by view: {dict(views)}')


# ---- part B: today's pair model, merged automatically ---------------------------------------------------------------
def fetch(day):
    base = f'http://localhost:8765/api/labeling/{day}'
    J = json.load(urllib.request.urlopen(base, timeout=600))
    raw = urllib.request.urlopen(base + '/pairs.bin', timeout=600).read()
    n = len(J['stacks'])
    M = np.frombuffer(raw, np.uint8)[:6 * n * n].reshape(6, n, n).astype(np.float64)
    M = (M + M.transpose(0, 2, 1)) / 2
    return J, dict(P=M[0] / 255, A=M[1] / 255, fluke=M[2], dorsal=M[3], body=M[4], seq=M[5] / 255)   # part matrices: 0 = not shown by both, else 1 + 254 x cosine


def agglo(S, cannot, K=None, thr=None, start=None):
    """Average linkage on similarity S with cannot-link pairs; stop at K pieces or when the best mean falls below thr.
    start = initial pieces (lists of indices)."""
    n = len(S)
    cl = [list(c) for c in start] if start is not None else [[i] for i in range(n)]
    m = len(cl)
    sums = np.zeros((m, m))
    bad = np.zeros((m, m), bool)
    for i in range(m):
        for j in range(i + 1, m):
            sums[i, j] = sums[j, i] = S[np.ix_(cl[i], cl[j])].sum()
            bad[i, j] = bad[j, i] = cannot[np.ix_(cl[i], cl[j])].any()
    size = np.array([len(c) for c in cl], float)
    alive = np.ones(m, bool)
    while alive.sum() > (K or 1):
        mean = sums / np.outer(size, size)
        mean[bad] = -1
        mean[~alive] = -1
        mean[:, ~alive] = -1
        np.fill_diagonal(mean, -1)
        i, j = np.unravel_index(np.argmax(mean), mean.shape)
        if mean[i, j] < 0 or (thr is not None and mean[i, j] < thr):
            break
        cl[i] += cl[j]
        sums[i] += sums[j]
        sums[:, i] += sums[:, j]
        bad[i] |= bad[j]
        bad[:, i] |= bad[:, j]
        size[i] += size[j]
        alive[j] = False
    return [cl[i] for i in range(m) if alive[i]]


def part_b(days):
    print(f'\n== part B, today\'s pair model on the val days (groups given; the pair model\'s weights were fitted on these days) ==')
    modes = [('stacks, nothing merged', dict(thr=2.0)), ('merge while mean P >= 0.9', dict(thr=0.9)), ('merge while mean P >= 0.7', dict(thr=0.7)),
             ('merge while mean P >= 0.5', dict(thr=0.5)), ('merge while mean P >= 0.3', dict(thr=0.3)), ('count of animals per group given', dict(count=True)),
             ('count given, appearance only (best part cosine)', dict(count=True, sim='A')),
             ('perfect 60-s linking first, then mean P >= 0.5', dict(thr=0.5, pre=True)), ('perfect 60-s linking first, then count given', dict(count=True, pre=True))]
    out = {name: [] for name, _ in modes}
    detail = collections.defaultdict(dict)
    for day in days:
        D = Day(day)
        try:
            J, MS = fetch(day)
        except Exception as e:                                     # the page is not running
            print(f'   {day}: page not reachable ({e}); part B skipped')
            return
        st = {s['cl']: s for s in J['stacks'] if s['cl'] is not None and s['cl'] in D.maj and D.maj[s['cl']]}
        by_group = collections.defaultdict(list)
        for k in st:
            by_group[group_of(D.maj[k])].append(k)
        for name, o in modes:
            comps = []
            for g, ks in by_group.items():
                ks = sorted(ks)
                idx = [st[k]['id'] for k in ks]
                S = MS[o.get('sim', 'P')][np.ix_(idx, idx)]
                photos = [{p for p, _ in st[k]['items']} for k in ks]
                cannot = np.array([[bool(photos[i] & photos[j]) and i != j for j in range(len(ks))] for i in range(len(ks))])
                start = None
                if o.get('pre'):
                    pos = {k: i for i, k in enumerate(ks)}
                    start = [[pos[k] for k in cm] for cm in oracle_comps(D, cond_gap(60), ks)]
                K = len({D.maj[k] for k in ks}) if o.get('count') else None
                for cm in agglo(S, cannot, K=K, thr=o.get('thr'), start=start):
                    comps.append([ks[i] for i in cm])
            r = score(D, comps)
            out[name].append(r)
            detail[name][day] = r
    for name, _ in modes:
        print(line(name, total(out[name])))
    for name in ('count of animals per group given', 'perfect 60-s linking first, then count given'):
        print(f'   {name}, per group (animals, whole + clean):')
        for day, r in detail[name].items():
            print(f'      {day}: ' + ', '.join(f'{g} {n}/{c}' for g, (n, c) in sorted(r['per_group'].items())) + f'   list {r["pieces"]} for {r["animals"]} animals, mixed pieces {r["mixed"]}')


# ---- part C: today's short-gap linker, joined automatically ----------------------------------------------------------
def part_c(days):
    print("\n== part C, today's sequence scores joined automatically on the val days (link_scores.json; union-find) ==")
    modes = [('within-burst pairs (<= 1.5 s) all joined', lambda r: r['gap'] <= 1.5),
             ('linker P >= 0.9 in 1.5-60 s', lambda r: 1.5 < r['gap'] <= 60 and (r.get('lp') or 0) >= 0.9),
             ('within-burst + linker P >= 0.9', lambda r: r['gap'] <= 1.5 or (r['gap'] <= 60 and (r.get('lp') or 0) >= 0.9)),
             ('within-burst + linker P >= 0.7', lambda r: r['gap'] <= 1.5 or (r['gap'] <= 60 and (r.get('lp') or 0) >= 0.7)),
             ('within-burst + linker P >= 0.5', lambda r: r['gap'] <= 1.5 or (r['gap'] <= 60 and (r.get('lp') or 0) >= 0.5))]
    for name, use in modes:
        rows, ok, bad = [], 0, 0
        for day in days:
            D = Day(day)
            L = json.load(open(DAYS / day / 'safe' / 'link_scores.json'))['pairs']
            pairs = [(r['a'], r['b']) for r in L if use(r) and D.maj.get(r['a']) and D.maj.get(r['b'])]
            ok += sum(D.maj[a] == D.maj[b] for a, b in pairs)
            bad += sum(D.maj[a] != D.maj[b] for a, b in pairs)
            rows.append(score(D, components(D.ids, pairs)))
        print(line(name, total(rows)) + f'  links right {ok} wrong {bad}')


# ---- part D: today's matchers on the pairs that matter (same group, more than 60 s apart) -----------------------------
def auc(pos, neg):
    if not len(pos) or not len(neg):
        return float('nan')
    v = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(v)) + 1.0
    for u in np.unique(v):                                        # mid-ranks for ties
        m = v == u
        if m.sum() > 1:
            r[m] = r[m].mean()
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def at_precision(pos, neg, p):
    """Same pairs found before precision falls below p (best prefix of the ranking)."""
    v = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.lexsort((y, -v))                                       # ties: different-animal pairs first (2026-11-18; they were counted in favour before)
    tp = np.cumsum(y[o])
    prec = tp / np.arange(1, len(o) + 1)
    ok = np.where(prec >= p)[0]
    return int(tp[ok[-1]]) if len(ok) else 0


def part_d(days):
    print('\n== part D, today\'s matchers on cluster pairs of one catalogue group more than 60 s apart (val days) ==')
    acc = collections.defaultdict(lambda: ([], []))
    for day in days:
        D = Day(day)
        z = np.load(DAYS / day / 'safe' / 'view_match.npz', allow_pickle=True)
        E = {}
        for part in ('fluke', 'dorsal'):
            e = z[f'{part}|E'].astype(np.float64)
            e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-9
            E[part] = dict(zip([str(k) for k in z[f'{part}|keys']], e))
        try:
            J, MS = fetch(day)
            sid = {s['cl']: s['id'] for s in J['stacks'] if s['cl'] is not None}
        except Exception:
            MS, sid = None, {}
        vec = {part: {k: np.array([E[part][key] for key in D.C[k]['keys'] if key in E[part]]) for k in D.ids} for part in E}
        for a, b in itertools.combinations(D.ids, 2):
            if group_of(D.maj[a]) != group_of(D.maj[b]) or D.gap(a, b) <= 60:
                continue
            same = D.maj[a] == D.maj[b]
            for part in ('fluke', 'dorsal'):
                A, B = vec[part][a], vec[part][b]
                if len(A) and len(B):
                    names = [f'{part}: view-match embedding']
                    if part == 'dorsal':
                        fa, fb = D.flank[a], D.flank[b]
                        rel = 'flank unknown' if not fa or not fb else 'same flank' if fa == fb and len(fa) == 1 else 'opposite flanks' if not (fa & fb) else 'mixed'
                        names.append(f'dorsal: view-match embedding, {rel}')
                    v = float((A @ B.T).max())
                    for n in names:
                        acc[n][0 if same else 1].append(v)
                if MS is not None and a in sid and b in sid:
                    c = MS[part][sid[a], sid[b]]
                    if c > 0:
                        acc[f'{part}: student part cosine (page)'][0 if same else 1].append((c - 1) / 254)
            if MS is not None and a in sid and b in sid:
                acc['page P(same whale)'][0 if same else 1].append(MS['P'][sid[a], sid[b]])
                acc['page best appearance cosine'][0 if same else 1].append(MS['A'][sid[a], sid[b]])
    for n in sorted(acc):
        pos, neg = np.array(acc[n][0]), np.array(acc[n][1])
        print(f'{n:<46s} same {len(pos):5d}  different {len(neg):5d}  AUC {auc(pos, neg):.3f}  same pairs found at precision 0.95: {at_precision(pos, neg, 0.95):4d}, at 0.90: {at_precision(pos, neg, 0.90):4d}')


if __name__ == '__main__':
    what = sys.argv[1:] or ['A', 'B', 'C', 'D']
    if 'A' in what:
        part_a(SPLIT['val'], 'val days')
        part_a(SPLIT['train'] + SPLIT['val'], 'train + val days')
    if 'B' in what:
        part_b(SPLIT['val'])
    if 'C' in what:
        part_c(SPLIT['val'])
    if 'D' in what:
        part_d(SPLIT['val'])
