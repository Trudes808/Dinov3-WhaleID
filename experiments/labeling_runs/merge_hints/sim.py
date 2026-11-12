"""Merge hints of the Photo Labeling page: which animal-to-animal score and which threshold?
The catalogue plays a researcher who sorted every stack right EXCEPT that one animal was left in two pieces (the kind of
error of the 2024-12-23 test run). Positives = the two pieces; negatives = pairs of different catalogue animals of the
same encounter. Scores are built from the page's own stack-pair matrices exactly as its rank() does.
usage: sim.py val|train|eval|all [--user]"""
import json, sys, re, math, collections, itertools
import numpy as np
from sklearn.metrics import roc_auc_score
SPLITS = {'eval': ["2024-12-18", "2024-12-26", "2025-02-19", "2025-03-21"], 'val': ["2025-01-15", "2025-02-27", "2025-03-14"]}
BIT = dict(body=1, fin=2, fluke=4, finL=8, finR=16, flukeUnder=32, flukeTop=64); LOG61 = math.log1p(60)
sig = lambda z: 1 / (1 + math.exp(-z))
class Day:
    def __init__(s, day):
        D = json.load(open(f'cache/{day}.json')); s.day = day; s.D = D; s.split = D['split']; n = len(D['stacks']); s.n = n
        raw = np.frombuffer(open(f'cache/{day}.bin', 'rb').read(), np.uint8); s.P, s.A, s.F, s.Dn, s.B, s.Q = [raw[i * n * n:(i + 1) * n * n].reshape(n, n) for i in range(6)]
        s.blend = D['model']['blend']; P = D['photos']; st = D['stacks']
        s.t0 = np.array([x['t0'] for x in st]); s.t1 = np.array([x['t1'] for x in st])
        s.bits = [0] * n; s.cat = [None] * n; s.photos = [0] * n
        for x in st:
            b = 0; c = collections.Counter()
            for p, k in x['items']:
                if k >= 0: b |= P[p]['parts'][k]
                if k <= 0 and P[p].get('cat'): c[P[p]['cat']] += 1
            s.bits[x['id']] = b; s.photos[x['id']] = sum(1 for p, k in x['items'] if k <= 0)
            if x['kind'] != 'extra' and c: s.cat[x['id']] = c.most_common(1)[0][0]
        byP = collections.defaultdict(list)
        for x in st:
            for p, k in x['items']: byP[p].append(x['id'])
        s.conf = set()
        for ids in byP.values():
            for a in ids:
                for b in ids:
                    if a != b: s.conf.add((a, b))
        # encounters as the catalogue has them: a G group; a passer-by (O) joins the group whose time span holds it
        an = collections.defaultdict(list)
        for i, c in enumerate(s.cat):
            if c: an[c].append(i)
        s.animals = dict(an); grp = {}
        for c in an:
            m = re.match(r'G0*(\d+)A', c.split('_')[-1]); grp[c] = f'G{int(m.group(1))}' if m else None
        span = collections.defaultdict(lambda: [1e18, -1e18])
        for c, ids in an.items():
            if grp[c]: span[grp[c]][0] = min(span[grp[c]][0], s.t0[ids].min()); span[grp[c]][1] = max(span[grp[c]][1], s.t1[ids].max())
        k = 0
        for c, ids in sorted(an.items(), key=lambda kv: s.t0[kv[1]].min()):
            if grp[c]: continue
            t = s.t0[ids].min(); inside = [g for g, (a, b) in span.items() if a <= t <= b]
            if inside: grp[c] = inside[0]
            else: k += 1; grp[c] = f'O{k}'
        s.scope = collections.defaultdict(list)
        for c in an: s.scope[grp[c]].append(c)
    def gap(s, A, B):
        return float(np.maximum(0, np.maximum(s.t0[A][:, None], s.t0[B][None, :]) - np.minimum(s.t1[A][:, None], s.t1[B][None, :])).min())
    def conflict(s, A, B): return any((a, b) in s.conf for a in A for b in B)
    def parts_score(s, f, d, b, q, gap):
        Bm = s.blend; o = Bm['other']; lg = math.log1p(gap); x = [(b - 1) / 254 if b else 0, 1 if b else 0, q / 255, 1 if q else 0, lg, min(lg, LOG61)]
        other = sig(o['intercept'] + sum(c * v for c, v in zip(o['coef'], x)))
        fl = sig(Bm['fluke']['coef'] * (f - 1) / 254 + Bm['fluke']['intercept']) if f else None; do = sig(Bm['dorsal']['coef'] * (d - 1) / 254 + Bm['dorsal']['intercept']) if d else None
        w = Bm['weights']; num, den = w['other'] * other, w['other']
        if fl is not None: num += w['fluke'] * fl; den += w['fluke']
        if do is not None: num += w['dorsal'] * do; den += w['dorsal']
        return num / den, fl, do, other
    def scores(s, A, B):
        """Every candidate animal-to-animal score for two sets of stacks."""
        A = np.asarray(A); B = np.asarray(B); ix = np.ix_(A, B); P, F, Dn, Bd, Q = s.P[ix], s.F[ix].astype(float), s.Dn[ix].astype(float), s.B[ix].astype(float), s.Q[ix]; gap = s.gap(A, B)
        out = {'link': P.max() / 255}
        bl, fl, do, ot = s.parts_score(F.max(), Dn.max(), Bd.max(), int(Q.max()), gap); out.update(blend=bl, fluke=fl, dorsal=do, other=ot)
        def bm(M):        # best-match mean: every stack that shows the part against its best match on the other side, both ways
            ra = M.max(1); rb = M.max(0); ra = ra[ra > 0]; rb = rb[rb > 0]
            return 0.0 if not len(ra) or not len(rb) else 0.5 * (ra.mean() + rb.mean())
        out['blend_bm'] = s.parts_score(bm(F), bm(Dn), bm(Bd), int(Q.max()), gap)[0]
        # stack by stack: each stack of the smaller animal against the whole other animal, as rank() would score it; the mean
        S_, L_ = (A, B) if len(A) <= len(B) else (B, A); v = []
        for x in S_:
            jx = np.ix_([x], L_); v.append(s.parts_score(s.F[jx].max(), s.Dn[jx].max(), s.B[jx].max(), int(s.Q[jx].max()), s.gap(np.array([x]), L_))[0])
        out['stack_mean'] = float(np.mean(v)); out['stack_max'] = float(np.max(v))
        out['shared'] = ('fluke' if fl is not None else '') + ('dorsal' if do is not None else '')
        return out

NAMES = ['link', 'blend', 'blend_bm', 'stack_mean', 'stack_max']
def splits(day, c, rng):
    """Ways one animal may have been left in two pieces: (kind, piece 1, piece 2)."""
    ids = day.animals[c]; out = []
    if len(ids) < 2: return out
    fl = [i for i in ids if day.bits[i] & BIT['fluke']]; nf = [i for i in ids if not day.bits[i] & BIT['fluke']]
    if fl and nf: out.append(('fluke stacks apart', fl, nf))
    L = [i for i in ids if day.bits[i] & BIT['finL'] and not day.bits[i] & BIT['finR']]; R = [i for i in ids if day.bits[i] & BIT['finR'] and not day.bits[i] & BIT['finL']]
    if L and R:
        small = L if len(L) <= len(R) else R; out.append(('one fin side apart', small, [i for i in ids if i not in small]))
    o = sorted(ids, key=lambda i: day.t0[i]); g = [day.t0[o[j + 1]] - day.t1[o[j]] for j in range(len(o) - 1)]; j = int(np.argmax(g))
    if g[j] > 60: out.append(('apart at its longest gap', o[:j + 1], o[j + 1:]))
    for i in ids: out.append(('one stack apart', [i], [x for x in ids if x != i]))
    for _ in range(6):
        k = int(rng.integers(1, min(3, len(ids) - 1) + 1)); small = list(rng.choice(ids, k, replace=False)); out.append(('1 to 3 stacks apart', small, [x for x in ids if x not in small]))
    return out
def collect(days):
    rng = np.random.default_rng(0); pos, neg, enc = [], [], []
    for d in days:
        day = Day(d)
        for g, cs in day.scope.items():
            wh = []
            for a, b in itertools.combinations(cs, 2):
                A, B = day.animals[a], day.animals[b]
                if day.conflict(A, B): continue                      # two whales of one frame are never hinted
                r = day.scores(A, B); r.update(day=d, scope=g, pair=(a, b)); neg.append(r); wh.append(r)
            enc.append(dict(day=d, scope=g, animals=len(cs), pairs=wh))
            for c in cs:
                for kind, p1, p2 in splits(day, c, rng):
                    if day.conflict(p1, p2): continue
                    r = day.scores(p1, p2); r.update(day=d, scope=g, kind=kind, animal=c, n1=len(p1), n2=len(p2))
                    # where the true pair stands among the pairs the researcher would be shown: the two pieces against each other and against the other animals
                    others = []
                    for b in cs:
                        if b == c or day.conflict(p1, day.animals[b]): continue
                        others.append(day.scores(p1, day.animals[b]))
                    r['others'] = others; pos.append(r)
    return pos, neg, enc
def report(tag, days):
    pos, neg, enc = collect(days); kinds = ['fluke stacks apart', 'one fin side apart', 'apart at its longest gap', 'one stack apart', '1 to 3 stacks apart']
    print(f"\n===== {tag}: {len(days)} days, {len(enc)} encounters, {len(neg)} pairs of different animals; split cases {dict(collections.Counter(p['kind'] for p in pos))} =====")
    print('AUC, the two pieces of one animal against pairs of different animals of an encounter')
    print(f"   {'score':12s}" + ''.join(f'{k[:22]:>24s}' for k in kinds))
    for nm in NAMES:
        row = []
        for k in kinds:
            p = [x[nm] for x in pos if x['kind'] == k]; y = [1] * len(p) + [0] * len(neg); row.append(roc_auc_score(y, p + [x[nm] for x in neg]) if p else float('nan'))
        print(f"   {nm:12s}" + ''.join(f'{v:24.3f}' for v in row))
    print('rank of the true pair among the pairs with the small piece (pieces against each other, small piece against every other animal): first / in the top 2')
    for nm in NAMES:
        row = []
        for k in kinds:
            p = [x for x in pos if x['kind'] == k and x['others']]; r = [1 + sum(o[nm] >= x[nm] for o in x['others']) for x in p]
            row.append(f"{np.mean([v == 1 for v in r]):.2f} / {np.mean([v <= 2 for v in r]):.2f}" if r else '-')
        print(f"   {nm:12s}" + ''.join(f'{v:>24s}' for v in row))
    for nm in ('link', 'blend', 'stack_mean'):
        print(f"thresholds for '{nm}': share of split cases shown (by kind) | wrong hints: share of different-animal pairs shown, mean per encounter, encounters with none")
        for t in (0.85, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3):
            rec = [np.mean([x[nm] >= t for x in pos if x['kind'] == k]) if any(x['kind'] == k for x in pos) else float('nan') for k in kinds]
            fp = [sum(x[nm] >= t for x in e['pairs']) for e in enc]; multi = [f for e, f in zip(enc, fp) if e['animals'] >= 2]
            print(f"   >= {t:.2f}: " + '  '.join(f'{v:.2f}' for v in rec) + f"   | {np.mean([x[nm] >= t for x in neg]):.3f}  {np.mean(multi):.2f}  {np.mean([f == 0 for f in multi]):.2f}  (max {max(multi)})")
    sh = collections.Counter((p['kind'], p['shared'] or 'neither') for p in pos)
    print('parts the two pieces share:', {k: {s: v for (kk, s), v in sh.items() if kk == k} for k in kinds[:3]})
    return pos, neg, enc
if __name__ == '__main__':
    import os
    alld = sorted(f[:-5] for f in os.listdir('cache') if f.endswith('.json')); which = sys.argv[1] if len(sys.argv) > 1 else 'val'
    sel = {'val': SPLITS['val'], 'eval': SPLITS['eval'], 'train': [d for d in alld if d not in SPLITS['val'] + SPLITS['eval']]}
    for w in (['val', 'train'] if which == 'all' else [which]): report(w, sel[w])
