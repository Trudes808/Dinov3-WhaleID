"""Use the researcher's boxes to learn WHERE to look (S) and HOW to score a window pair (H), grouped CV.
S: saliency classifier on windows (on a researcher box vs elsewhere on the same whale) -> match only the top-K windows.
H: diagonal bilinear window-pair scorer + unary terms, positives = boxed window pairs, negatives = hardest window
   pairs between different whales (mined), instance-pair score = max over window pairs."""
import pickle, numpy as np, json, collections, torch, sys, time
from match import *
from evalx import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
day = '2025-01-02'
ROOT = f'/home/sat3737/NOAA_whales/Dinov3-WhaleID/experiments/days_all/{day}'
d = load(day); n = len(d['keys']); kid = d['kid']; geo = d['geo']; inst = d['inst']; size = d['size']; N = len(inst)
DIM = int(sys.argv[1]) if len(sys.argv) > 1 else 128
mu, Pm = whiten_fit(d['F'], DIM); Z = transform(d['F'], mu, Pm).float(); Zc = Z.cpu().numpy()
z = json.load(open(f'{ROOT}/safe/clusters.json')); C = {c['id']: c for c in z['clusters']}
P = json.load(open(f'{ROOT}/safe/photos.json'))
cfold = DS['cfold']; K = DS['K']
inst_cluster = {}
for c in z['clusters']:
    for k in c['keys']:
        if k in kid: inst_cluster[kid[k]] = c['id']
ifold = np.array([cfold.get(inst_cluster.get(i, -1), -1) for i in range(n)])
wfold = ifold[inst]
by = collections.defaultdict(list)
for i, k in enumerate(inst): by[k].append(i)
by = {k: np.array(v) for k, v in by.items()}
# geometry of a window inside its instance
bbox = {}
for stem, p in P.items():
    for it in p['instances']:
        if it['key'] in kid: bbox[kid[it['key']]] = it['bbox']
G = np.zeros((N, 6), np.float32)
for i in range(n):
    if i not in by: continue
    w = by[i]; x0, y0, x1, y1 = bbox[i]; bw, bh = max(x1 - x0, 1), max(y1 - y0, 1)
    rx = (geo[w, 0] - x0) / bw; ry = (geo[w, 1] - y0) / bh
    G[w] = np.stack([geo[w, 3], np.log(geo[w, 2] / np.sqrt(bw * bh)), rx, ry, np.abs(rx - 0.5), np.abs(ry - 0.5)], 1)

def sq(b): return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, max(b[2] - b[0], b[3] - b[1]) * 1.15)
def iou_many(c, ids):
    g = geo[ids]; x0 = np.maximum(c[0] - c[2] / 2, g[:, 0] - g[:, 2] / 2); x1 = np.minimum(c[0] + c[2] / 2, g[:, 0] + g[:, 2] / 2)
    y0 = np.maximum(c[1] - c[2] / 2, g[:, 1] - g[:, 2] / 2); y1 = np.minimum(c[1] + c[2] / 2, g[:, 1] + g[:, 2] / 2)
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None); return inter / (c[2] ** 2 + g[:, 2] ** 2 - inter)

# ---------------------------------------------------------------- S: saliency
boxes = collections.defaultdict(list)          # instance -> list of square boxes
for g in DS['regions']:
    for s in 'ab':
        i = kid.get(g['ikey_' + s])
        if i is not None: boxes[i].append(sq(g['box_' + s]))
lab = np.full(N, -1, np.int8); best_iou = np.zeros(N, np.float32)
for i, bl in boxes.items():
    w = by[i]; m = np.max(np.stack([iou_many(b, w) for b in bl]), 0); best_iou[w] = m
    lab[w[m >= 0.3]] = 1; lab[w[m < 0.05]] = 0
print(f'saliency data: boxed instances {len(boxes)}, positive windows {(lab == 1).sum()}, negative windows {(lab == 0).sum()}')

def saliency(feats, Creg=0.1):
    X = feats; out = np.zeros(N, np.float32); models = []
    for f in range(K):
        tr = (wfold != f) & (lab >= 0)
        sc = StandardScaler().fit(X[tr]); clf = LogisticRegression(C=Creg, class_weight='balanced', max_iter=400).fit(sc.transform(X[tr]), lab[tr])
        models.append((sc, clf)); te = wfold == f
        out[te] = clf.decision_function(sc.transform(X[te]))
    te = wfold < 0
    if te.any(): out[te] = np.mean([m.decision_function(s.transform(X[te])) for s, m in models], 0)
    return out

def hit_rates(sal, name):
    ks = (1, 5, 10, 20); hits = {k: [] for k in ks}; ch = {k: [] for k in ks}
    rng = np.random.default_rng(0)
    for i, bl in boxes.items():
        w = by[i]; order = w[np.argsort(-sal[w])]
        for b in bl:
            pos = iou_many(b, order) >= 0.3
            if not pos.any(): continue
            first = int(np.argmax(pos)); p = pos.sum(); m = len(w)
            for k in ks:
                hits[k].append(first < k)
                # chance: P(at least one positive among k random windows)
                ch[k].append(1 - np.prod([(m - p - t) / (m - t) for t in range(min(k, m - p))]) if m - p >= k else 1.0)
    print(f'{name:28s} box found in the top 1 / 5 / 10 / 20 windows: ' + ' / '.join(f'{np.mean(hits[k]):.3f}' for k in ks) + '   chance: ' + ' / '.join(f'{np.mean(ch[k]):.3f}' for k in ks) + f'   (n={len(hits[1])})')

sal_geo = saliency(G); hit_rates(sal_geo, 'saliency geometry only')
sal_app = saliency(Zc); hit_rates(sal_app, 'saliency appearance only')
sal = saliency(np.concatenate([Zc, G], 1)); hit_rates(sal, 'saliency appearance+geometry')
pickle.dump(dict(sal=sal, sal_geo=sal_geo, sal_app=sal_app), open(f'saliency_{day}.pkl', 'wb'))

size_t = torch.tensor(size).cuda()
def pair_block(wi, wj, w1=None, a=None, b=0.0):
    """score matrix between window index arrays wi, wj (size-compatible pairs only)."""
    ti = torch.tensor(wi).cuda(); tj = torch.tensor(wj).cuda()
    Zi = Z[ti]; Zj = Z[tj]
    S = (Zi * w1) @ Zj.T if w1 is not None else Zi @ Zj.T
    if a is not None: S = S + a[ti][:, None] + a[tj][None, :] + b
    ok = (size_t[ti][:, None] - size_t[tj][None, :]).abs() <= 1
    return S.masked_fill(~ok, -1e4)

def record_scores(windows_of, w1=None, a=None, b=0.0, only=None):
    """per record: key-pair score, max over instance pairs, row-max mean."""
    key = np.full(len(RECS), np.nan); mx = np.full(len(RECS), np.nan); rm = np.full(len(RECS), np.nan); cache = {}
    def sc(i, j):
        if (i, j) not in cache:
            wi, wj = windows_of(i), windows_of(j)
            cache[(i, j)] = float(pair_block(wi, wj, w1, a, b).max()) if len(wi) and len(wj) else np.nan
        return cache[(i, j)]
    for q, r in enumerate(RECS):
        if only is not None and not only[q]: continue
        ia = [kid[k] for k in C[r['a']]['keys'] if k in kid]; ib = [kid[k] for k in C[r['b']]['keys'] if k in kid]
        ka, kb = kid.get(r['key_a']), kid.get(r['key_b'])
        if ka is not None and kb is not None: key[q] = sc(ka, kb)
        if ia and ib:
            if len(ia) * len(ib) > 64:   # keep the instances closest in time to the other cluster's key photo out of scope: subsample evenly
                ia = ia[::max(1, len(ia) // 8)][:8]; ib = ib[::max(1, len(ib) // 8)][:8]
            blk = np.array([[sc(i, j) for j in ib] for i in ia])
            if np.isfinite(blk).any():
                mx[q] = np.nanmax(blk); rm[q] = 0.5 * (np.nanmax(blk, 1).mean() + np.nanmax(blk, 0).mean())
    return key, mx, rm

SUB = (('gap>1.5s', 1.5), ('gap>60s', 60.0))
print('\n-- cosine on all windows (reference) and on the top-K salient windows --')
k_, m_, r_ = record_scores(lambda i: by.get(i, np.array([], int)))
report('cosine all windows, key pair', k_, subsets=SUB); report('cosine all windows, row-max', r_, subsets=SUB)
res = {'cos_all': (k_, m_, r_)}
for name, s in (('sal', sal), ('salgeo', sal_geo)):
    for topk in (5, 10, 20, 40):
        top = {i: w[np.argsort(-s[w])[:topk]] for i, w in by.items()}
        k_, m_, r_ = record_scores(lambda i: top.get(i, np.array([], int)))
        report(f'cosine top-{topk} {name}, key pair', k_, subsets=SUB); report(f'cosine top-{topk} {name}, row-max', r_, subsets=SUB)
        res[f'{name}_top{topk}'] = (k_, m_, r_)

# ---------------------------------------------------------------- H: learned window-pair scorer with mined negatives
print('\n-- learned window-pair scorer (diagonal bilinear + unary), hard negatives mined from different whales --')
def phi(u, v):
    zu, zv = Zc[u], Zc[v]
    return np.concatenate([zu * zv * DIM, (zu ** 2 + zv ** 2) * DIM, G[u] + G[v], np.concatenate([sal[u, None] + sal[v, None]], 1)], 1)
def fit_scorer(f, rounds=3, M=30, Creg=0.05, use_unary=True, seed=0):
    rng = np.random.default_rng(seed)
    pu, pv = [], []
    for g in DS['regions']:
        r = RECS[g['rec']]
        if cfold[r['a']] == f or cfold[r['b']] == f: continue
        i, j = kid.get(g['ikey_a']), kid.get(g['ikey_b'])
        if i is None or j is None: continue
        wa, wb = by[i], by[j]; ia_, ib_ = iou_many(sq(g['box_a']), wa), iou_many(sq(g['box_b']), wb)
        ua = wa[np.argsort(-ia_)[:4]]; ua = ua[iou_many(sq(g['box_a']), ua) >= 0.3]
        vb = wb[np.argsort(-ib_)[:4]]; vb = vb[iou_many(sq(g['box_b']), vb) >= 0.3]
        for u in ua:
            for v in vb:
                if abs(size[u] - size[v]) <= 1: pu.append(u); pv.append(v)
    pu, pv = np.array(pu), np.array(pv)
    negpairs = [(kid[r['key_a']], kid[r['key_b']]) for r in RECS if truth(r) == 0 and cfold[r['a']] != f and cfold[r['b']] != f and r['key_a'] in kid and r['key_b'] in kid]
    nu, nv = [], []
    def mine(w1=None, a=None, b=0.0, randoms=True):
        for (i, j) in negpairs:
            wi, wj = by[i], by[j]; S = pair_block(wi, wj, w1, a, b)
            flat = S.flatten(); kk = min(M, flat.numel()); top = torch.topk(flat, kk).indices.cpu().numpy()
            nu.extend(wi[top // len(wj)]); nv.extend(wj[top % len(wj)])
            if randoms:
                ok = (S > -1e3).nonzero().cpu().numpy()
                if len(ok):
                    pick = ok[rng.choice(len(ok), min(M, len(ok)), replace=False)]; nu.extend(wi[pick[:, 0]]); nv.extend(wj[pick[:, 1]])
    mine()
    w1 = a = None; b = 0.0
    for rd in range(rounds):
        U = np.r_[pu, np.array(nu)]; V = np.r_[pv, np.array(nv)]; y = np.r_[np.ones(len(pu)), np.zeros(len(nu))]
        X = phi(U, V); X = np.concatenate([X, phi(V, U)]); y2 = np.r_[y, y]      # symmetric by construction, harmless duplicate
        if not use_unary: X = X[:, :DIM]
        sc = StandardScaler().fit(X); clf = LogisticRegression(C=Creg, class_weight='balanced', max_iter=600).fit(sc.transform(X), y2)
        w = clf.coef_[0] / sc.scale_; b = float(clf.intercept_[0] - (clf.coef_[0] * sc.mean_ / sc.scale_).sum())
        w1 = torch.tensor(w[:DIM] * DIM, dtype=torch.float32).cuda()
        if use_unary:
            au = (Zc ** 2 * DIM) @ w[DIM:2 * DIM] + G @ w[2 * DIM:2 * DIM + 6] + sal * w[2 * DIM + 6]
            a = torch.tensor(au, dtype=torch.float32).cuda()
        else: a = torch.zeros(N).cuda()
        if rd < rounds - 1: mine(w1, a, b, randoms=False)
    return w1, a, b, len(pu), len(nu)

def run_H(tag, **kw):
    acc = {k: np.zeros(len(RECS)) for k in ('key', 'max', 'rm')}; cnt = np.zeros(len(RECS)); models = []
    for f in range(K):
        w1, a, b, npos, nneg = fit_scorer(f, **kw); models.append((w1, a, b))
        only = np.array([cfold[r['a']] == f or cfold[r['b']] == f for r in RECS])
        k_, m_, r_ = record_scores(lambda i: by.get(i, np.array([], int)), w1, a, b, only=only)
        for nm, v in (('key', k_), ('max', m_), ('rm', r_)): acc[nm][only] += np.nan_to_num(v[only], nan=-50.0)
        cnt += only
    out = {k: v / np.maximum(cnt, 1) for k, v in acc.items()}
    print(f'{tag}: positives per fold ~{npos}, negatives ~{nneg}')
    report(f'{tag}, key pair', out['key'], subsets=(('all', -1.0),) + SUB); report(f'{tag}, cluster max', out['max'], subsets=SUB); report(f'{tag}, row-max', out['rm'], subsets=SUB)
    return out, models

res['H'], models = run_H('learned scorer')
res['H_nounary'], _ = run_H('learned scorer, no unary terms', use_unary=False)
res['H_1round'], _ = run_H('learned scorer, no mining rounds', rounds=1)

# localisation of the learned scorer on held-out positive links
hit = []; both = []
for g in DS['regions']:
    r = RECS[g['rec']]; f = cfold[r['a']]; w1, a, b = models[f]
    i, j = kid.get(g['ikey_a']), kid.get(g['ikey_b'])
    if i is None or j is None: continue
    wi, wj = by[i], by[j]; S = pair_block(wi, wj, w1, a, b); q = int(S.argmax()); u, v = wi[q // len(wj)], wj[q % len(wj)]
    ha = iou_many(sq(g['box_a']), np.array([u]))[0] > 0.15; hb = iou_many(sq(g['box_b']), np.array([v]))[0] > 0.15
    hit += [ha, hb]; both.append(ha and hb)
print(f'learned scorer: best window pair lands on a researcher box {np.mean(hit):.3f}; both ends {np.mean(both):.3f} (n={len(both)})')
pickle.dump(res, open(f'mil_scores_{DIM}.pkl', 'wb'))
