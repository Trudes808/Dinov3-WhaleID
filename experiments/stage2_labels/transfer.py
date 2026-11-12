"""Out-of-day check against the catalog: frozen window matching (whitening + 'where to look' model fitted on
2025-01-02 only) next to the student channels, on the val days where the student is out-of-sample.
All cluster pairs within 15 min; truth = catalog majority animal of the two clusters."""
import pickle, numpy as np, json, collections, torch, sys
from match import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import rankdata
DAYS = '/home/sat3737/NOAA_whales/Dinov3-WhaleID/experiments/days_all'
REF = '2025-01-02'
DS = pickle.load(open('ds.pkl', 'rb'))

def sq(b): return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, max(b[2] - b[0], b[3] - b[1]) * 1.15)
def iou_many(geo, c, ids):
    g = geo[ids]; x0 = np.maximum(c[0] - c[2] / 2, g[:, 0] - g[:, 2] / 2); x1 = np.minimum(c[0] + c[2] / 2, g[:, 0] + g[:, 2] / 2)
    y0 = np.maximum(c[1] - c[2] / 2, g[:, 1] - g[:, 2] / 2); y1 = np.minimum(c[1] + c[2] / 2, g[:, 1] + g[:, 2] / 2)
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None); return inter / (c[2] ** 2 + g[:, 2] ** 2 - inter)

# ---- reference day: whitening and the where-to-look model (all boxes)
ref = load(REF); mu, Pm = whiten_fit(ref['F'], 128); Zr = transform(ref['F'], mu, Pm).float().cpu().numpy()
byr = collections.defaultdict(list)
for i, k in enumerate(ref['inst']): byr[k].append(i)
byr = {k: np.array(v) for k, v in byr.items()}
boxes = collections.defaultdict(list)
for g in DS['regions']:
    for s in 'ab':
        i = ref['kid'].get(g['ikey_' + s])
        if i is not None: boxes[i].append(sq(g['box_' + s]))
lab = np.full(len(Zr), -1, np.int8)
for i, bl in boxes.items():
    w = byr[i]; m = np.max(np.stack([iou_many(ref['geo'], b, w) for b in bl]), 0); lab[w[m >= 0.3]] = 1; lab[w[m < 0.05]] = 0
tr = lab >= 0; SC = StandardScaler().fit(Zr[tr]); CLF = LogisticRegression(C=0.1, class_weight='balanced', max_iter=400).fit(SC.transform(Zr[tr]), lab[tr])
SAL_CV = pickle.load(open(f'saliency_{REF}.pkl', 'rb'))['sal_app']     # out-of-fold saliency on the reference day itself

@torch.no_grad()
def topk_scores(Z, sal, inst, size, n, topk=10):
    by = collections.defaultdict(list)
    for i, k in enumerate(inst): by[k].append(i)
    idx = np.full((n, topk), -1, np.int64)
    for i, w in by.items():
        w = np.array(w); t = w[np.argsort(-sal[w])[:topk]]; idx[i, :len(t)] = t
    valid = torch.tensor(idx >= 0).cuda(); ti = torch.tensor(np.where(idx >= 0, idx, 0)).cuda()
    Zs = Z[ti]; sz = torch.tensor(size).cuda()[ti]
    D = torch.full((n, n), float('nan'), device='cuda')
    for a in range(0, n, 64):
        S = torch.einsum('ikd,jld->ijkl', Zs[a:a + 64], Zs)
        ok = ((sz[a:a + 64][:, None, :, None] - sz[None, :, None, :]).abs() <= 1) & valid[a:a + 64][:, None, :, None] & valid[None, :, None, :]
        S = S.masked_fill(~ok, -9.0).flatten(2).amax(2); D[a:a + 64] = torch.where(S > -8, S, torch.full_like(S, float('nan')))
    return D.cpu().numpy()

def run(day):
    d = ref if day == REF else load(day); n = len(d['keys']); kid = d['kid']
    Z = transform(d['F'], mu, Pm).float()
    sal = SAL_CV if day == REF else CLF.decision_function(SC.transform(Z.cpu().numpy())).astype(np.float32)
    D_sal = topk_scores(Z, sal, d['inst'], d['size'], n, 10); D_sal = np.fmax(D_sal, D_sal.T)
    M, A = best_match(Z.half(), d['inst'], d['size'], n); V = M.astype(np.float32); V[V <= -1] = np.nan
    D_all, _ = instance_scores(V, d['inst'], n); D_all = np.fmax(D_all, D_all.T)
    z = json.load(open(f'{DAYS}/{day}/safe/clusters.json')); C = {c['id']: c for c in z['clusters']}
    members = {c: [kid[k] for k in cl['keys'] if k in kid] for c, cl in C.items()}
    rows = pickle.load(open(f'feats_{day}_all.pkl', 'rb'))
    def cl_score(D, a, b, how):
        ia, ib = members[a], members[b]
        if not ia or not ib: return np.nan
        blk = D[np.ix_(ia, ib)]
        if not np.isfinite(blk).any(): return np.nan
        if how == 'max': return float(np.nanmax(blk))
        r = np.where(np.isfinite(blk), blk, -np.inf); r1 = r.max(1); r0 = r.max(0)
        return float(0.5 * (r1[np.isfinite(r1)].mean() + r0[np.isfinite(r0)].mean()))
    S = collections.defaultdict(list)
    for r in rows:
        for nm in ('app_fin_max', 'app_body_max', 'app_best', 'app_fluke_max'): S[nm].append(r.get(nm, np.nan) if r.get(nm) is not None else np.nan)
        S['win_all_max'].append(cl_score(D_all, r['a'], r['b'], 'max')); S['win_all_rm'].append(cl_score(D_all, r['a'], r['b'], 'rm'))
        S['win_sal_max'].append(cl_score(D_sal, r['a'], r['b'], 'max')); S['win_sal_rm'].append(cl_score(D_sal, r['a'], r['b'], 'rm'))
    S = {k: np.array(v, float) for k, v in S.items()}
    gap = np.array([r['gap'] for r in rows]); cat = np.array([r['cat'] for r in rows])
    pickle.dump(dict(S=S, gap=gap, cat=cat, pairs=[(r['a'], r['b']) for r in rows], cat_a=[r['cat_a'] for r in rows], cat_b=[r['cat_b'] for r in rows], D_sal=D_sal, D_all=D_all, keys=d['keys']), open(f'transfer_raw_{day}.pkl', 'wb'))
    def rk(v, m):
        x = v[m]; x = np.where(np.isfinite(x), x, np.nanmin(x) - 1 if np.isfinite(x).any() else 0.0); return rankdata(x) / len(x)
    out = {}
    for sub, g in (('gap>1.5s', 1.5), ('gap>60s', 60.0)):
        m = (gap > g) & (cat != '?'); y = (cat[m] == 'same').astype(int)
        cols = {'student fin max': rk(S['app_fin_max'], m), 'student body max': rk(S['app_body_max'], m), 'student best part': rk(S['app_best'], m),
                'window all (row-max)': rk(S['win_all_rm'], m), 'window where-to-look (row-max)': rk(S['win_sal_rm'], m), 'window where-to-look (max)': rk(S['win_sal_max'], m)}
        cols['student fin + window'] = cols['student fin max'] + cols['window where-to-look (row-max)']
        cols['student best + window'] = cols['student best part'] + cols['window where-to-look (row-max)']
        cols['student fin + body + window'] = cols['student fin max'] + cols['student body max'] + cols['window where-to-look (row-max)']
        cols['student fin + body'] = cols['student fin max'] + cols['student body max']
        res = {}
        for nm, v in cols.items():
            order = np.argsort(-v, kind='stable'); yy = y[order]; wrong = np.cumsum(1 - yy)
            res[nm] = (float(roc_auc_score(y, v)), float(average_precision_score(y, v)), int(yy[wrong <= 0].sum()), int(yy[wrong <= 5].sum()), int(yy[wrong <= 20].sum()))
        out[sub] = (int(y.sum()), int((1 - y).sum()), res)
    return out

if __name__ == '__main__':
    days = sys.argv[1:] or [REF, '2025-01-15', '2025-02-27', '2025-03-14']
    allres = {}
    for day in days:
        allres[day] = run(day)
        for sub, (npos, nneg, res) in allres[day].items():
            print(f'\n{day}  {sub}: catalog-same pairs {npos}, catalog-different pairs {nneg}')
            for nm, (auc, ap, r0, r5, r20) in res.items():
                print(f'   {nm:32s} AUC {auc:.3f}  AP {ap:.3f} | same pairs ranked above the 1st / 6th / 21st wrong pair: {r0:4d} {r5:4d} {r20:4d}')
    pickle.dump(allres, open('transfer.pkl', 'wb'))
    print('\n== mean over the three val days ==')
    vd = [d for d in days if d != REF]
    for sub in ('gap>1.5s', 'gap>60s'):
        print(sub)
        for nm in allres[vd[0]][sub][2]:
            print(f'   {nm:32s} AUC {np.mean([allres[d][sub][2][nm][0] for d in vd]):.3f}  AP {np.mean([allres[d][sub][2][nm][1] for d in vd]):.3f} | same above the 1st / 6th / 21st wrong (sum): {sum(allres[d][sub][2][nm][2] for d in vd)} {sum(allres[d][sub][2][nm][3] for d in vd)} {sum(allres[d][sub][2][nm][4] for d in vd)}')
