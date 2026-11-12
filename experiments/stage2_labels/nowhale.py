"""Val-day rankings and score-order merging with the tracked non-whales (people, deck, gear; found by eye among
wide-angle clusters) left out."""
import pickle, numpy as np, json, collections
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, average_precision_score
DAYS='/home/sat3737/NOAA_whales/Dinov3-WhaleID/experiments/days_all'
REF='2025-01-02'; VAL=['2025-01-15','2025-02-27','2025-03-14']
NOT_WHALE={'2025-03-14':{45,47,48,49,50,51,53,54,55,56,62,63,64,65,66,67},'2025-01-02':{344,349,363,364,367,396,399,401,402,147,148},'2025-02-27':set(),'2025-01-15':set()}
def nrank(v):
    out=np.full(len(v),np.nan); f=np.isfinite(v)
    if f.any(): out[f]=rankdata(v[f])/f.sum()
    return out
def combo(cols):
    M=np.stack(cols,1); n=np.isfinite(M).sum(1); s=np.nansum(M,1)/np.maximum(n,1); s[n==0]=0.0; return s
def top_stats(y,s):
    s=np.where(np.isfinite(s),s,-1e9); order=np.argsort(-s,kind='stable'); yy=y[order]; cum=np.cumsum(yy); k=np.arange(1,len(yy)+1); prec=cum/k
    def at(p):
        ok=np.where(prec>=p)[0]; ok=ok[ok>=9]; return int(cum[ok[-1]]) if len(ok) else 0
    wrong=np.cumsum(1-yy)
    return dict(p50=float(prec[49]),p100=float(prec[99]),p200=float(prec[199]),r95=at(.95),r90=at(.90),r80=at(.80),w0=int(yy[wrong<=0].sum()),w5=int(yy[wrong<=5].sum()),auc=float(roc_auc_score(y,s)),ap=float(average_precision_score(y,s)))
def find(par,x):
    while par[x]!=x: par[x]=par[par[x]]; x=par[x]
    return x
def scores(R,rows,m):
    S=dict(R['S'])
    for nm in ('app_fin_max','app_body_max','app_best'): S[nm]=np.array([r.get(nm,np.nan) if r.get(nm) is not None else np.nan for r in rows],float)
    rk={nm:nrank(S[nm][m]) for nm in ('app_fin_max','app_body_max','app_best','win_sal_max','win_all_rm')}
    return {'student body':S['app_body_max'][m],'student best part':S['app_best'][m],'window, all windows':S['win_all_rm'][m],'window, where-to-look':S['win_sal_max'][m],
            'rank mean: student best + window':combo([rk['app_best'],rk['win_sal_max']]),'rank mean: student fin + body + window':combo([rk['app_fin_max'],rk['app_body_max'],rk['win_sal_max']]),
            'rank mean: student fin + body':combo([rk['app_fin_max'],rk['app_body_max']])}
def run(day,exclude,g=1.5):
    R=pickle.load(open(f'transfer_raw_{day}.pkl','rb')); rows=pickle.load(open(f'feats_{day}_all.pkl','rb'))
    z=json.load(open(f'{DAYS}/{day}/safe/clusters.json')); C={c['id']:c for c in z['clusters']}
    keep=np.array([a not in exclude and b not in exclude for a,b in R['pairs']])
    m=(R['cat']!='?')&(R['gap']>g)&keep; y=(R['cat'][m]=='same').astype(int)
    pair={nm:top_stats(y,v) for nm,v in scores(R,rows,m).items()}
    # merging in score order over all pairs within 15 min
    def major(c):
        e=sorted((c.get('catalog') or {}).items(),key=lambda kv:-kv[1]); return e[0][0] if e else None
    maj={k:major(c) for k,c in C.items() if k not in exclude}; ids=[k for k in maj if maj[k]]; animals=sorted(set(maj[k] for k in ids))
    m2=(R['cat']!='?')&keep; pairs=[p for p,k in zip(R['pairs'],m2) if k]
    par={k:k for k in ids}
    for a,b in pairs:
        if maj[a]==maj[b]: par[find(par,a)]=find(par,b)
    oracle=len(set(find(par,k) for k in ids)); need=len(ids)-oracle; kr={}
    for nm,v in scores(R,rows,m2).items():
        v=np.where(np.isfinite(v),v,-1e9); order=np.argsort(-v,kind='stable'); par={k:k for k in ids}; an=dict(maj); good=wrong=0; marks={}
        for q in order:
            a,b=pairs[q]; fa,fb=find(par,a),find(par,b)
            if fa==fb: continue
            if an[fa]==an[fb]: good+=1
            else:
                wrong+=1
                if wrong in (1,3,6): marks[wrong]=good
                if wrong>=6: break
            par[fa]=fb
        for w in (1,3,6): marks.setdefault(w,good)
        kr[nm]=marks
    return dict(pair=pair,npos=int(y.sum()),nneg=int((1-y).sum()),kr=kr,need=need,start=len(ids)/len(animals),oracle=oracle/len(animals),clusters=len(ids),animals=len(animals))
if __name__=='__main__':
    out={}
    for tag,use in (('as is',False),('non-whales left out',True)):
        for g in (1.5,60.0):
            res={d:run(d,NOT_WHALE[d] if use else set(),g) for d in VAL+[REF]}; out[(tag,g)]=res
            print(f"\n==== {tag}, gap > {g} s ====")
            for d in ('2025-03-14',REF):
                r=res[d]; print(f"  {d}: catalog-same pairs {r['npos']}, different {r['nneg']}; clusters with an animal {r['clusters']}, animals {r['animals']}, pieces per animal {r['start']:.2f}, merges needed {r['need']}")
                for nm,st in r['pair'].items():
                    k=r['kr'][nm]; print(f"     {nm:40s} top 50 / 100 / 200: {st['p50']:.2f} {st['p100']:.2f} {st['p200']:.2f} | at 0.95 / 0.90 / 0.80: {st['r95']:4d} {st['r90']:4d} {st['r80']:4d} | above 1st / 6th wrong pair: {st['w0']:3d} {st['w5']:3d} | merges before the 1st / 3rd / 6th wrong merge: {k[1]:3d} {k[3]:3d} {k[6]:3d} | AUC {st['auc']:.3f} AP {st['ap']:.3f}")
            print(f"  val days together: same {sum(res[d]['npos'] for d in VAL)}, different {sum(res[d]['nneg'] for d in VAL)}, merges needed {sum(res[d]['need'] for d in VAL)}")
            for nm in res[VAL[0]]['pair']:
                a=[res[d]['pair'][nm] for d in VAL]; k=[res[d]['kr'][nm] for d in VAL]
                print(f"     {nm:40s} top-100 precision (mean) {np.mean([x['p100'] for x in a]):.2f} | at 0.95 / 0.90 / 0.80 (sum): {sum(x['r95'] for x in a):4d} {sum(x['r90'] for x in a):4d} {sum(x['r80'] for x in a):4d} | merges before the 1st / 3rd / 6th wrong merge (sum): {sum(x[1] for x in k):3d} {sum(x[3] for x in k):3d} {sum(x[6] for x in k):3d} | mean AUC {np.mean([x['auc'] for x in a]):.3f} AP {np.mean([x['ap'] for x in a]):.3f}")
    pickle.dump(out,open('nowhale.pkl','wb'))
