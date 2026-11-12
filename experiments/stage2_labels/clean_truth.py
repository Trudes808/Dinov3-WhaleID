"""How much of the top-of-ranking 'wrong' on val days is catalog ambiguity? Re-score with clusters whose catalog is
unambiguous (one animal only among its photos) and count photos that carry several animals."""
import pickle, numpy as np, json, collections
from scipy.stats import rankdata
DAYS='/home/sat3737/NOAA_whales/Dinov3-WhaleID/experiments/days_all'
REF='2025-01-02'; VAL=['2025-01-15','2025-02-27','2025-03-14']
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
    return dict(p50=float(prec[min(49,len(prec)-1)]),p100=float(prec[min(99,len(prec)-1)]),r95=at(.95),r90=at(.90),r80=at(.80),w0=int(yy[wrong<=0].sum()),w5=int(yy[wrong<=5].sum()))
for day in VAL+[REF]:
    R=pickle.load(open(f'transfer_raw_{day}.pkl','rb')); rows=pickle.load(open(f'feats_{day}_all.pkl','rb'))
    z=json.load(open(f'{DAYS}/{day}/safe/clusters.json')); C={c['id']:c for c in z['clusters']}
    def share(c):
        e=sorted((c.get('catalog') or {}).values(),reverse=True); return (e[0]/sum(e), len(e), sum(e)) if e else (0,0,0)
    sh={k:share(c) for k,c in C.items()}
    ncl=sum(1 for k in C if sh[k][1]>0); nmix=sum(1 for k in C if sh[k][1]>1)
    multi=sum(1 for k in C if (C[k].get('n_multi') or 0)>0)
    print(f"\n{day}: clusters with a catalog animal {ncl}; with more than one animal in their catalog counts {nmix}; clusters with multi-instance frames {multi}")
    S=dict(R['S'])
    for nm in ('app_fin_max','app_body_max','app_best'): S[nm]=np.array([r.get(nm,np.nan) if r.get(nm) is not None else np.nan for r in rows],float)
    base=(R['cat']!='?')&(R['gap']>1.5)
    pure=np.array([sh[a][1]==1 and sh[b][1]==1 for a,b in R['pairs']])
    nom=np.array([(C[a].get('n_multi') or 0)==0 and (C[b].get('n_multi') or 0)==0 for a,b in R['pairs']])
    for tag,m in (('all catalog-known pairs',base),('both clusters carry one catalog animal only',base&pure),('... and no multi-instance frames',base&pure&nom)):
        y=(R['cat'][m]=='same').astype(int)
        if y.sum()<5 or (1-y).sum()<5: print('  ',tag,'too few'); continue
        rk={nm:nrank(S[nm][m]) for nm in ('app_fin_max','app_body_max','app_best','win_sal_max')}
        sc={'student best part':S['app_best'][m],'window where-to-look':S['win_sal_max'][m],'rank mean best + window':combo([rk['app_best'],rk['win_sal_max']]),'rank mean fin + body + window':combo([rk['app_fin_max'],rk['app_body_max'],rk['win_sal_max']])}
        print(f"   {tag}: same {int(y.sum())}, different {int((1-y).sum())}")
        for nm,v in sc.items():
            st=top_stats(y,v); print(f"      {nm:30s} top-50 / top-100 precision {st['p50']:.2f} {st['p100']:.2f} | same pairs at precision 0.95 / 0.90 / 0.80: {st['r95']:4d} {st['r90']:4d} {st['r80']:4d} | above the 1st / 6th wrong: {st['w0']:3d} {st['w5']:3d}")
