"""Component-level view of the same scores: merge cluster pairs in score order (skip pairs already joined) and count
correct merges before the 1st / 3rd / 6th wrong merge; pieces per catalog animal at the zero-wrong point."""
import pickle, numpy as np, json, collections, sys
from scipy.stats import rankdata
DAYS='/home/sat3737/NOAA_whales/Dinov3-WhaleID/experiments/days_all'
REF='2025-01-02'; VAL=['2025-01-15','2025-02-27','2025-03-14']
def nrank(v):
    out=np.full(len(v),np.nan); f=np.isfinite(v)
    if f.any(): out[f]=rankdata(v[f])/f.sum()
    return out
def combo(cols):
    M=np.stack(cols,1); n=np.isfinite(M).sum(1); s=np.nansum(M,1)/np.maximum(n,1); s[n==0]=0.0; return s
def run(day, mingap=-1.0):
    R=pickle.load(open(f'transfer_raw_{day}.pkl','rb')); rows=pickle.load(open(f'feats_{day}_all.pkl','rb'))
    z=json.load(open(f'{DAYS}/{day}/safe/clusters.json')); C={c['id']:c for c in z['clusters']}
    def major(c):
        e=sorted((c.get('catalog') or {}).items(), key=lambda kv:-kv[1]); return e[0][0] if e else None
    maj={k:major(c) for k,c in C.items()}; ids=[k for k in C if maj[k]]; animals=sorted(set(maj[k] for k in ids))
    S=dict(R['S'])
    for nm in ('app_fin_max','app_body_max','app_best'): S[nm]=np.array([r.get(nm,np.nan) if r.get(nm) is not None else np.nan for r in rows],float)
    known=(R['cat']!='?')&(R['gap']>mingap); pairs=[p for p,k in zip(R['pairs'],known) if k]
    sc={'student best part':S['app_best'][known],'student body':S['app_body_max'][known],'window, all windows':S['win_all_rm'][known],'window, where-to-look':S['win_sal_max'][known]}
    rk={k:nrank(v) for k,v in sc.items()}; rkf=nrank(S['app_fin_max'][known])
    sc['student best + window where-to-look']=combo([rk['student best part'],rk['window, where-to-look']])
    sc['student fin + body + window where-to-look']=combo([rkf,rk['student body'],rk['window, where-to-look']])
    sc['student fin + body']=combo([rkf,rk['student body']])
    out={}
    start=len(ids)/len(animals)
    # oracle within 15 min
    par={k:k for k in ids}
    def find(par,x):
        while par[x]!=x: par[x]=par[par[x]]; x=par[x]
        return x
    for a,b in pairs:
        if maj[a]==maj[b]: par[find(par,a)]=find(par,b)
    oracle=len(set(find(par,k) for k in ids))/len(animals)
    for nm,v in sc.items():
        v=np.where(np.isfinite(v),v,-1e9); order=np.argsort(-v,kind='stable'); par={k:k for k in ids}; comp_an={k:maj[k] for k in ids}
        good=0; wrong=0; marks={}; ncomp=len(ids)
        for q in order:
            a,b=pairs[q]; fa,fb=find(par,a),find(par,b)
            if fa==fb: continue
            if comp_an[fa]==comp_an[fb]: good+=1
            else:
                wrong+=1
                if wrong in (1,3,6): marks[wrong]=(good,(ncomp)/len(animals))
                if wrong>=6: break
            par[fa]=fb; ncomp-=1
        out[nm]=marks
    return dict(start=start,oracle=oracle,animals=len(animals),clusters=len(ids),res=out,needed=len(ids)-round(oracle*len(animals)))
if __name__=='__main__':
    allr={d:run(d) for d in [REF]+VAL}
    for d,r in allr.items():
        print(f"\n{d}: {r['clusters']} clusters with a catalog animal, {r['animals']} animals; pieces per animal at the start {r['start']:.2f}, oracle within 15 min {r['oracle']:.2f}; merges needed {r['needed']}")
        for nm,m in r['res'].items():
            print(f"   {nm:44s} correct merges before the 1st / 3rd / 6th wrong merge: {m.get(1,(0,0))[0]:4d} {m.get(3,(0,0))[0]:4d} {m.get(6,(0,0))[0]:4d} | pieces per animal at the first wrong merge {m.get(1,(0,r['start']))[1]:.2f}")
    print('\n== val days together ==')
    need=sum(allr[d]['needed'] for d in VAL)
    for nm in allr[VAL[0]]['res']:
        g=[sum(allr[d]['res'][nm].get(k,(0,0))[0] for d in VAL) for k in (1,3,6)]
        pcs=[allr[d]['res'][nm].get(1,(0,allr[d]['start']))[1] for d in VAL]
        print(f"   {nm:44s} correct merges before the 1st / 3rd / 6th wrong merge: {g[0]:4d} {g[1]:4d} {g[2]:4d} of {need} needed | pieces per animal at the first wrong merge (mean of days) {np.mean(pcs):.2f} (start {np.mean([allr[d]['start'] for d in VAL]):.2f}, oracle {np.mean([allr[d]['oracle'] for d in VAL]):.2f})")
    pickle.dump(allr,open('kruskal.pkl','wb'))
