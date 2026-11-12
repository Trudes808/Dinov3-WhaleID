"""One-sided mark search: the researcher boxes a mark on ONE photo; rank the other clusters of the interaction by the
best match of that box among their windows. Truth = the cluster the researcher linked it to (and its component)."""
import pickle, numpy as np, json, collections, torch
from match import *
from evalx import *
day='2025-01-02'
ROOT=f'/home/sat3737/NOAA_whales/Dinov3-WhaleID/experiments/days_all/{day}'
d=load(day); kid=d['kid']; inst=d['inst']; size=d['size']; geo=d['geo']; n=len(d['keys']); W=d['W']
mu,Pm=whiten_fit(d['F'],128); Z=transform(d['F'],mu,Pm).float()
R=pickle.load(open('regions_feat.pkl','rb')); meta=R['meta']
Q=transform(R['cls11'],mu,Pm).float()
qidx={(int(m[0]),m[1],float(m[2])):i for i,m in enumerate(meta)}
z=json.load(open(f'{ROOT}/safe/clusters.json')); C={c['id']:c for c in z['clusters']}
icl={}
for c in z['clusters']:
    for k in c['keys']:
        if k in kid: icl[kid[k]]=c['id']
wcl=torch.tensor(np.array([icl.get(i,-1) for i in inst])).cuda()
stems=sorted(set(w[1] for w in W)); sid={s:i for i,s in enumerate(stems)}; wstem=torch.tensor(np.array([sid[w[1]] for w in W])).cuda()
wside=torch.tensor(geo[:,2]).cuda()
comp=DS['comp']; ids=sorted(C); cpos={c:i for i,c in enumerate(ids)}
def gap(a,b):
    A,B=C[a],C[b]; return max(0.0,max(A['start'],B['start'])-min(A['end'],B['end']))
rows=pickle.load(open(f'feats_{day}_all.pkl','rb')); F={(r['a'],r['b']):r for r in rows}
wcl_idx=torch.tensor(np.array([cpos.get(int(c),-1) for c in wcl.cpu().numpy()])).cuda()
def cluster_max(sim, ok):
    """max of sim over the windows of each cluster (ok = allowed windows)."""
    out=torch.full((len(ids),),-9.0,device='cuda'); m=ok&(wcl_idx>=0)
    out.scatter_reduce_(0,wcl_idx[m],sim[m],reduce='amax',include_self=True); return out.cpu().numpy()
res=collections.defaultdict(list)
for gi,g in enumerate(DS['regions']):
    r=RECS[g['rec']]
    for s,o in (('a','b'),('b','a')):
        qc=r[s]; tc=r[o]
        cand=[c for c in ids if c!=qc and gap(qc,c)<=900.0]
        if tc not in cand: continue
        x0,y0,x1,y1=g['box_'+s]; side=max(x1-x0,y1-y0)*1.15
        stem=g['key_'+s].split('~')[0]
        notself=wstem!=sid.get(stem,-1)
        q1=Q[qidx[(gi,s,1.15)]]; q2=Q[qidx[(gi,s,2.5)]]
        s1=cluster_max(Z@q1, notself&(wside>=side*0.6)&(wside<=side*1.8))
        s2=cluster_max(Z@q2, notself&(wside>=side*2.5/1.15*0.6)&(wside<=side*2.5/1.15*1.8))
        sc={'box':s1,'box + context':0.5*(s1+np.where(s2>-5,s2,s1))}
        # baselines: student scores of the cluster pair
        for nm,fn in (('student fin max','app_fin_max'),('student best part','app_best')):
            v=np.full(len(ids),-9.0)
            for c in cand:
                f=F.get((min(qc,c),max(qc,c)))
                if f is not None and f.get(fn) is not None: v[cpos[c]]=f[fn]
            sc[nm]=v
        ci=np.array([cpos[c] for c in cand]); same_comp=np.array([comp[c]==comp[tc] for c in cand])
        for nm,v in sc.items():
            vv=v[ci]; order=np.argsort(-vv,kind='stable'); rank_t=int(np.where(ci[order]==cpos[tc])[0][0]); rank_c=int(np.argmax(same_comp[order]))
            res[nm].append((rank_t,rank_c,len(cand),r['gap']))
print('queries',len(res['box']),' candidate clusters per query: median',int(np.median([x[2] for x in res['box']])))
for sub,lo in (('all links',-1),('links across a gap > 1.5 s',1.5),('links across a gap > 60 s',60)):
    print(sub)
    for nm,v in res.items():
        v=np.array([x for x in v if x[3]>lo]); 
        print(f'   {nm:20s} n {len(v):4d} | linked cluster ranked 1st {np.mean(v[:,0]==0):.3f}  top 5 {np.mean(v[:,0]<5):.3f}  top 10 {np.mean(v[:,0]<10):.3f}  median rank {int(np.median(v[:,0]))+1:3d} | any cluster of the same whale 1st {np.mean(v[:,1]==0):.3f}  top 5 {np.mean(v[:,1]<5):.3f}')
