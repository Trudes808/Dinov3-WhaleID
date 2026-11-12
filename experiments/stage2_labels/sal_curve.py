"""Learning curve of the 'where does the researcher look' model: fraction of the training animals used."""
import pickle, numpy as np, json, collections, torch, sys
from match import *
from evalx import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
day='2025-01-02'; ROOT=f'/home/sat3737/NOAA_whales/Dinov3-WhaleID/experiments/days_all/{day}'
d=load(day); n=len(d['keys']); kid=d['kid']; geo=d['geo']; inst=d['inst']; N=len(inst)
mu,Pm=whiten_fit(d['F'],128); Zc=transform(d['F'],mu,Pm).float().cpu().numpy()
z=json.load(open(f'{ROOT}/safe/clusters.json')); cfold=DS['cfold']; K=DS['K']; grp=DS['grp']
icl={kid[k]:c['id'] for c in z['clusters'] for k in c['keys'] if k in kid}
ifold=np.array([cfold.get(icl.get(i,-1),-1) for i in range(n)]); wfold=ifold[inst]
igrp=[grp.get(icl.get(i,-1)) for i in range(n)]
by=collections.defaultdict(list)
for i,k in enumerate(inst): by[k].append(i)
by={k:np.array(v) for k,v in by.items()}
def sq(b): return ((b[0]+b[2])/2,(b[1]+b[3])/2,max(b[2]-b[0],b[3]-b[1])*1.15)
def iou_many(c,ids):
    g=geo[ids]; x0=np.maximum(c[0]-c[2]/2,g[:,0]-g[:,2]/2); x1=np.minimum(c[0]+c[2]/2,g[:,0]+g[:,2]/2); y0=np.maximum(c[1]-c[2]/2,g[:,1]-g[:,2]/2); y1=np.minimum(c[1]+c[2]/2,g[:,1]+g[:,2]/2)
    inter=np.clip(x1-x0,0,None)*np.clip(y1-y0,0,None); return inter/(c[2]**2+g[:,2]**2-inter)
boxes=collections.defaultdict(list)
for g in DS['regions']:
    for s in 'ab':
        i=kid.get(g['ikey_'+s])
        if i is not None: boxes[i].append(sq(g['box_'+s]))
lab=np.full(N,-1,np.int8)
for i,bl in boxes.items():
    w=by[i]; m=np.max(np.stack([iou_many(b,w) for b in bl]),0); lab[w[m>=0.3]]=1; lab[w[m<0.05]]=0
groups=sorted(set(str(igrp[i]) for i in boxes)); print('boxed instances',len(boxes),'animal groups with boxes',len(groups))
wgrp=np.array([str(igrp[i]) for i in inst])
def run(frac,seed):
    rng=np.random.default_rng(seed); out=np.zeros(N,np.float32); used=[]
    for f in range(K):
        gtr=sorted(set(wgrp[(wfold!=f)&(lab>=0)])); keep=set(rng.choice(gtr,max(1,int(round(frac*len(gtr)))),replace=False)); used.append(len(keep))
        tr=(wfold!=f)&(lab>=0)&np.isin(wgrp,list(keep))
        sc=StandardScaler().fit(Zc[tr]); clf=LogisticRegression(C=0.1,class_weight='balanced',max_iter=400).fit(sc.transform(Zc[tr]),lab[tr])
        te=wfold==f; out[te]=clf.decision_function(sc.transform(Zc[te]))
    hits={1:[],5:[],10:[]}
    for i,bl in boxes.items():
        if ifold[i]<0: continue
        w=by[i]; order=w[np.argsort(-out[w])]
        for b in bl:
            pos=iou_many(b,order)>=0.3
            if pos.any():
                first=int(np.argmax(pos))
                for k in hits: hits[k].append(first<k)
    return np.mean(used),[np.mean(hits[k]) for k in (1,5,10)]
for frac in (0.1,0.25,0.5,0.75,1.0):
    r=[run(frac,s) for s in range(3 if frac<1 else 1)]
    print(f'fraction of training animals {frac:.2f} (~{np.mean([x[0] for x in r]):.0f} animal groups): box in the top 1 / 5 / 10 = '+' / '.join(f'{np.mean([x[1][j] for x in r]):.3f}' for j in range(3)))
