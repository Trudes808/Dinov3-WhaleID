import json, numpy as np
from regions_head import *
P=json.load(open('/home/sat3737/NOAA_whales/Dinov3-WhaleID/experiments/days_all/2025-01-02/safe/photos.json'))
ta=np.array([P[d['key_a'].split('~')[0]]['time'] for d in R]); tb=np.array([P[d['key_b'].split('~')[0]]['time'] for d in R])
cam_a=np.array([P[d['key_a'].split('~')[0]]['camera'] for d in R]); cam_b=np.array([P[d['key_b'].split('~')[0]]['camera'] for d in R])
dt=np.abs(ta[:,None]-tb[None,:]); samecam=cam_a[:,None]==cam_b[None,:]
print('positive pairs: dt median',np.median(np.diag(dt)).round(1),'p90',np.percentile(np.diag(dt),90).round(0),'same camera',np.mean(np.diag(samecam)).round(3))
def ev2(name,S):
    pos=np.diag(S); out=[]
    for lab,m in (('all other-animal',neg),('same cue',neg&samecue),('within 5 min + same camera',neg&(dt<300)&samecam),('within 2 min + same camera + same cue',neg&(dt<120)&samecam&samecue)):
        # positives restricted to the same time window for the time-matched rows
        if 'min' in lab:
            lim=300 if '5 min' in lab else 120; pp=pos[(np.diag(dt)<lim)&np.diag(samecam)]
        else: pp=pos
        out.append((lab,roc_auc_score(np.r_[np.ones(len(pp)),np.zeros(m.sum())],np.r_[pp,S[m]]),len(pp),int(m.sum())))
    print(f'{name:40s} '+' | '.join(f'{l}: {a:.3f} ({p}/{q})' for l,a,p,q in out),flush=True)
def cvS(A,B,maker):
    S=np.zeros((n,n),np.float32)
    for f in range(K):
        tr=np.where(fold!=f)[0]; te=np.where(fold==f)[0]
        fn=maker(A,B,tr); M=fn(A)@fn(B).T
        S[te]=M[te]; S[:,te]=np.where(np.isin(np.arange(n),te)[:,None],S[:,te],M[:,te])
    return S
for kind in ('cls11@1.15','cls11@1.15+2.5'):
    A,B=[np.asarray(x,np.float32) for x in feats(kind)]
    ev2(kind+' raw',A@B.T)
    ev2(kind+' wccn128',cvS(A,B,lambda A,B,tr:wccn(A,B,tr,128,0.3)))
    ev2(kind+' nce',cvS(A,B,lambda A,B,tr:head_nce(A,B,tr,dim=128,drop=0.6)))
