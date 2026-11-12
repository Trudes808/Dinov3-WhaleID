import pickle, numpy as np, json, collections, torch
from match import *
from evalx import *
from sklearn.metrics import roc_auc_score
AUC=lambda y,s: round(float(roc_auc_score(y,s)),3)
day='2025-01-02'; d=load(day); n=len(d['keys']); kid=d['kid']; geo=d['geo']; inst=d['inst']
mu,Pm=whiten_fit(d['F'],128); Z=transform(d['F'],mu,Pm).float().cpu().numpy()
M=np.load(f'M_{day}_pcaw128.npy').astype(np.float32)
by=collections.defaultdict(list)
for i,k in enumerate(inst): by[k].append(i)
def iou_many(c,ids):
    g=geo[ids]; x0=np.maximum(c[0]-c[2]/2,g[:,0]-g[:,2]/2); x1=np.minimum(c[0]+c[2]/2,g[:,0]+g[:,2]/2); y0=np.maximum(c[1]-c[2]/2,g[:,1]-g[:,2]/2); y1=np.minimum(c[1]+c[2]/2,g[:,1]+g[:,2]/2)
    inter=np.clip(x1-x0,0,None)*np.clip(y1-y0,0,None); return inter/(c[2]**2+g[:,2]**2-inter)
box=[];best=[];rank=[]
for g in DS['regions']:
    ia,ib=kid[g['ikey_a']],kid[g['ikey_b']]
    ca=((g['box_a'][0]+g['box_a'][2])/2,(g['box_a'][1]+g['box_a'][3])/2,max(g['box_a'][2]-g['box_a'][0],g['box_a'][3]-g['box_a'][1])*1.15)
    cb=((g['box_b'][0]+g['box_b'][2])/2,(g['box_b'][1]+g['box_b'][3])/2,max(g['box_b'][2]-g['box_b'][0],g['box_b'][3]-g['box_b'][1])*1.15)
    wa=np.array(by[ia]); wb=np.array(by[ib])
    ua=wa[np.argsort(-iou_many(ca,wa))[:3]]; vb=wb[np.argsort(-iou_many(cb,wb))[:3]]
    s=float((Z[ua]@Z[vb].T).max()); box.append(s)
    S=Z[wa]@Z[wb].T; best.append(float(S.max())); rank.append(int((S>s).sum()))
box=np.array(box);best=np.array(best);rank=np.array(rank)
print('positive links: sim at the researcher boxes  pct',np.percentile(box,[10,25,50,75,90]).round(3))
print('positive links: best window pair (any scale) pct',np.percentile(best,[10,25,50,75,90]).round(3))
print('window pairs scoring above the boxed pair, per link: pct',np.percentile(rank,[10,25,50,75,90]).round(0),' boxed pair is the top pair in',np.mean(rank==0).round(3),' in the top 10:',np.mean(rank<10).round(3))
# negatives: best window pair between key instances of 'different' and other-animal pairs
neg=[];negd=[]
for r in RECS:
    if truth(r)==0 and r['gap']>1.5:
        ia,ib=kid.get(r['key_a']),kid.get(r['key_b'])
        if ia is None or ib is None: continue
        S=Z[np.array(by[ia])]@Z[np.array(by[ib])].T; neg.append(float(S.max()))
        if r['verdict']=='different': negd.append(float(S.max()))
neg=np.array(neg); negd=np.array(negd)
print('other-animal pairs: best window pair pct',np.percentile(neg,[10,25,50,75,90]).round(3),' (different only)',np.percentile(negd,[10,25,50,75,90]).round(3))
gapok=np.array([RECS[g['rec']]['gap']>1.5 for g in DS['regions']])
print('AUC boxed-pair sim (positives, gap>1.5s) vs best window pair of other-animal pairs:',AUC(np.r_[np.ones(gapok.sum()),np.zeros(len(neg))],np.r_[box[gapok],neg]),
      ' vs different only:',AUC(np.r_[np.ones(gapok.sum()),np.zeros(len(negd))],np.r_[box[gapok],negd]))
print('AUC best-pair (positives) vs best-pair (other-animal):',AUC(np.r_[np.ones(gapok.sum()),np.zeros(len(neg))],np.r_[best[gapok],neg]),' vs different only:',AUC(np.r_[np.ones(gapok.sum()),np.zeros(len(negd))],np.r_[best[gapok],negd]))
