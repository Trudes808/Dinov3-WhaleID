"""Build the labelled dataset for the stage-2 experiments (read-only on project data)."""
import pickle, collections, json, sys
import numpy as np
from common import *
day='2025-01-02'
z,C,P,L=load(day)
W,H=2560,1709

class UF:
    def __init__(s): s.p={}
    def f(s,x):
        s.p.setdefault(x,x)
        while s.p[x]!=x:
            s.p[x]=s.p[s.p[x]]; x=s.p[x]
        return x
    def u(s,a,b): s.p[s.f(a)]=s.f(b)

# researcher components (same links only)
uf=UF()
for c in C: uf.f(c)
for r in L:
    if r['verdict']=='same': uf.u(r['set_a'],r['set_b'])
comp={c:uf.f(c) for c in C}
# CV groups: the researcher's component, labelled by the majority catalog animal of the whole component (photo votes).
# Components with the same majority animal share a group; unlabelled components stay alone.
votes=collections.defaultdict(collections.Counter)
for c,cl in C.items():
    for a,n in (cl.get('catalog') or {}).items(): votes[comp[c]][a]+=n
cmaj={k:v.most_common(1)[0][0] for k,v in votes.items()}
grp={c:(('a',cmaj[comp[c]]) if comp[c] in cmaj else ('c',comp[c])) for c in C}
gsz=collections.Counter(grp.values())
print('clusters',len(C),'researcher comps',len(set(comp.values())),'cv groups',len(gsz),'largest',gsz.most_common(8))
mixed=sum(1 for k,v in votes.items() if len(v)>1); print('components with >1 catalog animal',mixed)

def attributed(key, box):
    stem=key.split('~')[0]; cx,cy=(box[0]+box[2])/2,(box[1]+box[3])/2
    own=inst(P,key); x0,y0,x1,y1=own['bbox']
    if x0<=cx<=x1 and y0<=cy<=y1: return key
    for j in P[stem]['instances']:
        x0,y0,x1,y1=j['bbox']
        if x0<=cx<=x1 and y0<=cy<=y1: return j['key']
    return key

recs=[]; regions=[]
for i,r in enumerate(L):
    a,b=r['set_a'],r['set_b']
    v=r['verdict']
    ca,cb=major(C[a]),major(C[b])
    cat='?' if (ca is None or cb is None) else ('same' if ca==cb else 'diff')
    rec=dict(i=i,a=a,b=b,verdict=v,gap=gap(C,a,b),key_a=r['key_a'],key_b=r['key_b'],cue=r['cue'],cat=cat,cat_a=ca,cat_b=cb,
             comp_a=comp[a],comp_b=comp[b],grp_a=grp[a],grp_b=grp[b],model_score=r.get('model_score'),nreg=len(r.get('regions') or []))
    recs.append(rec)
    if v!='same': continue
    for j,g in enumerate(r.get('regions') or []):
        d=dict(rec=i,j=j,cue=r['cue'],comp=comp[a],grp=grp[a])
        for s in 'ab':
            box=box_px(g[s],W,H); key=r['key_'+s]
            d['box_'+s]=box; d['key_'+s]=key; d['ikey_'+s]=attributed(key,box)
        regions.append(d)
print('records',len(recs),'region pairs',len(regions), 'moved to other instance', sum(d['key_a']!=d['ikey_a'] for d in regions)+sum(d['key_b']!=d['ikey_b'] for d in regions))

# folds over groups, balanced by number of same links
K=5
load_=collections.Counter()
for r in recs:
    if r['verdict']=='same': load_[r['grp_a']]+=1
allg=sorted(set(grp.values()), key=lambda g:(-load_[g],str(g)))
fold={}; tot=[0]*K; cnt=[0]*K
for g in allg:
    k=int(np.argmin([t*1000+c for t,c in zip(tot,cnt)])); fold[g]=k; tot[k]+=load_[g]; cnt[k]+=1
print('same links per fold',tot,'groups per fold',cnt)
cfold={c:fold[grp[c]] for c in C}
pickle.dump(dict(day=day,recs=recs,regions=regions,comp=comp,grp=grp,cfold=cfold,K=K),open('ds.pkl','wb'))

# quick stats: scale ratio between the two boxes vs ppm ratio and bbox ratio
from PIL import Image
sr=[];pr=[];br=[];mf=[]
for d in regions:
    sa=max(d['box_a'][2]-d['box_a'][0],d['box_a'][3]-d['box_a'][1]); sb=max(d['box_b'][2]-d['box_b'][0],d['box_b'][3]-d['box_b'][1])
    pa=P[d['key_a'].split('~')[0]]; pb=P[d['key_b'].split('~')[0]]
    ia=inst(P,d['ikey_a']); ib=inst(P,d['ikey_b'])
    la=max(ia['bbox'][2]-ia['bbox'][0],ia['bbox'][3]-ia['bbox'][1]); lb=max(ib['bbox'][2]-ib['bbox'][0],ib['bbox'][3]-ib['bbox'][1])
    sr.append(np.log2(sa/sb)); br.append(np.log2(la/lb))
    pr.append(np.log2(pa['ppm']/pb['ppm']) if pa.get('ppm') and pb.get('ppm') else np.nan)
sr=np.array(sr);pr=np.array(pr);br=np.array(br)
ok=np.isfinite(pr)
print('log2 box-size ratio a/b: pct',np.percentile(np.abs(sr),[50,75,90,95]).round(2))
print('corr with log2 ppm ratio',np.corrcoef(sr[ok],pr[ok])[0,1].round(3),'n',ok.sum(),' corr with log2 bbox ratio',np.corrcoef(sr,br)[0,1].round(3))
print('residual |log2| after ppm norm',np.percentile(np.abs(sr[ok]-pr[ok]),[50,75,90,95]).round(2),' after bbox norm',np.percentile(np.abs(sr-br),[50,75,90,95]).round(2))
# mask fraction inside region boxes
for d in regions[:288]:
    for s in 'ab':
        stem=d['key_'+s].split('~')[0]
        m=np.asarray(Image.open(f'{DAYS}/{day}/canon/{stem}_mask.png'))
        x0,y0,x1,y1=[int(round(v)) for v in d['box_'+s]]
        sub=m[max(0,y0):y1,max(0,x0):x1]; mf.append(float(sub.mean()) if sub.size else np.nan)
mf=np.array(mf); print('whale-mask fraction inside region boxes pct',np.nanpercentile(mf,[5,10,25,50,75,95]).round(2),' zero:',np.mean(mf==0).round(3))
