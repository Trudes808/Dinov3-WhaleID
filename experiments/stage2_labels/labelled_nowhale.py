"""Labelled-day table (2025-01-02, grouped CV) again without the pairs that touch a tracked non-whale cluster."""
import pickle, numpy as np
from scipy.stats import rankdata
from linker_cv import *
NW={344,349,363,364,367,396,399,401,402,147,148}
keys=[key(r) for r in RECS]
mil=pickle.load(open('mil_scores_128.pkl','rb')); lc=pickle.load(open('linker_cv.pkl','rb')); cb=pickle.load(open('combine.pkl','rb'))
assert lc['keys']==keys and cb['keys']==keys
base=lambda n: np.array([F[k].get(n,np.nan) if F[k].get(n) is not None else np.nan for k in keys])
S={'student fin max':base('app_fin_max'),'student body max':base('app_body_max'),'student best part':base('app_best'),
   'stage-1 model P':np.array([r['model_score'] if r['model_score'] is not None else np.nan for r in RECS],float),
   'gbt appearance, researcher':lc['pred']['gbt appearance, researcher'],'gbt all, researcher':lc['pred']['gbt all, researcher'],
   'window all row-max':np.asarray(mil['cos_all'][2],float),'window top-10 where-to-look row-max':np.asarray(mil['sal_top10'][2],float),'window H row-max':np.asarray(mil['H']['rm'],float)}
def rk(v):
    v=np.where(np.isfinite(v),v,np.nanmin(v[np.isfinite(v)])); return rankdata(v)/len(v)
for tag,keep in (('ALL LABELLED PAIRS',np.ones(len(RECS),bool)),('WITHOUT PAIRS TOUCHING A NON-WHALE CLUSTER',np.array([r['a'] not in NW and r['b'] not in NW for r in RECS]))):
    recs=[r for r,k in zip(RECS,keep) if k]; print(f'\n=== {tag}: {len(recs)} records ===')
    T={n:v[keep] for n,v in S.items()}
    T['rank mean: fin + window top-10']=rk(T['student fin max'])+rk(T['window top-10 where-to-look row-max'])
    T['rank mean: best + window top-10']=rk(T['student best part'])+rk(T['window top-10 where-to-look row-max'])
    T['rank mean: fin + body + window top-10']=rk(T['student fin max'])+rk(T['student body max'])+rk(T['window top-10 where-to-look row-max'])
    for n in cb['pred']: T[n]=np.asarray(cb['pred'][n])[keep]
    for n,v in T.items(): report(n[:34],v,recs=recs,subsets=(('gap>1.5s',1.5),('gap>60s',60.0)))
