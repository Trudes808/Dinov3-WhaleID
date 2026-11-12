import sys; sys.argv=['x']
from linker_cv import *
import linker_cv
keys=[key(r) for r in RECS]
def tgt(r,mode):
    if mode=='soft': return {'same':1,'likely_same':1,'different':0,'no_similarity':0}.get(r['verdict'])
    if mode=='seqonly': return {'likely_same':1,'no_similarity':0,'different':0}.get(r['verdict'])
    return {'same':1,'different':0,'no_similarity':0}.get(r['verdict'])
linker_cv.target=tgt
from sklearn.metrics import roc_auc_score
v=np.array([r['verdict'] for r in RECS]); g=np.array([r['gap'] for r in RECS]); cat=np.array([r['cat'] for r in RECS])
def auc(p,a,b,m=None):
    m=np.ones(len(v),bool) if m is None else m
    A=p[m&np.isin(v,a)]; B=p[m&np.isin(v,b)]
    return round(float(roc_auc_score(np.r_[np.ones(len(A)),np.zeros(len(B))],np.r_[A,B])),3),len(A),len(B)
base=lambda n: np.array([F[k].get(n,np.nan) if F[k].get(n) is not None else np.nan for k in keys])
print('likely_same vs no_similarity (both without a mark), pairs across a gap > 1.5 s:')
m=g>1.5
print('  time gap alone',auc(-np.nan_to_num(base('log_gap')),['likely_same'],['no_similarity','different'],m))
for name,names,mode in (('sequence features, trained on likely_same vs no mark',SEQ,'seqonly'),('all features, trained on likely_same vs no mark',ALL,'seqonly'),('sequence features, trained on same+likely vs rest',SEQ,'soft')):
    ps=np.mean([cv(names,mode,seed=s)(keys) for s in range(3)],0)
    print(' ',name,auc(ps,['likely_same'],['no_similarity','different'],m),' | catalog check inside likely_same: AUC catalog-same vs catalog-diff',
          round(float(roc_auc_score((cat[(v=='likely_same')&(cat!='?')]=='same').astype(int),ps[(v=='likely_same')&(cat!='?')])),3))
print('catalog truth of likely_same:',collections.Counter(cat[v=='likely_same']),' of no_similarity:',collections.Counter(cat[v=='no_similarity']),' of same:',collections.Counter(cat[v=='same']),' of different:',collections.Counter(cat[v=='different']))
