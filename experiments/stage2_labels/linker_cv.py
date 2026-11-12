import sys, pickle, numpy as np, collections
from sklearn.ensemble import HistGradientBoostingClassifier
from evalx import *
day='2025-01-02'
rows=pickle.load(open(f'feats_{day}_all.pkl','rb')); F={(r['a'],r['b']):r for r in rows}
cfold=DS['cfold']; K=DS['K']
SKIP={'a','b','cat','cat_a','cat_b','suggested'}
ALL=sorted(k for k in rows[0].keys()|rows[5].keys()|set().union(*[set(r) for r in rows[:2000]]) if k not in SKIP)
APP=[k for k in ALL if k.startswith('app_')]
SEQ=[k for k in ALL if not k.startswith('app_')]
def mat(prs,names):
    X=np.full((len(prs),len(names)),np.nan)
    for i,p in enumerate(prs):
        f=F[p]
        for j,n in enumerate(names):
            v=f.get(n)
            if v is not None: X[i,j]=v
    return X
def key(r): return (min(r['a'],r['b']),max(r['a'],r['b']))
def target(r,mode):
    if mode=='researcher':
        return {'same':1,'different':0,'no_similarity':0}.get(r['verdict'])
    if mode=='catalog':
        return {'same':1,'diff':0}.get(r['cat'])
def cv(names, mode='researcher', train_recs=None, seed=0, leaf=10, iters=300, depth=3, lr=0.05):
    """K models; a pair is predicted by the models that never saw either... (models of the folds of its two clusters, averaged)."""
    train_recs=train_recs if train_recs is not None else RECS
    models=[]
    for f in range(K):
        tr=[r for r in train_recs if cfold[r['a']]!=f and cfold[r['b']]!=f and target(r,mode) is not None]
        X=mat([key(r) for r in tr],names); y=np.array([target(r,mode) for r in tr])
        clf=HistGradientBoostingClassifier(max_depth=depth,learning_rate=lr,max_iter=iters,l2_regularization=1.0,min_samples_leaf=leaf,early_stopping=False,random_state=seed)
        clf.fit(X,y); models.append(clf)
    def predict(prs):
        X=mat(prs,names); P=np.stack([m.predict_proba(X)[:,1] for m in models],1)
        out=np.zeros(len(prs))
        for i,(a,b) in enumerate(prs):
            fs={cfold[a],cfold[b]}; out[i]=np.mean([P[i,f] for f in fs])
        return out
    return predict
if __name__=='__main__':
    keys=[key(r) for r in RECS]
    print('labelled',len(RECS),'features',len(ALL))
    res={}
    base=lambda n: np.array([F[k].get(n,np.nan) if F[k].get(n) is not None else np.nan for k in keys])
    for n in ('app_fin_max','app_fin_mean','app_body_max','app_fluke_max','app_best'):
        res[n]=report('student '+n, base(n))
    res['model_score']=report('stage-1 model P (record)', np.array([r['model_score'] if r['model_score'] is not None else np.nan for r in RECS]))
    res['neg_gap']=report('time gap only', -base('log_gap'))
    out={}
    for name,names,mode in (('gbt appearance, researcher',APP,'researcher'),('gbt sequence only, researcher',SEQ,'researcher'),('gbt all, researcher',ALL,'researcher'),('gbt all, catalog labels',ALL,'catalog'),('gbt appearance, catalog labels',APP,'catalog')):
        ps=np.mean([cv(names,mode,seed=s)(keys) for s in range(3)],0)
        res[name]=report(name,ps); out[name]=ps
    pickle.dump(dict(res=res,pred=out,keys=keys),open('linker_cv.pkl','wb'))
