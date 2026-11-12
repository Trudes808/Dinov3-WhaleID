import pickle, numpy as np, sys
from linker_cv import *
res=pickle.load(open('mil_scores_128.pkl','rb'))
keys=[key(r) for r in RECS]
new={'win_all_key':res['cos_all'][0],'win_all_rm':res['cos_all'][2],'win_sal10_key':res['sal_top10'][0],'win_sal10_rm':res['sal_top10'][2],'win_sal5_key':res['sal_top5'][0],
     'win_H_key':res['H']['key'],'win_H_rm':res['H']['rm']}
for n,v in new.items():
    for k,x in zip(keys,v): F[k][n]=float(x) if np.isfinite(x) else None
WIN=list(new); WINU=['win_all_key','win_all_rm']; WINS=['win_sal10_key','win_sal10_rm','win_sal5_key']
SUB=(('all',-1.0),('gap>1.5s',1.5),('gap>60s',60.0))
base=lambda n: np.array([F[k].get(n,np.nan) if F[k].get(n) is not None else np.nan for k in keys])
print('-- references --')
for n in ('app_fin_max','app_body_max','app_best'): report('student '+n, base(n),subsets=SUB)
report('window top-10 salient row-max',base('win_sal10_rm'),subsets=SUB)
# simple rank-average of student fin and window scores (no training)
from scipy.stats import rankdata
def rk(v): 
    v=np.where(np.isfinite(v),v,np.nanmin(v[np.isfinite(v)])); return rankdata(v)/len(v)
report('rank mean: student fin max + window sal10', rk(base('app_fin_max'))+rk(base('win_sal10_rm')),subsets=SUB)
report('rank mean: student best + window sal10', rk(base('app_best'))+rk(base('win_sal10_rm')),subsets=SUB)
report('rank mean: fin max + body max + window sal10', rk(base('app_fin_max'))+rk(base('app_body_max'))+rk(base('win_sal10_rm')),subsets=SUB)
print('-- GBT, grouped CV, researcher labels (3 seeds) --')
out={}
for name,names in (('gbt student appearance',APP),('gbt student + window (unsupervised)',APP+WINU),('gbt student + window salient',APP+WINU+WINS),('gbt student + all window scores',APP+WIN),('gbt window scores only',WIN),('gbt all features + all window scores',ALL+WIN)):
    ps=np.mean([cv(names,'researcher',seed=s)(keys) for s in range(3)],0); out[name]=ps
    report(name,ps,subsets=SUB)
pickle.dump(dict(pred=out,keys=keys),open('combine.pkl','wb'))
