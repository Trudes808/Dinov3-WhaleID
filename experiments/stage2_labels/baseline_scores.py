import numpy as np, collections
from common import *
from sklearn.metrics import roc_auc_score
day='2025-01-02'; z,C,P,L=load(day)
ch=np.load(f'{DAYS}/{day}/cluster_v3/channels.npz', allow_pickle=True)
keys=list(ch['keys']); ki={k:i for i,k in enumerate(keys)}
M={p:ch[f'id:pair_student_g|{p}'] for p in ('body','fin','fluke')}
print({p:(float(np.nanmin(m)),float(np.nanmax(m)),float(np.isnan(m).mean())) for p,m in M.items()})
def feats(a,b):
    ia=[ki[k] for k in C[a]['keys'] if k in ki]; ib=[ki[k] for k in C[b]['keys'] if k in ki]
    out={}
    for p,m in M.items():
        s=m[np.ix_(ia,ib)]; s=s[np.isfinite(s)]
        out[p+'_max']=float(s.max()) if s.size else np.nan
        out[p+'_mean']=float(s.mean()) if s.size else np.nan
    v=[out[p+'_max'] for p in M if np.isfinite(out[p+'_max'])]
    out['best']=max(v) if v else np.nan
    return out
sug={tuple(sorted((g['a'],g['b']))):g for g in z['suggestions']}
rows=[]
for x in L:
    a,b=x['set_a'],x['set_b']; f=feats(a,b); g=sug.get(tuple(sorted((a,b))),{})
    f.update(verdict=x['verdict'],gap=gap(C,a,b),p=g.get('p'),app=g.get('app'),score=x.get('model_score'),cat=x.get('catalog'))
    rows.append(f)
import pandas as pd
df=pd.DataFrame(rows)
def auc(pos,neg,col,mask=None):
    d=df if mask is None else df[mask]
    d=d[d.verdict.isin(pos+neg)&d[col].notna()]
    y=d.verdict.isin(pos).astype(int)
    if y.nunique()<2: return float('nan'),len(d)
    return roc_auc_score(y,d[col]),len(d)
for name,mask in (('all',None),('gap>1.5s',df.gap>1.5),('gap>60s',df.gap>60)):
    print('==',name, dict(collections.Counter((df if mask is None else df[mask]).verdict)))
    for col in ('score','app','best','fin_max','fluke_max','body_max','fin_mean','body_mean'):
        r=[auc(['same'],['different'],col,mask), auc(['same'],['no_similarity'],col,mask), auc(['same'],['different','no_similarity'],col,mask), auc(['same','likely_same'],['different'],col,mask)]
        print(f'  {col:10s} same/diff {r[0][0]:.3f} (n {r[0][1]})  same/nomark {r[1][0]:.3f}  same/(diff+nomark) {r[2][0]:.3f} (n {r[2][1]})  (same+likely)/diff {r[3][0]:.3f}')
# precision at the top: how many 'same' can be linked before the first 'different' / first non-same
m=df.gap>1.5
for col in ('app','best','fin_max'):
    d=df[m & df[col].notna()].sort_values(col,ascending=False)
    v=list(d.verdict)
    first_diff=next((i for i,t in enumerate(v) if t=='different'),len(v))
    same_before=sum(t=='same' for t in v[:first_diff])
    print(col,'across-gap: same links ranked above the first "different":',same_before,'of',sum(t=='same' for t in v), '| top-50 composition', dict(collections.Counter(v[:50])), '| top-100', dict(collections.Counter(v[:100])))
df.to_pickle('baseline_scores.pkl')
