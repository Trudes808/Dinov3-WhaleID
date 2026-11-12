import numpy as np, pickle
from sklearn.metrics import roc_auc_score, average_precision_score
import os; DS=pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'ds.pkl'),'rb'))
RECS=DS['recs']
def truth(r):
    """1 = researcher link, 0 = different animals as far as known (explicit 'different', or no mark + catalog differs + not joined), None otherwise."""
    if r['verdict']=='same': return 1
    if r['verdict']=='different': return 0
    if r['verdict']=='no_similarity' and r['cat']=='diff' and r['comp_a']!=r['comp_b']: return 0
    return None
def report(name, score, recs=RECS, subsets=(('all',0.0-1),('gap>1.5s',1.5),('gap>60s',60.0)), quiet=False):
    """score: dict (a,b)->value or array aligned with recs. Higher = more same."""
    out={}
    for sname,g in subsets:
        rows=[]
        for i,r in enumerate(recs):
            if r['gap']<=g: continue
            s=score[i] if not isinstance(score,dict) else score.get((min(r['a'],r['b']),max(r['a'],r['b'])))
            if s is None or not np.isfinite(s): s=-1e9
            rows.append((r,s))
        same=np.array([s for r,s in rows if r['verdict']=='same'])
        diff=np.array([s for r,s in rows if r['verdict']=='different'])
        nos=np.array([s for r,s in rows if r['verdict']=='no_similarity'])
        neg=np.array([s for r,s in rows if truth(r)==0])
        def auc(p,n):
            return float(roc_auc_score(np.r_[np.ones(len(p)),np.zeros(len(n))],np.r_[p,n])) if len(p) and len(n) else float('nan')
        srt=np.sort(neg)[::-1]
        d=dict(n_same=len(same),n_diff=len(diff),n_neg=len(neg),auc_diff=auc(same,diff),auc_nosim=auc(same,nos),auc_neg=auc(same,neg),
               ap_neg=float(average_precision_score(np.r_[np.ones(len(same)),np.zeros(len(neg))],np.r_[same,neg])),
               rec0=int((same>srt[0]).sum()),rec1=int((same>srt[1]).sum()),rec3=int((same>srt[3]).sum()),rec5=int((same>srt[5]).sum()))
        out[sname]=d
        if not quiet:
            print(f"{name:34s} {sname:9s} same {d['n_same']:3d} neg {d['n_neg']:3d} | AUC vs different {d['auc_diff']:.3f}  vs no-mark {d['auc_nosim']:.3f}  vs other-animal {d['auc_neg']:.3f}  AP {d['ap_neg']:.3f} | same links above the 1st/2nd/4th/6th wrong: {d['rec0']:3d} {d['rec1']:3d} {d['rec3']:3d} {d['rec5']:3d}")
    return out
