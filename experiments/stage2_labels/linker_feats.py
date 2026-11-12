"""Cluster-pair features from what already exists (sequence, colour, student embeddings). Read-only on project data.
usage: linker_feats.py DAY [all]   -> feats_<day>.pkl   (pairs = suggestions + labelled pairs; 'all' adds every pair within 15 min)
"""
import sys, pickle, math, json, itertools
import numpy as np
from pathlib import Path
sys.path.insert(0,'/home/sat3737/NOAA_whales/Dinov3-WhaleID')
from model_fine_tuning.cluster.safe_clusters import load_embeddings, pair_features
from common import *
PARTS=('body','fin','fluke')

def build(day, every=False, window=900.0):
    z=json.load(open(f'{DAYS}/{day}/safe/clusters.json')); C={c['id']:c for c in z['clusters']}
    P=json.load(open(f'{DAYS}/{day}/safe/photos.json'))
    emb=load_embeddings(Path(f'{DAYS}/{day}'))
    I={it['key']:it for p in P.values() for it in p['instances']}
    T={k:P[k.split('~')[0]]['time'] for k in I}
    pairs=set()
    for s in z['suggestions']: pairs.add((min(s['a'],s['b']),max(s['a'],s['b'])))
    sugg=set(pairs)
    mk=f'{ROOT}/Whale-Review-Web/data/{day}/pair_marks.json'
    if os.path.exists(mk):
        for r in json.load(open(mk))['labels']:
            if r.get('level')=='cluster': pairs.add((min(r['set_a'],r['set_b']),max(r['set_a'],r['set_b'])))
    if every:
        ids=sorted(C)
        for a,b in itertools.combinations(ids,2):
            if gap(C,a,b)<=window: pairs.add((a,b))
    # per-cluster matrices
    Em={part:{c:(np.asarray([emb[part][k] for k in cl['keys'] if k in emb.get(part,{})]) if any(k in emb.get(part,{}) for k in cl['keys']) else None) for c,cl in C.items()} for part in PARTS}
    lab={c:[I[k]['lab_whale'] for k in cl['keys'] if I[k].get('lab_whale')] for c,cl in C.items()}
    rows=[]
    for a,b in sorted(pairs):
        A,B=C[a],C[b]
        if A['start']>B['start']: A,B=B,A
        g=max(0.0,B['start']-A['end'])
        f=dict(a=a,b=b,gap=g,log_gap=math.log1p(g),same_cam=float(A['camera']==B['camera']),n_a=len(A['keys']),n_b=len(B['keys']),
               n_min=min(len(A['keys']),len(B['keys'])),span_a=A['end']-A['start'],span_b=B['end']-B['start'],overlap=float(B['start']<A['end']),
               suggested=float((a,b) in sugg))
        # the two instances closest in time across the pair
        ka=max(A['keys'],key=lambda k:T[k]); kb=min(B['keys'],key=lambda k:T[k])
        pf=pair_features(P[ka.split('~')[0]],I[ka],P[kb.split('~')[0]],I[kb],1,{})
        for k in ("d_pos","d_size","log_area_min","d_focal","d_ev","d_range","d_rot","d_nose","side_flip","d_lab_whale","d_lab_water","d_asym","d_contrast","d_lum_std","n_inst_max","parts_shared","parts_union"):
            if pf.get(k) is not None: f[k]=pf[k]
        best=None;n=0
        for part in PARTS:
            Ea,Eb=Em[part][A['id']],Em[part][B['id']]
            if Ea is not None and Eb is not None:
                S=Ea@Eb.T; v=np.sort(S.ravel())[::-1]
                f[f'app_{part}_max']=float(v[0]); f[f'app_{part}_mean']=float(v.mean()); f[f'app_{part}_top3']=float(v[:3].mean()); f[f'app_{part}_med']=float(np.median(v)); f[f'app_{part}_n']=float(len(v))
                f[f'app_{part}_rowmax']=float(0.5*(S.max(1).mean()+S.max(0).mean()))
                best=v[0] if best is None else max(best,v[0]); n+=1
        if best is not None: f['app_best']=float(best)
        f['app_n']=float(n)
        if lab[A['id']] and lab[B['id']]:
            f['lab_mean_d']=float(np.linalg.norm(np.mean(lab[A['id']],0)-np.mean(lab[B['id']],0)))
        ca,cb=major(A),major(B)
        f['cat']='?' if (ca is None or cb is None) else ('same' if ca==cb else 'diff'); f['cat_a']=ca; f['cat_b']=cb
        rows.append(f)
    return rows

if __name__=='__main__':
    day=sys.argv[1]; every=len(sys.argv)>2 and sys.argv[2]=='all'
    rows=build(day,every)
    out=f'feats_{day}{"_all" if every else ""}.pkl'; pickle.dump(rows,open(out,'wb'))
    print(day,'pairs',len(rows),'feature names',sorted(k for k in rows[0] if k not in('a','b','cat','cat_a','cat_b')))
