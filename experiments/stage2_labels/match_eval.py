import pickle, numpy as np, sys, json, collections
from match import *
from evalx import *
day='2025-01-02'
def cluster_scores(variant, d=None, verbose=True):
    d=d or load(day); n=len(d['keys']); kid=d['kid']
    M=np.load(f'M_{day}_{variant}.npy'); 
    z=json.load(open(f'/home/sat3737/NOAA_whales/Dinov3-WhaleID/experiments/days_all/{day}/safe/clusters.json')); C={c['id']:c for c in z['clusters']}
    Vz=znorm(M,d['inst']); Vr=M.astype(np.float32); Vr[Vr<=-1]=np.nan
    Dz,Uz=instance_scores(Vz,d['inst'],n); Dr,Ur=instance_scores(Vr,d['inst'],n)
    out={}
    for name,D in (('z',Dz),('raw',Dr)):
        sym={'max':np.fmax(D,D.T),'min':np.fmin(D,D.T),'mean':0.5*(D+D.T)}
        for sn,S in sym.items():
            sc_key=[];sc_max=[];sc_top3=[];sc_rowmax=[]
            for r in RECS:
                ia=[kid[k] for k in C[r['a']]['keys'] if k in kid]; ib=[kid[k] for k in C[r['b']]['keys'] if k in kid]
                ka,kb=kid.get(r['key_a']),kid.get(r['key_b'])
                sc_key.append(S[ka,kb] if ka is not None and kb is not None else np.nan)
                if ia and ib:
                    blk=S[np.ix_(ia,ib)]; v=np.sort(blk[np.isfinite(blk)])[::-1]
                    sc_max.append(v[0] if len(v) else np.nan); sc_top3.append(v[:3].mean() if len(v) else np.nan)
                    sc_rowmax.append(0.5*(np.nanmax(blk,1).mean()+np.nanmax(blk,0).mean()) if len(v) else np.nan)
                else: sc_max.append(np.nan); sc_top3.append(np.nan); sc_rowmax.append(np.nan)
            out[(name,sn,'key')]=np.array(sc_key); out[(name,sn,'max')]=np.array(sc_max); out[(name,sn,'top3')]=np.array(sc_top3); out[(name,sn,'rowmax')]=np.array(sc_rowmax)
    if verbose:
        for k,v in out.items():
            if k[1] in ('max','min') and k[2] in ('key','max','rowmax'):
                report(f'{variant} {k[0]} sym-{k[1]} {k[2]}',v,subsets=(('gap>1.5s',1.5),('gap>60s',60.0)))
    return out,dict(Dz=Dz,Uz=Uz,Dr=Dr,Ur=Ur,d=d,C=C)
def localisation(aux, variant):
    d=aux['d']; kid=d['kid']; geo=d['geo']; A=np.load(f'A_{day}_{variant}.npy')
    def iou(c1,c2):
        x0=max(c1[0]-c1[2]/2,c2[0]-c2[2]/2); x1=min(c1[0]+c1[2]/2,c2[0]+c2[2]/2); y0=max(c1[1]-c1[2]/2,c2[1]-c2[2]/2); y1=min(c1[1]+c1[2]/2,c2[1]+c2[2]/2)
        inter=max(x1-x0,0)*max(y1-y0,0); return inter/(c1[2]**2+c2[2]**2-inter)
    for nm,U in (('z',aux['Uz']),('raw',aux['Ur'])):
        hit=[];both=[]
        for g in DS['regions']:
            ia,ib=kid.get(g['ikey_a']),kid.get(g['ikey_b'])
            if ia is None or ib is None: continue
            res=[]
            for (i,j,bi,bj) in ((ia,ib,g['box_a'],g['box_b']),(ib,ia,g['box_b'],g['box_a'])):
                u=U[i,j]; v=A[u,j]
                ci=((bi[0]+bi[2])/2,(bi[1]+bi[3])/2,max(bi[2]-bi[0],bi[3]-bi[1])*1.15); cj=((bj[0]+bj[2])/2,(bj[1]+bj[3])/2,max(bj[2]-bj[0],bj[3]-bj[1])*1.15)
                res.append((iou(ci,geo[u][:3]),iou(cj,geo[v][:3])))
            hit+= [r[0]>0.15 for r in res]; both+=[ (r[0]>0.15 and r[1]>0.15) for r in res]
        print(f'{variant} {nm}: best window lands on the researcher box (IoU>0.15): {np.mean(hit):.3f}; both ends on the boxed pair: {np.mean(both):.3f}  (n={len(hit)})')
if __name__=='__main__':
    variant=sys.argv[1]
    out,aux=cluster_scores(variant); localisation(aux,variant)
    pickle.dump(out,open(f'match_scores_{variant}.pkl','wb'))
