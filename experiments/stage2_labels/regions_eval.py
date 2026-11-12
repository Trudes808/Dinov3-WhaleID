import pickle, numpy as np, sys, torch
from sklearn.metrics import roc_auc_score
sys.path.insert(0,'/home/sat3737/NOAA_whales/Dinov3-WhaleID')
from pathlib import Path
from model_fine_tuning.cluster.safe_clusters import load_embeddings
ds=pickle.load(open('ds.pkl','rb')); R=ds['regions']; n=len(R)
F=pickle.load(open('regions_feat.pkl','rb')); meta=F['meta']
grp=np.array([str(d['grp']) for d in R])
neg=grp[:,None]!=grp[None,:]
def cueclass(d):
    c=set(d['cue'])
    if 'fluke_shape' in c: return 'fluke'
    if 'dorsal_shape' in c: return 'dorsal'
    if c: return 'mark'
    return 'none'
cc=np.array([cueclass(d) for d in R]); print('cue classes',{k:int((cc==k).sum()) for k in set(cc)})
samecue=cc[:,None]==cc[None,:]
def idx(side,scale): return np.array([k for k,(i,s,sc) in enumerate(meta) if s==side and abs(sc-scale)<1e-6])
def norm(x): x=x.astype(np.float32); return x/np.linalg.norm(x,axis=-1,keepdims=True)
def evalS(name,S):
    pos=np.diag(S); 
    a_all=roc_auc_score(np.r_[np.ones(n),np.zeros(neg.sum())],np.r_[pos,S[neg]])
    m=neg&samecue; a_cue=roc_auc_score(np.r_[np.ones(n),np.zeros(m.sum())],np.r_[pos,S[m]])
    # retrieval both directions among other-animal candidates
    ranks=[]
    for i in range(n):
        ranks.append(int((S[i][neg[i]]>pos[i]).sum())); ranks.append(int((S[:,i][neg[:,i]]>pos[i]).sum()))
    ranks=np.array(ranks)
    print(f'{name:44s} AUC all {a_all:.3f}  same-cue {a_cue:.3f} | partner rank: top1 {np.mean(ranks==0):.3f} top5 {np.mean(ranks<5):.3f} median {np.median(ranks):.0f}')
    return a_all,a_cue,np.mean(ranks==0)
res={}
for scale in (1.15,2.5,5.0):
    ia,ib=idx('a',scale),idx('b',scale)
    for ft in ('cls11','mean11','cls8','mean8'):
        A,B=norm(F[ft][ia]),norm(F[ft][ib]); res[(scale,ft)]=evalS(f'scale {scale} {ft}',A@B.T)
    A=np.concatenate([norm(F['cls11'][ia]),norm(F['mean11'][ia])],1); B=np.concatenate([norm(F['cls11'][ib]),norm(F['mean11'][ib])],1)
    evalS(f'scale {scale} cls11+mean11',A@B.T/2)
# multi-scale sums
def S_of(scale,ft): return norm(F[ft][idx('a',scale)])@norm(F[ft][idx('b',scale)]).T
evalS('cls11 scales 1.15+2.5',S_of(1.15,'cls11')+S_of(2.5,'cls11'))
evalS('cls11 scales 1.15+2.5+5',S_of(1.15,'cls11')+S_of(2.5,'cls11')+S_of(5.0,'cls11'))
evalS('mean11 scales 1.15+2.5',S_of(1.15,'mean11')+S_of(2.5,'mean11'))
# best-buddy grid matching at the tight scale
for scale in (1.15,2.5):
    ga=torch.tensor(F['grid'][idx('a',scale)]).cuda().float(); gb=torch.tensor(F['grid'][idx('b',scale)]).cuda().float()
    ga=ga/ga.norm(dim=-1,keepdim=True); gb=gb/gb.norm(dim=-1,keepdim=True)
    S=np.zeros((n,n),np.float32)
    for i in range(n):
        sim=torch.einsum('pc,jqc->jpq',ga[i],gb)          # (n,196,196)
        S[i]=(0.5*(sim.max(2).values.mean(1)+sim.max(1).values.mean(1))).cpu().numpy()
    evalS(f'scale {scale} grid best-buddy mean',S)
# student, same instance pairs (in-sample on this day)
emb=load_embeddings(Path('/home/sat3737/NOAA_whales/Dinov3-WhaleID/experiments/days_all/2025-01-02'))
for part in ('fin','fluke','body','best'):
    S=np.full((n,n),np.nan,np.float32)
    parts=('fin','fluke','body') if part=='best' else (part,)
    for p in parts:
        E=emb[p]; A=[E.get(d['ikey_a']) for d in R]; B=[E.get(d['ikey_b']) for d in R]
        ok_a=np.array([a is not None for a in A]); ok_b=np.array([b is not None for b in B])
        Am=np.stack([a if a is not None else np.zeros(512,np.float32) for a in A]); Bm=np.stack([b if b is not None else np.zeros(512,np.float32) for b in B])
        Sp=Am@Bm.T; Sp[~ok_a]=np.nan; Sp[:,~ok_b]=np.nan
        S=np.fmax(S,Sp)
    have=np.isfinite(np.diag(S)); S2=np.where(np.isfinite(S),S,-1.0)
    print(f'student {part}: pairs with the part on both sides {have.sum()} of {n}')
    evalS(f'student whole-chart {part} (missing = -1)',S2)
