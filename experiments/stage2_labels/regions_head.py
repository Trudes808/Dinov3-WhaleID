"""Can a small head on frozen region features learn the researcher's matches?  CV grouped by animal."""
import pickle, numpy as np, torch, sys
from sklearn.metrics import roc_auc_score
ds=pickle.load(open('ds.pkl','rb')); R=ds['regions']; n=len(R); cfold=ds['cfold']; K=ds['K']
recs=ds['recs']
F=pickle.load(open('regions_feat.pkl','rb')); meta=F['meta']
grp=np.array([str(d['grp']) for d in R]); fold=np.array([cfold[recs[d['rec']]['a']] for d in R])
neg=grp[:,None]!=grp[None,:]
def cueclass(d):
    c=set(d['cue'])
    return 'fluke' if 'fluke_shape' in c else 'dorsal' if 'dorsal_shape' in c else 'mark' if c else 'none'
cc=np.array([cueclass(d) for d in R]); samecue=cc[:,None]==cc[None,:]
def idx(side,scale): return np.array([k for k,(i,s,sc) in enumerate(meta) if s==side and abs(sc-scale)<1e-6])
def norm(x): x=np.asarray(x,np.float32); return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-6)
def feats(kind):
    if kind=='cls11@1.15': return norm(F['cls11'][idx('a',1.15)]),norm(F['cls11'][idx('b',1.15)])
    if kind=='cls11@1.15+2.5':
        return [np.concatenate([norm(F['cls11'][idx(s,1.15)]),norm(F['cls11'][idx(s,2.5)])],1)/np.sqrt(2) for s in 'ab']
    if kind=='cls11+cls8@1.15+2.5':
        return [np.concatenate([norm(F[f][idx(s,sc)]) for f in ('cls11','cls8') for sc in (1.15,2.5)],1)/2 for s in 'ab']
def evalS(name,S,rows=None):
    rows=np.arange(n) if rows is None else rows
    pos=np.diag(S)[rows]
    m=neg&samecue; mm=np.zeros_like(m); mm[rows]=m[rows]
    a_cue=roc_auc_score(np.r_[np.ones(len(rows)),np.zeros(mm.sum())],np.r_[pos,S[mm]])
    ranks=[]
    for i in rows:
        ranks.append(int((S[i][neg[i]]>S[i,i]).sum())); ranks.append(int((S[:,i][neg[:,i]]>S[i,i]).sum()))
    ranks=np.array(ranks)
    print(f'{name:50s} same-cue AUC {a_cue:.3f} | partner top1 {np.mean(ranks==0):.3f} top5 {np.mean(ranks<5):.3f} median rank {np.median(ranks):.0f}',flush=True)
    return a_cue,float(np.mean(ranks==0)),float(np.mean(ranks<5))

def wccn(A,B,tr,dim,shrink):
    X=np.concatenate([A[tr],B[tr]]); mu=X.mean(0); Xc=X-mu
    U,s,Vt=np.linalg.svd(Xc,full_matrices=False); Pm=Vt[:dim].T/ (s[:dim]/np.sqrt(len(X)))   # PCA-whiten
    d=((A[tr]-B[tr])@Pm)/np.sqrt(2); Sw=d.T@d/len(d)
    Sw=(1-shrink)*Sw+shrink*np.eye(dim)*np.trace(Sw)/dim
    w,V=np.linalg.eigh(Sw); T=V@np.diag(w**-0.5)@V.T
    f=lambda Z: norm(((Z-mu)@Pm)@T)
    return f
def head_nce(A,B,tr,dim=128,epochs=300,lr=1e-3,wd=1e-2,drop=0.3,temp=0.07,seed=0,hidden=0):
    torch.manual_seed(seed)
    a=torch.tensor(A[tr]).cuda(); b=torch.tensor(B[tr]).cuda(); g=torch.tensor(np.unique(grp[tr],return_inverse=True)[1]).cuda()
    D=A.shape[1]
    net=torch.nn.Sequential(torch.nn.Dropout(drop),torch.nn.Linear(D,dim)) if not hidden else torch.nn.Sequential(torch.nn.Dropout(drop),torch.nn.Linear(D,hidden),torch.nn.GELU(),torch.nn.Linear(hidden,dim))
    net=net.cuda(); opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=wd)
    same=(g[:,None]==g[None,:])
    for ep in range(epochs):
        net.train(); za=torch.nn.functional.normalize(net(a),dim=1); zb=torch.nn.functional.normalize(net(b),dim=1)
        S=za@zb.T/temp
        mask=same&~torch.eye(len(a),dtype=torch.bool,device=S.device)     # same animal, other link: not a negative
        S=S.masked_fill(mask,-1e4)
        t=torch.arange(len(a),device=S.device)
        loss=0.5*(torch.nn.functional.cross_entropy(S,t)+torch.nn.functional.cross_entropy(S.T,t))
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return lambda Z: torch.nn.functional.normalize(net(torch.tensor(np.asarray(Z,np.float32)).cuda()),dim=1).detach().cpu().numpy()
def cv(name,A,B,maker):
    S=np.zeros((n,n),np.float32)
    for f in range(K):
        tr=np.where(fold!=f)[0]; te=np.where(fold==f)[0]
        fn=maker(A,B,tr); Za,Zb=fn(A),fn(B)
        S[te]=(Za@Zb.T)[te]         # rows = held-out queries (a side); columns = everything
        # for column-direction ranks of held-out items use the same transform
        S[:,te]=np.where(np.isin(np.arange(n),te)[:,None],S[:,te],(Za@Zb.T)[:,te])
    return evalS(name,S)
if __name__=='__main__':
    for kind in ('cls11@1.15','cls11@1.15+2.5','cls11+cls8@1.15+2.5'):
        A,B=feats(kind)
        evalS(f'{kind} raw cosine',A@B.T)
        for dim in (32,64,128):
            for sh in (0.3,0.7):
                cv(f'{kind} wccn dim {dim} shrink {sh}',A,B,lambda A,B,tr:wccn(A,B,tr,dim,sh))
        for dim,drop,hid in ((128,0.3,0),(128,0.6,0),(256,0.5,512)):
            cv(f'{kind} nce dim {dim} drop {drop} hid {hid}',A,B,lambda A,B,tr:head_nce(A,B,tr,dim=dim,drop=drop,hidden=hid))
