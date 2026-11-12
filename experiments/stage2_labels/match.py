"""Window-bag matching between whale instances of one day (frozen DINOv3 window features).
M[u, j] = best similarity of window u to any window of instance j (window sizes within one step)."""
import pickle, numpy as np, torch, time, sys, json
SIZES=(100,150,225,340,510,765,1150)
def load(day, feat='cls11'):
    W=pickle.load(open(f'windows_{day}.pkl','rb'))
    F=np.load(f'windows_feat_{day}.npz')[feat]
    keys=sorted(set(w[0] for w in W)); kid={k:i for i,k in enumerate(keys)}
    inst=np.array([kid[w[0]] for w in W]); size=np.array([SIZES.index(int(w[4])) for w in W])
    geo=np.array([(w[2],w[3],w[4],w[5]) for w in W],np.float32)
    return dict(W=W,F=F,keys=keys,kid=kid,inst=inst,size=size,geo=geo)
def whiten_fit(X, dim=128, sample=60000, seed=0):
    rng=np.random.default_rng(seed); idx=rng.choice(len(X),min(sample,len(X)),replace=False)
    Xs=X[idx].astype(np.float32); Xs/=np.linalg.norm(Xs,axis=1,keepdims=True)
    mu=Xs.mean(0); U,s,Vt=np.linalg.svd(Xs-mu,full_matrices=False)
    return mu, (Vt[:dim].T/(s[:dim]/np.sqrt(len(Xs)))).astype(np.float32)
def transform(X, mu=None, Pm=None, T=None, bs=65536):
    out=[]
    for k in range(0,len(X),bs):
        x=torch.tensor(X[k:k+bs].astype(np.float32)).cuda(); x=torch.nn.functional.normalize(x,dim=1)
        if mu is not None: x=(x-torch.tensor(mu).cuda())@torch.tensor(Pm).cuda()
        if T is not None: x=x@torch.tensor(T).cuda()
        out.append(torch.nn.functional.normalize(x,dim=1).half())
    return torch.cat(out)
@torch.no_grad()
def best_match(Z, inst, size, n_inst, step=1, chunk=4096):
    """Z: (N,d) unit half tensor on GPU. Returns M (N, n_inst) float16 and the argmax window index A (N, n_inst) int32."""
    N=len(Z); M=np.full((N,n_inst),-1.0,np.float16); A=np.full((N,n_inst),-1,np.int32)
    inst_t=torch.tensor(inst).cuda(); 
    for k in sorted(set(size.tolist())):
        ui=np.where(size==k)[0]; vi=np.where(np.abs(size-k)<=step)[0]
        vi=vi[np.argsort(inst[vi],kind='stable')]; vinst=inst[vi]
        js,starts=np.unique(vinst,return_index=True); ends=np.r_[starts[1:],len(vi)]
        Zv=Z[torch.tensor(vi).cuda()]; vi_t=torch.tensor(vi).cuda()
        for c in range(0,len(ui),chunk):
            u=ui[c:c+chunk]; S=Z[torch.tensor(u).cuda()]@Zv.T            # (chunk, Nv)
            mx=torch.empty((len(u),len(js)),dtype=torch.float16,device='cuda'); am=torch.empty((len(u),len(js)),dtype=torch.int64,device='cuda')
            for q,(a,b) in enumerate(zip(starts,ends)):
                m,i=S[:,a:b].max(1); mx[:,q]=m; am[:,q]=vi_t[a+i]
            M[np.ix_(u,js)]=mx.cpu().numpy(); A[np.ix_(u,js)]=am.cpu().numpy().astype(np.int32)
    return M,A
def znorm(M, inst, cluster_of_inst=None):
    """Per window: how unusual is its best match in instance j, against its matches in all instances (robust stats)."""
    Mf=M.astype(np.float32); Mf[Mf<=-1]=np.nan
    med=np.nanmedian(Mf,axis=1,keepdims=True); mad=np.nanmedian(np.abs(Mf-med),axis=1,keepdims=True)*1.4826+1e-3
    return (Mf-med)/mad
def instance_scores(V, inst, n_inst):
    """D[i,j] = max over windows u of instance i of V[u,j]; also which window."""
    D=np.full((n_inst,n_inst),np.nan,np.float32); U=np.full((n_inst,n_inst),-1,np.int32)
    order=np.argsort(inst,kind='stable'); js,starts=np.unique(inst[order],return_index=True); ends=np.r_[starts[1:],len(order)]
    for i,a,b in zip(js,starts,ends):
        rows=order[a:b]; blk=V[rows]; blk=np.where(np.isfinite(blk),blk,-1e9)
        k=blk.argmax(0); D[i]=blk[k,np.arange(n_inst)]; U[i]=rows[k]
    D[D<=-1e8]=np.nan
    return D,U
if __name__=='__main__':
    day=sys.argv[1]; variant=sys.argv[2] if len(sys.argv)>2 else 'pcaw128'
    t=time.time(); d=load(day); n=len(d['keys'])
    if variant=='raw': Z=transform(d['F'])
    else:
        dim=int(variant.replace('pcaw','')); mu,Pm=whiten_fit(d['F'],dim); Z=transform(d['F'],mu,Pm)
    M,A=best_match(Z,d['inst'],d['size'],n)
    np.save(f'M_{day}_{variant}.npy',M); np.save(f'A_{day}_{variant}.npy',A)
    print(day,variant,'windows',len(Z),'instances',n,'time',round(time.time()-t),'s')
