import json, numpy as np
from regions_head2 import *
def pcaw(A,B,tr,dim):
    X=np.concatenate([A[tr],B[tr]]); mu=X.mean(0); U,s,Vt=np.linalg.svd(X-mu,full_matrices=False); Pm=Vt[:dim].T/(s[:dim]/np.sqrt(len(X)))
    return lambda Z: norm((Z-mu)@Pm)
def pcaonly(A,B,tr,dim):
    X=np.concatenate([A[tr],B[tr]]); mu=X.mean(0); U,s,Vt=np.linalg.svd(X-mu,full_matrices=False); Pm=Vt[:dim].T
    return lambda Z: norm((Z-mu)@Pm)
print('--- ablation: what the labels add over unsupervised whitening')
for kind in ('cls11@1.15','cls11@1.15+2.5'):
    A,B=[np.asarray(x,np.float32) for x in feats(kind)]
    for dim in (64,128,256):
        ev2(f'{kind} centre+PCA {dim}',cvS(A,B,lambda A,B,tr:pcaonly(A,B,tr,dim)))
        ev2(f'{kind} PCA-whiten {dim}',cvS(A,B,lambda A,B,tr:pcaw(A,B,tr,dim)))
        ev2(f'{kind} wccn {dim}',cvS(A,B,lambda A,B,tr:wccn(A,B,tr,dim,0.3)))
