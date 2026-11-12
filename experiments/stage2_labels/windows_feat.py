import pickle, numpy as np, sys, time
from feat import extract
day=sys.argv[1]
W=pickle.load(open(f'windows_{day}.pkl','rb'))
crops=[(w[1],w[2],w[3],w[4]) for w in W]
t=time.time()
F=extract(day,crops,R=224,grid=False,layers=(8,11),log=20000)
np.savez(f'windows_feat_{day}.npz',cls11=F['cls11'],cls8=F['cls8'],mean11=F['mean11'])
print('done',len(crops),round(time.time()-t),'s')
