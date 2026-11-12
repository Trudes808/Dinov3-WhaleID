import pickle, numpy as np, sys
from feat import extract
ds=pickle.load(open('ds.pkl','rb')); R=ds['regions']
crops=[];meta=[]
for i,d in enumerate(R):
    for s in 'ab':
        x0,y0,x1,y1=d['box_'+s]; cx,cy=(x0+x1)/2,(y0+y1)/2; side=max(x1-x0,y1-y0)
        stem=d['key_'+s].split('~')[0]
        for scale in (1.15,2.5,5.0):
            crops.append((stem,cx,cy,max(side*scale,32))); meta.append((i,s,scale))
F=extract(ds['day'],crops,R=224,grid=True)
F['meta']=np.array(meta,dtype=object)
pickle.dump(F,open('regions_feat.pkl','wb'),protocol=4)
print('crops',len(crops),{k:v.shape for k,v in F.items()})
