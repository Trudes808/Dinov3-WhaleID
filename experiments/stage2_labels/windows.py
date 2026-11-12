"""Window proposals on every whale instance of a day: square windows at several sizes on the whale mask (2560-px frame)."""
import sys, json, pickle, numpy as np
from PIL import Image
DAYS='/home/sat3737/NOAA_whales/Dinov3-WhaleID/experiments/days_all'
SIZES=(100,150,225,340,510,765,1150)
def integral(m):
    return np.pad(m.astype(np.int64).cumsum(0).cumsum(1),((1,0),(1,0)))
def boxsum(I,x0,y0,x1,y1):
    return I[y1,x1]-I[y0,x1]-I[y1,x0]+I[y0,x0]
def proposals(day, min_frac=0.2, sizes=SIZES):
    P=json.load(open(f'{DAYS}/{day}/safe/photos.json'))
    out=[]   # (key, stem, cx, cy, side, frac)
    for stem,p in P.items():
        if not p['instances']: continue
        m=np.asarray(Image.open(f'{DAYS}/{day}/canon/{stem}_mask.png'))>0
        H,W=m.shape; I=integral(m)
        for it in p['instances']:
            x0,y0,x1,y1=it['bbox']; bw,bh=x1-x0,y1-y0
            for s in sizes:
                st=s/2
                xs=np.arange(x0, x1+1e-6, st) if bw>st else np.array([(x0+x1)/2])
                ys=np.arange(y0, y1+1e-6, st) if bh>st else np.array([(y0+y1)/2])
                for cy in ys:
                    for cx in xs:
                        a0=int(np.clip(round(cx-s/2),0,W)); a1=int(np.clip(round(cx+s/2),0,W)); b0=int(np.clip(round(cy-s/2),0,H)); b1=int(np.clip(round(cy+s/2),0,H))
                        if a1<=a0 or b1<=b0: continue
                        fr=boxsum(I,a0,b0,a1,b1)/float(s*s)
                        if fr>=min_frac: out.append((it['key'],stem,float(cx),float(cy),float(s),float(fr)))
    return out
if __name__=='__main__':
    day=sys.argv[1]
    w=proposals(day); pickle.dump(w,open(f'windows_{day}.pkl','wb'))
    import collections
    per=collections.Counter(x[0] for x in w); n=np.array(list(per.values()))
    print(day,'windows',len(w),'instances',len(per),'per instance pct',np.percentile(n,[5,25,50,75,95]).round(0),'by size',collections.Counter(x[4] for x in w))
