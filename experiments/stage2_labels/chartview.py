import numpy as np, glob, sys
from common import *
day='2025-01-02'; z,C,P,L=load(day)
def chart(key,part):
    f=f'{DAYS}/{day}/parts_hr/{key}__{part}.npz'
    if not os.path.exists(f): return None,None
    n=np.load(f,allow_pickle=True); im=Image.open(f'{DAYS}/{day}/parts_hr/charts/{key}__{part}.jpg').convert('RGB')
    return n,im
def reg_in_chart(n, reg, W=2560, H=None, im_size=None):
    # region (photo fractions) -> full-res px -> chart px via M
    Wp,Hp=im_size; s=float(n['scale_full'])
    x0,y0,x1,y1=reg['x']*Wp*s, reg['y']*Hp*s, (reg['x']+reg['w'])*Wp*s, (reg['y']+reg['h'])*Hp*s
    pts=np.array([[x0,y0,1],[x1,y0,1],[x1,y1,1],[x0,y1,1]]).T
    q=(n['M']@pts).T
    return q
tiles=[]
recs=[x for x in L if x['verdict']=='same' and gap(C,x['set_a'],x['set_b'])>60][:400]
import random; random.seed(1); random.shuffle(recs)
for x in recs[:6]:
    row=[]
    for side in 'ab':
        key=x['key_'+side]; ph=photo(day,key)
        for part in ('fin','fluke','body'):
            n,im=chart(key,part)
            if n is None: continue
            q=reg_in_chart(n,x['regions'][0][side],im_size=ph.size)
            d=ImageDraw.Draw(im); d.polygon([tuple(p) for p in q],outline=(255,255,0)); 
            for w in range(1,4): d.polygon([tuple(p+w) for p in q],outline=(255,255,0))
            # overlay masks faintly
            m=np.asarray(im).copy()
            for name,col in (('silhouette',(255,0,0)),('fin_band',(0,255,0)),('fluke',(0,128,255))):
                if name in n.files:
                    mk=n[name]; e=mk ^ np.roll(mk,3,0) | mk ^ np.roll(mk,3,1); m[e]=col
            im=Image.fromarray(m); d=ImageDraw.Draw(im); d.text((10,10),f"{part} view_side {int(n['view_side'])} mirrored {bool(n['mirrored'])} " + (f"underside {bool(n['underside'])} flipv {bool(n['flipped_v'])}" if part=='fluke' else ''),fill=(255,255,255))
            row.append(im.resize((384,384)))
    tiles.append(row)
W=max(len(r) for r in tiles)*384; S=Image.new('RGB',(W,384*len(tiles)))
for i,r in enumerate(tiles):
    for j,t in enumerate(r): S.paste(t,(j*384,i*384))
S.save('charts_regions.jpg',quality=85); print(S.size)
