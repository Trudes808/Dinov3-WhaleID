"""One frame per cluster shot with a wide lens (< 100 mm): which tracked instances are not whales?"""
import json, numpy as np, sys
from PIL import Image, ImageDraw
DAYS='/home/sat3737/NOAA_whales/Dinov3-WhaleID/experiments/days_all'
day=sys.argv[1]; T=330
z=json.load(open(f'{DAYS}/{day}/safe/clusters.json')); C={c['id']:c for c in z['clusters']}
P=json.load(open(f'{DAYS}/{day}/safe/photos.json')); I={it['key']:(it,p) for p in P.values() for it in p['instances']}
tiles=[]
for cid,c in sorted(C.items()):
    its=[(k,)+I[k] for k in c['keys'] if k in I]
    foc=np.nanmedian([p.get('focal_mm') or np.nan for k,it,p in its])
    if not foc<100: continue
    k,it,p=its[len(its)//2]; stem=k.split('~')[0]
    im=Image.open(f'{DAYS}/{day}/images/{stem}.jpg').convert('RGB'); d=ImageDraw.Draw(im)
    for o in P[stem]['instances']: d.rectangle(o['bbox'],outline=(255,220,0) if o['key']==k else (255,60,60),width=12)
    W,H=im.size; s=T/max(W,H); im=im.resize((int(W*s),int(H*s)),Image.LANCZOS)
    t=Image.new('RGB',(T,T+18),(20,20,20)); t.paste(im,((T-im.width)//2,18)); b=it['bbox']
    ImageDraw.Draw(t).text((3,3),f"{cid} n{len(its)} {foc:.0f}mm h/w {(b[3]-b[1])/(b[2]-b[0]):.1f} {'+'.join(x.split('_')[-1] for x in (c.get('catalog') or {}))}",fill=(255,255,255))
    tiles.append(t)
cols=6; per=cols*4
for s0 in range(0,len(tiles),per):
    part=tiles[s0:s0+per]; rows=(len(part)+cols-1)//cols; sheet=Image.new('RGB',(cols*(T+4),rows*(T+22)),(20,20,20))
    for i,t in enumerate(part): sheet.paste(t,((i%cols)*(T+4)+2,(i//cols)*(T+22)+2))
    fn=f'wide_{day}_{s0//per+1}.jpg'; sheet.save(fn,quality=85); print('saved',fn,sheet.size,len(part))
