import pickle, numpy as np, json, collections
from PIL import Image, ImageDraw
from match import load
from evalx import DS
day='2025-01-02'; DAYS='/home/sat3737/NOAA_whales/Dinov3-WhaleID/experiments/days_all'
d=load(day); kid=d['kid']; geo=d['geo']; inst=d['inst']; keys=d['keys']
sal=pickle.load(open(f'saliency_{day}.pkl','rb'))['sal_app']
P=json.load(open(f'{DAYS}/{day}/safe/photos.json'))
bbox={it['key']:it['bbox'] for p in P.values() for it in p['instances']}
by=collections.defaultdict(list)
for i,k in enumerate(inst): by[k].append(i)
boxes=collections.defaultdict(list)
for g in DS['regions']:
    for s in 'ab': boxes[g['ikey_'+s]].append(g['box_'+s])
rng=np.random.default_rng(3); pick=list(rng.choice(sorted(boxes),18,replace=False))
tiles=[]
for k in pick:
    stem=k.split('~')[0]; im=Image.open(f'{DAYS}/{day}/images/{stem}.jpg').convert('RGB'); dr=ImageDraw.Draw(im)
    w=np.array(by[kid[k]]); top=w[np.argsort(-sal[w])[:5]]
    for r,u in enumerate(top):
        cx,cy,s=geo[u][:3]; dr.rectangle((cx-s/2,cy-s/2,cx+s/2,cy+s/2),outline=(255,220,0),width=6 if r==0 else 3)
    for b in boxes[k]: dr.rectangle(b,outline=(0,255,80),width=5)
    x0,y0,x1,y1=bbox[k]; pad=0.15*max(x1-x0,y1-y0); cx,cy=(x0+x1)/2,(y0+y1)/2; s=max(x1-x0,y1-y0)/2+pad
    c=im.crop((int(cx-s),int(cy-s),int(cx+s),int(cy+s))).resize((420,420),Image.LANCZOS); tiles.append(c)
sheet=Image.new('RGB',(6*424,3*424),(20,20,20))
for i,t in enumerate(tiles): sheet.paste(t,((i%6)*424+2,(i//6)*424+2))
sheet.save('saliency_sheet.jpg',quality=88); print('saved',sheet.size)
