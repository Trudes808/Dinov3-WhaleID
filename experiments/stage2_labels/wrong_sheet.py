"""Contact sheet of the highest-ranked catalog-DIFFERENT cluster pairs on a val day (rank mean student best + window
where-to-look): are they really different whales, or is the cluster's catalog animal the other whale of the frame?"""
import pickle, numpy as np, json, sys, collections
from PIL import Image, ImageDraw
from scipy.stats import rankdata
DAYS='/home/sat3737/NOAA_whales/Dinov3-WhaleID/experiments/days_all'
day=sys.argv[1]; N=int(sys.argv[2]) if len(sys.argv)>2 else 12
def nrank(v):
    out=np.full(len(v),np.nan); f=np.isfinite(v)
    if f.any(): out[f]=rankdata(v[f])/f.sum()
    return out
def combo(cols):
    M=np.stack(cols,1); n=np.isfinite(M).sum(1); s=np.nansum(M,1)/np.maximum(n,1); s[n==0]=0.0; return s
R=pickle.load(open(f'transfer_raw_{day}.pkl','rb')); rows=pickle.load(open(f'feats_{day}_all.pkl','rb'))
z=json.load(open(f'{DAYS}/{day}/safe/clusters.json')); C={c['id']:c for c in z['clusters']}
P=json.load(open(f'{DAYS}/{day}/safe/photos.json')); bbox={it['key']:it['bbox'] for p in P.values() for it in p['instances']}
ninst={s:len(p['instances']) for s,p in P.items()}
kid={k:i for i,k in enumerate(R['keys'])}
best=np.array([r.get('app_best',np.nan) if r.get('app_best') is not None else np.nan for r in rows],float)
m=(R['cat']!='?')&(R['gap']>1.5); idx=np.where(m)[0]
s=combo([nrank(best[m]),nrank(R['S']['win_sal_max'][m])]); order=np.argsort(-s,kind='stable')
y=(R['cat'][m]=='same').astype(int)
picks=[]
for r,q in enumerate(order):
    if y[q]==0: picks.append((r+1,idx[q]))
    if len(picks)>=N: break
def tile(key,T=400):
    stem=key.split('~')[0]; im=Image.open(f'{DAYS}/{day}/images/{stem}.jpg').convert('RGB')
    x0,y0,x1,y1=bbox[key]; pad=0.12*max(x1-x0,y1-y0); cx,cy=(x0+x1)/2,(y0+y1)/2; h=max(x1-x0,y1-y0)/2+pad
    crop=im.crop((int(cx-h),int(cy-h),int(cx+h),int(cy+h))).resize((T,T),Image.LANCZOS)
    full=im.copy(); d=ImageDraw.Draw(full)
    for it in P[stem]['instances']:
        d.rectangle(it['bbox'],outline=(255,220,0) if it['key']==key else (255,60,60),width=10)
    W,H=full.size; full=full.resize((int(T*W/H),T),Image.LANCZOS)
    return crop,full
rowsimg=[]
for rank,i in picks:
    a,b=R["pairs"][i]; ka=[k for k in C[a]["keys"] if k in kid]; kb=[k for k in C[b]["keys"] if k in kid]
    if not ka or not kb: ka=list(C[a]["keys"]); kb=list(C[b]["keys"]); blk=np.full((1,1),np.nan)
    else: blk=R["D_sal"][np.ix_([kid[k] for k in ka],[kid[k] for k in kb])]
    if np.isfinite(blk).any(): u,v=np.unravel_index(np.nanargmax(blk),blk.shape)
    else: u,v=0,0
    ca,fa=tile(ka[u]); cb,fb=tile(kb[v])
    W=ca.width+fa.width+cb.width+fb.width+5*6; row=Image.new('RGB',(W,400+34),(20,20,20)); d=ImageDraw.Draw(row)
    x=6
    for t in (fa,ca,cb,fb): row.paste(t,(x,30)); x+=t.width+6
    txt=f"rank {rank}  clusters {a} | {b}  catalog {R['cat_a'][i]} vs {R['cat_b'][i]}  gap {R['gap'][i]:.0f} s  frames {ka[u]} ({ninst[ka[u].split('~')[0]]} inst) / {kb[v]} ({ninst[kb[v].split('~')[0]]} inst)  multi-frames in cluster {C[a].get('n_multi')} / {C[b].get('n_multi')}  sizes {len(C[a]['keys'])} / {len(C[b]['keys'])}"
    d.text((8,8),txt,fill=(255,255,255)); rowsimg.append(row); print(txt)
per=4
for s0 in range(0,len(rowsimg),per):
    part=rowsimg[s0:s0+per]; W=max(r.width for r in part); sheet=Image.new('RGB',(W,sum(r.height for r in part)),(20,20,20)); yy=0
    for r in part: sheet.paste(r,(0,yy)); yy+=r.height
    fn=f'wrong_{day}_{s0//per+1}.jpg'; sheet.save(fn,quality=86); print('saved',fn,sheet.size)
