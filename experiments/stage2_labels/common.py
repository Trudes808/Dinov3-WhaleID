import json, collections, os
import numpy as np
from PIL import Image, ImageDraw
ROOT='/home/sat3737/NOAA_whales'
DAYS=f'{ROOT}/Dinov3-WhaleID/experiments/days_all'
def load(day='2025-01-02'):
    z=json.load(open(f'{DAYS}/{day}/safe/clusters.json'))
    C={c['id']:c for c in z['clusters']}
    P=json.load(open(f'{DAYS}/{day}/safe/photos.json'))
    r=json.load(open(f'{ROOT}/Whale-Review-Web/data/{day}/pair_marks.json'))
    L=[x for x in r['labels'] if x.get('level')=='cluster']
    return z,C,P,L
def major(c):
    e=sorted((c.get('catalog') or {}).items(), key=lambda kv:-kv[1]); return e[0][0] if e else None
def gap(C,a,b):
    A,B=C[a],C[b]; return max(0.0, max(A['start'],B['start'])-min(A['end'],B['end']))
def photo(day,key):
    stem=key.split('~')[0]
    return Image.open(f'{DAYS}/{day}/images/{stem}.jpg').convert('RGB')
def inst(P,key):
    stem,i=key.split('~'); p=P[stem]
    for it in p['instances']:
        if it['key']==key: return it
    return None
def box_px(reg,W,H):
    return (reg['x']*W, reg['y']*H, (reg['x']+reg['w'])*W, (reg['y']+reg['h'])*H)
def crop_pad(im, box, pad=0.0, minside=0, square=True):
    x0,y0,x1,y1=box; w,h=x1-x0,y1-y0; cx,cy=(x0+x1)/2,(y0+y1)/2
    w,h=w*(1+2*pad),h*(1+2*pad)
    if square: w=h=max(w,h)
    w=max(w,minside); h=max(h,minside)
    b=(int(round(cx-w/2)),int(round(cy-h/2)),int(round(cx+w/2)),int(round(cy+h/2)))
    return im.crop(b), b
