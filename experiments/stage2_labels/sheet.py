import sys, json
from common import *
day='2025-01-02'; z,C,P,L=load(day)
def panel(x, side, size=360):
    key=x['key_'+side]; im=photo(day,key); W,H=im.size
    it=inst(P,key); bb=it['bbox'] if it else (0,0,W,H)
    regs=[box_px(r[side],W,H) for r in x['regions']]
    # context: union of instance bbox and regions
    xs=[bb[0],bb[2]]+[v for r in regs for v in (r[0],r[2])]; ys=[bb[1],bb[3]]+[v for r in regs for v in (r[1],r[3])]
    ctx,b=crop_pad(im,(min(xs),min(ys),max(xs),max(ys)),pad=0.08,square=False)
    d=ImageDraw.Draw(ctx)
    for r in regs: d.rectangle((r[0]-b[0],r[1]-b[1],r[2]-b[0],r[3]-b[1]),outline=(255,255,0),width=3)
    s=min(size*2/ctx.size[0], size/ctx.size[1]); ctx=ctx.resize((max(1,int(ctx.size[0]*s)),max(1,int(ctx.size[1]*s))),Image.LANCZOS)
    can=Image.new('RGB',(size*2,size),(20,20,20)); can.paste(ctx,(0,0))
    zooms=[]
    for r in regs[:1]:
        zc,_=crop_pad(im,r,pad=0.6,minside=96); zooms.append(zc.resize((size,size),Image.LANCZOS))
    if not zooms:
        zc,_=crop_pad(im,bb,pad=0.05); zooms.append(zc.resize((size,size),Image.LANCZOS))
    return can, zooms[0]
def row(x, size=360):
    ca,za=panel(x,'a',size); cb,zb=panel(x,'b',size)
    r=Image.new('RGB',(size*6+30,size+26),(0,0,0))
    r.paste(ca,(0,26)); r.paste(za,(size*2+5,26)); r.paste(zb,(size*3+15,26)); r.paste(cb,(size*4+25,26))
    d=ImageDraw.Draw(r); a,b=x['set_a'],x['set_b']
    d.text((4,6),f"{a}|{b} {x['verdict']} cat:{x.get('catalog')}  A {major(C[a]) and major(C[a])[-6:]} ({x['key_a'].split('-')[-1]})  B {major(C[b]) and major(C[b])[-6:]} ({x['key_b'].split('-')[-1]})  gap {gap(C,a,b):.0f}s cue {','.join(x['cue'])} score {x.get('model_score')}",fill=(255,255,255))
    return r
def sheet(recs,out,size=360):
    rows=[row(x,size) for x in recs]
    S=Image.new('RGB',(rows[0].size[0],sum(r.size[1] for r in rows)))
    y=0
    for r in rows: S.paste(r,(0,y)); y+=r.size[1]
    S.save(out,quality=88); print(out,S.size)
if __name__=='__main__':
    conf=[x for x in L if x['verdict']=='same' and major(C[x['set_a']]) and major(C[x['set_b']]) and major(C[x['set_a']])!=major(C[x['set_b']])]
    for i in range(0,len(conf),4): sheet(conf[i:i+4],f'conflict_{i//4}.jpg')
