"""Frozen DINOv3 ViT-B/16 features of square crops cut from the 2560-px day images.
crops: list of (stem, cx, cy, side) in 2560-frame px.  Returns dict of arrays aligned with crops."""
import sys, numpy as np, torch
from PIL import Image
sys.path.insert(0,'/home/sat3737/NOAA_whales/Dinov3-WhaleID')
from model_fine_tuning.common.backbone import load_backbone, IMAGENET_MEAN, IMAGENET_STD
DAYS='/home/sat3737/NOAA_whales/Dinov3-WhaleID/experiments/days_all'
MEAN=np.array(IMAGENET_MEAN,np.float32)*255; STD=np.array(IMAGENET_STD,np.float32)*255

def cut(im, cx, cy, side, R):
    W,H=im.size; s=float(side)
    x0=cx-s/2; y0=cy-s/2
    if s<=W: x0=min(max(x0,0),W-s)
    if s<=H: y0=min(max(y0,0),H-s)
    c=im.crop((int(round(x0)),int(round(y0)),int(round(x0+s)),int(round(y0+s))))
    return c.resize((R,R), Image.BICUBIC if s<R else Image.LANCZOS)

class Photos(torch.utils.data.Dataset):
    def __init__(s, day, crops, R):
        s.day=day; s.R=R; s.by={}
        for i,(stem,cx,cy,side) in enumerate(crops): s.by.setdefault(stem,[]).append((i,cx,cy,side))
        s.stems=sorted(s.by)
    def __len__(s): return len(s.stems)
    def __getitem__(s,j):
        stem=s.stems[j]; im=Image.open(f'{DAYS}/{s.day}/images/{stem}.jpg').convert('RGB')
        idx=[];arr=[]
        for i,cx,cy,side in s.by[stem]:
            a=(np.asarray(cut(im,cx,cy,side,s.R),np.float32)-MEAN)/STD
            arr.append(a.transpose(2,0,1)); idx.append(i)
        return torch.tensor(np.array(idx)), torch.from_numpy(np.stack(arr))
def collate(b): return torch.cat([x[0] for x in b]), torch.cat([x[1] for x in b])

_model=None
def model():
    global _model
    if _model is None:
        _model=load_backbone('vitb16').cuda().eval().half()
    return _model

@torch.no_grad()
def extract(day, crops, R=224, grid=False, layers=(8,11), bs=256, workers=12, log=None):
    m=model(); n=len(crops)
    out={f'cls{l}':np.zeros((n,768),np.float16) for l in layers}
    out.update({f'mean{l}':np.zeros((n,768),np.float16) for l in layers})
    if grid: out['grid']=np.zeros((n,(R//16)**2,768),np.float16)
    dl=torch.utils.data.DataLoader(Photos(day,crops,R),batch_size=8,num_workers=workers,collate_fn=collate)
    done=0
    def run(idx,x):
        for k in range(0,len(idx),bs):
            xb=x[k:k+bs].cuda(non_blocking=True).half(); ib=idx[k:k+bs].numpy()
            res=m.get_intermediate_layers(xb,n=list(layers),reshape=False,return_class_token=True,norm=True)
            for l,(pt,cl) in zip(layers,res):
                out[f'cls{l}'][ib]=cl.float().cpu().numpy(); out[f'mean{l}'][ib]=pt.float().mean(1).cpu().numpy()
            if grid: out['grid'][ib]=res[-1][0].cpu().numpy()
    for idx,x in dl:
        run(idx,x); done+=len(idx)
        if log and done//log != (done-len(idx))//log: print('  crops',done,'/',n,flush=True)
    return out
