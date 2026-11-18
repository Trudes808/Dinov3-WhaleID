"""pixels: what the frozen DINOv3 ViT-B/16 sees. Pooled features of the bench's fluke / dorsal instances under input
transforms and poolings (agent `pixels`, 2026-11-18). GPU; run under the shared lock.

    python extract.py JOB [DAY ...]      JOB = base | norm | res   -> moonshot_cache/pixels/JOB/<day>.npz

Every view is cut like cluster/view_class.py (box x 1.15; sq = 448-px square, ar = shape-keeping bucket), then
transformed, then run through the frozen backbone (layers 7 and 11, final norm). Per view and layer the npz keeps
  cls      the class token
  mean     mean of all patch tokens                       (BASE when view = sq / ar, transform rgb, layer 11)
  wc       mean weighted by the whale mask (canon/<stem>_mask.png inside the instance box) per patch
  wi       the same with instances_v2/masks
  part     mean weighted by the part region (parts_seg_r05_a: fluke pixels; dorsal fin box widened to fin + hump,
           times the whale mask); falls back to wc when the instance has no part pixels
Transforms: rgb; fill (water -> flat ImageNet-mean grey, whale mask dilated 5 px); grey; expo (linear stretch of the
whale's own luminance p2..p98 -> 16..240); clahe (CLAHE on L of Lab, clip 2, 8x8 tiles); bin (Otsu inside the
whale mask -> black / white, outside flat grey); and *_fill of grey / expo / clahe.
JOB res: sq896 (the box square at 896 px), partsq (448-px square around the part region), chart (the full-resolution
RAW cut-out parts_up/charts/<key>__fin|fluke.jpg at 448 px with its own whale / fin_band masks).
"""
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(REPO))
import bench  # noqa: E402
from model_fine_tuning.cluster.view_class import BUCKETS, shaped  # noqa: E402
from model_fine_tuning.common.backbone import IMAGENET_MEAN, IMAGENET_STD, load_backbone  # noqa: E402

CACHE = Path('/home/sat3737/NOAA_whales/moonshot_cache/pixels')
MEAN = np.array(IMAGENET_MEAN, np.float32) * 255
STD = np.array(IMAGENET_STD, np.float32) * 255
GREY = np.round(MEAN).astype(np.uint8)
LAYERS = [7, 11]
JOBS = {'base': ['rgb', 'fill'], 'norm': ['grey', 'expo', 'clahe', 'bin'], 'res': ['rgb']}
# (the first 9 norm days were cut with grey_fill / expo_fill / clahe_fill as well; dropped after the filled crops
#  lost on the bench in base: -0.02 to -0.04 AUC)
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def cut_rgb(im, cx, cy, w, h, W, H):
    c = im.crop((int(round(cx - w / 2)), int(round(cy - h / 2)), int(round(cx + w / 2)), int(round(cy + h / 2))))
    return np.asarray(c.resize((W, H), Image.BICUBIC if w < W else Image.LANCZOS))


def cut_mask(m, cx, cy, w, h, W, H):
    c = m.crop((int(round(cx - w / 2)), int(round(cy - h / 2)), int(round(cx + w / 2)), int(round(cy + h / 2))))
    return np.asarray(c.resize((W, H), Image.BILINEAR), np.float32) / 255.0


def lum(a):
    return a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114


def transform(a, t, m, md):
    """a uint8 HxWx3 crop, m whale fraction (pooling mask), md dilated whale fraction (fill mask)."""
    base, fill = t.replace('_fill', ''), t.endswith('_fill') or t == 'fill'
    a = a.astype(np.float32)
    inside = m > 0.5
    if base == 'grey':
        a = np.repeat(lum(a)[..., None], 3, 2)
    elif base == 'expo':
        L = lum(a)
        if inside.sum() >= 20:
            lo, hi = np.percentile(L[inside], [2, 98])
            g = 224.0 / max(hi - lo, 8.0)
            a = np.clip((a - lo) * g + 16, 0, 255)
    elif base == 'clahe':
        lab = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_RGB2LAB)
        lab[..., 0] = CLAHE.apply(lab[..., 0])
        a = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32)
    elif base == 'bin':
        L = lum(a)
        out = np.broadcast_to(GREY.astype(np.float32), a.shape).copy()
        if inside.sum() >= 20:
            v = L[inside].astype(np.uint8)
            thr, _ = cv2.threshold(v.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            out[inside] = np.where(L[inside] > thr, 255.0, 0.0)[:, None]
        return out
    if fill:
        a = a * md[..., None] + GREY.astype(np.float32) * (1 - md[..., None])
    return a


def tokw(m, W, H):
    """patch weights (H/16 * W/16,) = mean of the fraction map in each 16-px patch."""
    return m.reshape(H // 16, 16, W // 16, 16).mean((1, 3)).reshape(-1)


def inst_mask(M, box, pad=None):
    """photo mask (uint8 0/1, 2560 px) restricted to the instance box + 4 % -> PIL L image 0/255 (full size, sparse)."""
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    H, W = M.shape
    pad = int(round(0.04 * max(x1 - x0, y1 - y0))) if pad is None else pad
    x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad + 1), min(H, y1 + pad + 1)
    out = np.zeros_like(M)
    out[y0:y1, x0:x1] = M[y0:y1, x0:x1]
    return out


def part_region(seg, M, box, part):
    """part weight map (2560 px, uint8 0/1) or None. seg = parts_seg_r05_a full map (half size)."""
    x0, y0, x1, y1 = [int(round(v / 2)) for v in box]
    sub = seg[max(0, y0):y1 + 1, max(0, x0):x1 + 1]
    cls = 1 if part == 'fluke' else 2
    ys, xs = np.nonzero(sub == cls)
    if len(xs) < 4:
        return None, None
    ys, xs = (ys + max(0, y0)) * 2, (xs + max(0, x0)) * 2
    if part == 'fluke':
        full = np.zeros_like(M)
        full[ys, xs] = 1
        full = cv2.dilate(full, np.ones((5, 5), np.uint8))
        r = (xs.min() - 4, ys.min() - 4, xs.max() + 6, ys.max() + 6)
        return full, r
    fx0, fy0, fx1, fy1 = xs.min(), ys.min(), xs.max() + 2, ys.max() + 2
    h, w = max(fy1 - fy0, 8), max(fx1 - fx0, 8)
    half = max(w, 2.7 * h)                      # fin + hump: about twice the fin base (base ~ 2.7 fin heights)
    cx = (fx0 + fx1) / 2
    r = (cx - half, fy0 - 0.2 * h, cx + half, fy1 + 1.0 * h)
    full = np.zeros_like(M)
    rr = [int(round(v)) for v in r]
    full[max(0, rr[1]):max(0, rr[3]), max(0, rr[0]):max(0, rr[2])] = 1
    return full * M, r


class Day(torch.utils.data.Dataset):
    def __init__(self, day, job):
        self.day, self.job = day, job
        parts = defaultdict(set)
        for p in bench.PARTS:
            for k, c, a in bench.instances(day, p):
                parts[k].add(p)
        self.parts = parts
        by = defaultdict(list)
        for k in sorted(parts):
            by[k.rsplit('~', 1)[0]].append(k)
        self.stems = sorted(by)
        self.by = by
        self.dd = bench.DAYS / day
        self.chart_ok = (self.dd / 'parts_up' / 'charts').exists()

    def __len__(self):
        return len(self.stems)

    def _load_mask(self, p):
        if not p.exists():
            return None
        return (np.asarray(Image.open(p)) > 0).astype(np.uint8)

    def __getitem__(self, j):
        stem = self.stems[j]
        im = Image.open(self.dd / 'images' / f'{stem}.jpg').convert('RGB')
        Mc = self._load_mask(self.dd / 'canon' / f'{stem}_mask.png')
        Mi = self._load_mask(self.dd / 'instances_v2' / 'masks' / f'{stem}.png')
        sp = self.dd / 'parts_seg_r05_a' / 'full' / f'{stem}.png'
        seg = np.asarray(Image.open(sp)) if sp.exists() else None
        if Mc is None:
            Mc = np.zeros((im.size[1], im.size[0]), np.uint8)
        if Mi is None or Mi.shape != Mc.shape:
            Mi = Mc
        out = []
        for key in self.by[stem]:
            b = bench.box(self.day, key)
            mc, mi = inst_mask(Mc, b), inst_mask(Mi, b)
            mcd = cv2.dilate(mc, np.ones((11, 11), np.uint8))                    # 5 px either way at 2560 px
            pmaps = {}
            for p in self.parts[key]:
                pm, reg = part_region(seg, mc, b, p) if seg is not None else (None, None)
                pmaps[p] = (pm, reg)
            L = {n: Image.fromarray(x * 255) for n, x in (('mc', mc), ('mi', mi), ('mcd', mcd))}
            for p, (pm, reg) in pmaps.items():
                L['p_' + p] = Image.fromarray(pm * 255) if pm is not None else None
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            s = max(max(b[2] - b[0], b[3] - b[1]) * 1.15, 64.0)
            k, w, h = shaped(b)
            geoms = []
            if self.job in ('base', 'norm'):
                geoms = [('sq', cx, cy, s, s, 448, 448), ('ar', cx, cy, w, h, BUCKETS[k][1], BUCKETS[k][0])]
            else:
                geoms = [('sq896', cx, cy, s, s, 896, 896)]
                for p, (pm, reg) in pmaps.items():
                    if reg is not None:
                        ps = max(reg[2] - reg[0], reg[3] - reg[1]) * 1.15
                        ps = max(ps, 48.0)
                        geoms.append((f'partsq_{p}', (reg[0] + reg[2]) / 2, (reg[1] + reg[3]) / 2, ps, ps, 448, 448))
            for g, gx, gy, gw, gh, W, H in geoms:
                a = cut_rgb(im, gx, gy, gw, gh, W, H)
                ms = {n: (cut_mask(v, gx, gy, gw, gh, W, H) if v is not None else None) for n, v in L.items()}
                wts = {'wc': tokw(ms['mc'], W, H), 'wi': tokw(ms['mi'], W, H)}
                for p in self.parts[key]:
                    wts['part_' + p] = tokw(ms['p_' + p], W, H) if ms.get('p_' + p) is not None else wts['wc']
                for t in JOBS[self.job]:
                    x = transform(a, t, ms['mc'], ms['mcd'])
                    out.append((key, f'{g}|{t}', ((x - MEAN) / STD).transpose(2, 0, 1).astype(np.float16), wts))
            if self.job == 'res' and self.chart_ok:
                for p in self.parts[key]:
                    pn = 'fin' if p == 'dorsal' else 'fluke'
                    cj = self.dd / 'parts_up' / 'charts' / f'{key}__{pn}.jpg'
                    cz = self.dd / 'parts_up' / f'{key}__{pn}.npz'
                    if not (cj.exists() and cz.exists()):
                        continue
                    z = np.load(cz)
                    a = np.asarray(Image.open(cj).convert('RGB').resize((448, 448), Image.LANCZOS))
                    wh = np.asarray(Image.fromarray(z['whale'].astype(np.uint8) * 255).resize((448, 448), Image.BILINEAR), np.float32) / 255
                    if pn == 'fin':
                        pr = np.asarray(Image.fromarray((z['fin_band'] & z['whale']).astype(np.uint8) * 255).resize((448, 448), Image.BILINEAR), np.float32) / 255
                    else:
                        pr = wh
                    whd = cv2.dilate((wh > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
                    wts = {'wc': tokw(wh, 448, 448), 'wi': tokw(wh, 448, 448), 'part_' + p: tokw(pr, 448, 448)}
                    for t in JOBS[self.job]:
                        x = transform(a, t, wh, whd)
                        out.append((key, f'chart_{p}|{t}', ((x - MEAN) / STD).transpose(2, 0, 1).astype(np.float16), wts))
        return out


def run(day, job, model):
    ds = Day(day, job)
    dl = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=8, collate_fn=lambda b: [x for l in b for x in l])
    res = defaultdict(dict)                      # name -> key -> vector
    pend = defaultdict(list)
    BS = {448 * 448: 48, 896 * 896: 12}

    def flush(shape, force=False):
        bs = BS.get(shape[0] * shape[1], 32)
        while pend[shape] and (force or len(pend[shape]) >= bs):
            chunk, pend[shape] = pend[shape][:bs], pend[shape][bs:]
            x = torch.from_numpy(np.stack([c[2] for c in chunk])).cuda()
            with torch.no_grad():
                outs = model.get_intermediate_layers(x, n=LAYERS, reshape=False, return_class_token=True, norm=True)
            for li, (pt, cl) in zip(LAYERS, outs):
                pt = pt.float()
                cl = cl.float().cpu().numpy()
                mean = pt.mean(1).cpu().numpy()
                for r, (key, vn, _, wts) in enumerate(chunk):
                    pre = f'{vn}|{li}'
                    res[pre + '|cls'][key] = cl[r]
                    res[pre + '|mean'][key] = mean[r]
                    for wn, wv in wts.items():
                        wt = torch.from_numpy(wv).cuda()
                        sw = float(wt.sum())
                        res[pre + '|' + wn][key] = ((wt[:, None] * pt[r]).sum(0) / sw).cpu().numpy() if sw >= 0.5 else mean[r]

    for batch in dl:
        for key, vn, x, wts in batch:
            pend[x.shape[1:]].append((key, vn, x, wts))
        for sh in list(pend):
            flush(sh)
    for sh in list(pend):
        flush(sh, True)
    out = {}
    for name, d in res.items():
        keys = sorted(d)
        out[name + '#keys'] = np.array(keys)
        out[name] = np.stack([d[k] for k in keys]).astype(np.float16)
    (CACHE / job).mkdir(parents=True, exist_ok=True)
    np.savez(CACHE / job / f'{day}.npz', **out)


def main():
    days = sys.argv[2:] or (bench.TRAIN + bench.VAL)
    model = load_backbone('vitb16').cuda().eval().half()
    t0 = time.time()
    jobs = sys.argv[1].split(',')
    for job, d in [(j, d) for j in jobs for d in days]:
        if (CACHE / job / f'{d}.npz').exists():
            continue
        if time.time() - t0 > float(os.environ.get('MAXSEC', 1000)):
            print('time budget spent, stopping', flush=True)
            break
        t = time.time()
        run(d, job, model)
        print(f'{job} {d} {time.time() - t:.0f} s  peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB', flush=True)
    print(f'{sys.argv[1]} done {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
