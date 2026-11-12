"""Richer frozen DINOv3 ViT-B/16 features of every whale instance, for the view-class experiment:
    python experiments/view_class/vc_feats.py [DAY ...]      -> experiments/view_class/feats/DAY.npz
  sq   the square crop on the box (x 1.15), 448 px (inst_feat.npz uses 224): cls, mean / max over the patch tokens of
       the last block, mean of block 8
  ar   a crop that keeps the box's shape (1:1, 3:2, 2:1 or 3:1, landscape or portrait; ~800 patches): cls, mean, max
A wide back is 448 px wide in the square crop and 672 px in the shaped one (224-px crop: 224)."""
import json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model_fine_tuning.common.backbone import IMAGENET_MEAN, IMAGENET_STD, load_backbone

ROOT = Path(__file__).resolve().parents[1] / 'days_all'; OUT = Path(__file__).resolve().parent / 'feats'
BUCKETS = [(448, 448), (384, 576), (336, 672), (224, 672), (576, 384), (672, 336)]          # (H, W)
MEAN = np.array(IMAGENET_MEAN, np.float32) * 255; STD = np.array(IMAGENET_STD, np.float32) * 255


def cut(im, cx, cy, w, h, W, H):
    c = im.crop((int(round(cx - w / 2)), int(round(cy - h / 2)), int(round(cx + w / 2)), int(round(cy + h / 2))))        # PIL pads outside the frame with black
    return ((np.asarray(c.resize((W, H), Image.BICUBIC if w < W else Image.LANCZOS), np.float32) - MEAN) / STD).transpose(2, 0, 1)


def shaped(b):
    x0, y0, x1, y1 = b; w, h = (x1 - x0) * 1.15, (y1 - y0) * 1.15; w, h = max(w, 64.0), max(h, 64.0); a = np.log(w / h)
    k = int(np.argmin([abs(a - np.log(W / H)) for H, W in BUCKETS])); H, W = BUCKETS[k]
    if w / h < W / H: w = h * W / H
    else: h = w * H / W
    return k, w, h


class Photos(torch.utils.data.Dataset):
    def __init__(self, day, items): self.day, self.items = day, items
    def __len__(self): return len(self.items)
    def __getitem__(self, j):
        stem, insts = self.items[j]; im = Image.open(ROOT / self.day / 'images' / f'{stem}.jpg').convert('RGB'); out = []
        for i, b in insts:
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2; s = max(max(b[2] - b[0], b[3] - b[1]) * 1.15, 64.0); k, w, h = shaped(b)
            out.append((i, cut(im, cx, cy, s, s, 448, 448), k, cut(im, cx, cy, w, h, BUCKETS[k][1], BUCKETS[k][0])))
        return out


def run(day, model):
    P = json.loads((ROOT / day / 'safe' / 'photos.json').read_text()); keys = []; items = []
    for stem, p in P.items():
        ins = []
        for it in p['instances']:
            ins.append((len(keys), it['bbox'])); keys.append(it['key'])
        if ins: items.append((stem, ins))
    n = len(keys); F = {k: np.zeros((n, 768), np.float16) for k in ('sq_cls', 'sq_mean', 'sq_max', 'sq_mid', 'ar_cls', 'ar_mean', 'ar_max')}
    dl = torch.utils.data.DataLoader(Photos(day, items), batch_size=8, num_workers=8, collate_fn=lambda b: [x for l in b for x in l])
    pend = defaultdict(list)

    def flush(k, force=False):
        while pend[k] and (force or len(pend[k]) >= 48):
            chunk, pend[k] = pend[k][:48], pend[k][48:]
            x = torch.from_numpy(np.stack([c[1] for c in chunk])).cuda().half(); idx = [c[0] for c in chunk]
            with torch.no_grad():
                (p8, _), (pt, cl) = model.get_intermediate_layers(x, n=[8, 11], reshape=False, return_class_token=True, norm=True)
            pre = 'sq' if k == 'sq' else 'ar'
            F[pre + '_cls'][idx] = cl.float().cpu().numpy(); F[pre + '_mean'][idx] = pt.float().mean(1).cpu().numpy(); F[pre + '_max'][idx] = pt.float().amax(1).cpu().numpy()
            if k == 'sq': F['sq_mid'][idx] = p8.float().mean(1).cpu().numpy()
    for batch in dl:
        for i, sq, k, ar in batch:
            pend['sq'].append((i, sq)); pend[k].append((i, ar))
        for k in list(pend): flush(k)
    for k in list(pend): flush(k, True)
    tmp = OUT / f'{day}.tmp.npz'; np.savez(tmp, keys=np.array(keys), **F); tmp.replace(OUT / f'{day}.npz')
    return n


if __name__ == '__main__':
    days = sys.argv[1:] or sorted(d.name for d in ROOT.iterdir() if (d / 'safe' / 'clusters.json').exists())
    model = load_backbone('vitb16').cuda().eval().half(); t = time.time()
    for d in days:
        if (OUT / f'{d}.npz').exists(): continue
        t0 = time.time(); n = run(d, model); print(f'{d}: {n} instances in {time.time() - t0:.0f} s', flush=True)
    print(f'done in {time.time() - t:.0f} s')
