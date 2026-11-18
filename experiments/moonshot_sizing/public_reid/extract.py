"""Public re-identification embeddings (MiewID msv3, MegaDescriptor-L-384) of the bench's fluke / dorsal instances.

    flock /var/tmp/sat3737-parts-cache/gpu_train.lock env OMP_NUM_THREADS=8 .venv/bin/python \
        experiments/moonshot_sizing/public_reid/extract.py [--days all|val|D1,D2] [--models miewid,mega]

Crops (cut once at 448 px uint8 in the loader, resized on the GPU to each model's input, then normalised as each model's
card says: MiewID ImageNet mean / std at 440, MegaDescriptor 0.5 / 0.5 at 384):
  box       square crop of the instance box x 1.15 from DAY/images (2560 px), as BASE's "sq" crop (outside = black)
  boxm      the same with every pixel outside the photo's whale mask (canon/<stem>_mask.png) set to a flat fill
  fin       parts_up/charts/<key>__fin.jpg (dorsal instances)          finm   same, outside the cut-out's whale mask filled
  fluke     parts_up/charts/<key>__fluke.jpg (fluke instances; the chart convention is peduncle UP)
  flukem    same, masked                                               flukenat  the fluke chart flipped vertically
                                                                                 (peduncle down, blades up: how fluke
                                                                                 catalogues usually show a fluke)
Flat fill = the value that normalises to zero for the model (its mean colour), applied after the resize.
Output: CACHE/feat/<model>/<variant>/<day>.npz with keys (instance keys) and X (float16, raw embedding, not normalised).
MiewID is rebuilt here from timm (efficientnetv2_rw_m, GeM, BatchNorm1d 2152) and its safetensors: the hub's
remote code is not executed.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import bench  # noqa: E402

CACHE = Path('/home/sat3737/NOAA_whales/moonshot_cache/public_reid')
MIEW = CACHE / 'models' / 'miewid-msv3' / 'model.safetensors'
HW = CACHE / 'models' / 'hw_b6_1024' / 'checkpoints' / 'last_checkpoint.pth'
SKIPM = False
S = 448                                                                  # crop size in the loader (--size changes it)
CHART = {'dorsal': 'fin', 'fluke': 'fluke'}


def items_of(day):
    """[(key, kind)] kind in box / fin / fluke for the instances the bench uses."""
    keys = {}
    for part in bench.PARTS:
        for key, c, a in bench.instances(day, part):
            keys.setdefault(key, set()).add(part)
    out = [(k, 'box') for k in sorted(keys)]
    for k in sorted(keys):
        for part in sorted(keys[k]):
            if (bench.DAYS / day / 'parts_up' / 'charts' / f'{k}__{CHART[part]}.jpg').exists():
                out.append((k, CHART[part]))
    return out


class Crops(torch.utils.data.Dataset):
    def __init__(self, day, items):
        self.day, self.items = day, items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        key, kind = self.items[i]
        d = bench.DAYS / self.day
        if kind == 'box':
            im = Image.open(bench.image_path(self.day, key)).convert('RGB')
            stem = key.rsplit('~', 1)[0]
            mp = d / 'canon' / f'{stem}_mask.png'
            mask = Image.open(mp).convert('L').point(lambda v: 255 if v else 0) if mp.exists() else None
            if mask is not None and mask.size != im.size:
                mask = mask.resize(im.size, Image.NEAREST)
            b = bench.box(self.day, key)
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            s = max(max(b[2] - b[0], b[3] - b[1]) * 1.15, 64.0)
            r = (int(round(cx - s / 2)), int(round(cy - s / 2)), int(round(cx + s / 2)), int(round(cy + s / 2)))
            flt = Image.BICUBIC if s < S else Image.LANCZOS
            x = np.asarray(im.crop(r).resize((S, S), flt), np.uint8)
            m = np.asarray(mask.crop(r).resize((S, S), Image.BILINEAR), np.uint8) >= 128 if mask is not None else np.ones((S, S), bool)
            return key, kind, x, m, mask is not None
        im = Image.open(d / 'parts_up' / 'charts' / f'{key}__{kind}.jpg').convert('RGB')
        z = np.load(d / 'parts_up' / f'{key}__{kind}.npz')
        x = np.asarray(im.resize((S, S), Image.LANCZOS), np.uint8)
        m = np.asarray(Image.fromarray(z['whale'].astype(np.uint8) * 255).resize((S, S), Image.BILINEAR), np.uint8) >= 128
        return key, kind, x, m, True


def collate(b):
    return b


class GeM(torch.nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p, self.eps = torch.nn.Parameter(torch.ones(1) * p), eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1. / self.p)


class MiewID(torch.nn.Module):
    """Same layers and forward as the hub's modeling_miewid.MiewIdNet (extract_feat), without the training heads."""

    def __init__(self):
        super().__init__()
        import timm
        from safetensors.torch import load_file
        self.backbone = timm.create_model('efficientnetv2_rw_m', pretrained=False, num_classes=0)
        self.backbone.global_pool = GeM()
        self.bn = torch.nn.BatchNorm1d(2152)
        sd = {k: v for k, v in load_file(str(MIEW)).items() if not k.startswith('final.')}
        self.load_state_dict(sd, strict=True)

    def forward(self, x):
        return self.bn(self.backbone(x).view(x.shape[0], -1))


class HappyWhaleB6(torch.nn.Module):
    """liu-peilin/happywhale_b6_pseudo_charm_round2_1024_v2 (a 2026 re-run of the Happywhale 2022 1st-place recipe:
    tf_efficientnet_b6_ns, 1024 px, GeM, BatchNorm embedding, sub-centre ArcFace id + species heads). Rebuilt from the
    checkpoint's tensors (torch.load weights_only=True); embedding = BatchNorm1d(GeM(features)), heads dropped."""

    def __init__(self):
        super().__init__()
        import timm
        ck = torch.load(str(HW), map_location='cpu', weights_only=True)['model']
        self.backbone = timm.create_model('tf_efficientnet_b6.ns_jft_in1k', pretrained=False, num_classes=0, global_pool='')
        self.backbone.load_state_dict({k[9:]: v for k, v in ck.items() if k.startswith('backbone.')}, strict=True)
        self.pool = GeM()
        self.pool.load_state_dict({'p': ck['pool.p']})
        self.bn = torch.nn.BatchNorm1d(2304)
        self.bn.load_state_dict({k[12:]: v for k, v in ck.items() if k.startswith('embedding.0.')})
        assert not [k for k in ck if k.startswith('embedding.') and not k.startswith('embedding.0.')]

    def forward(self, x):
        return self.bn(self.pool(self.backbone.forward_features(x)).flatten(1))


def load_models(names, dev):
    out = {}
    for n in names:
        if n == 'miewid':
            m, res, mean, std = MiewID(), 440, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        elif n == 'hw_b6':
            m, res, mean, std = HappyWhaleB6(), 1024, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        elif n == 'mega':
            import timm
            os.environ.setdefault('HF_HUB_OFFLINE', '1')
            m, res, mean, std = timm.create_model('hf-hub:BVRA/MegaDescriptor-L-384', pretrained=True, num_classes=0), 384, (0.5,) * 3, (0.5,) * 3
        else:
            raise ValueError(n)
        out[n] = (m.to(dev).eval().half(), res, torch.tensor(mean, device=dev).view(1, 3, 1, 1), torch.tensor(std, device=dev).view(1, 3, 1, 1))
    return out


@torch.no_grad()
def run(day, models, dev, bs, workers):
    items = items_of(day)
    feats = {}                                                          # (model, variant) -> {key: vec}
    dl = torch.utils.data.DataLoader(Crops(day, items), batch_size=bs, num_workers=workers, collate_fn=collate)
    nomask = 0
    for batch in dl:
        x = torch.from_numpy(np.stack([b[2] for b in batch])).to(dev).permute(0, 3, 1, 2).float() / 255.
        m = torch.from_numpy(np.stack([b[3] for b in batch])).to(dev)[:, None].float()
        nomask += sum(not b[4] for b in batch)
        for name, (model, res, mean, std) in models.items():
            xr = F.interpolate(x, size=(res, res), mode='bilinear', antialias=True, align_corners=False)
            mr = F.interpolate(m, size=(res, res), mode='bilinear', align_corners=False) >= 0.5
            xn = (xr - mean) / std
            variants = {'': xn} if SKIPM else {'': xn, 'm': xn * mr}
            for suf, inp in variants.items():
                e = model(inp.half()).float().cpu().numpy()
                for (key, kind, _, _, hasmask), v in zip(batch, e):
                    if suf == 'm' and not hasmask:
                        continue
                    feats.setdefault((name, kind + suf), {})[key] = v
            fl = [i for i, b in enumerate(batch) if b[1] == 'fluke']
            if fl:
                e = model(xn[fl].flip(2).half()).float().cpu().numpy()
                for i, v in zip(fl, e):
                    feats.setdefault((name, 'flukenat'), {})[batch[i][0]] = v
    for (name, var), d in feats.items():
        p = CACHE / 'feat' / name / var / f'{day}.npz'
        p.parent.mkdir(parents=True, exist_ok=True)
        ks = sorted(d)
        np.savez(p, keys=np.array(ks), X=np.stack([d[k] for k in ks]).astype(np.float16))
    return len(items), nomask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', default='all')
    ap.add_argument('--models', default='miewid,mega')
    ap.add_argument('--bs', type=int, default=24)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--size', type=int, default=448)
    ap.add_argument('--skip-masked', action='store_true')
    ap.add_argument('--max-minutes', type=float, default=17.0, help='stop starting new days after this long')
    a = ap.parse_args()
    global S, SKIPM
    S, SKIPM = a.size, a.skip_masked
    days = bench.VAL + bench.TRAIN if a.days == 'all' else bench.VAL if a.days == 'val' else a.days.split(',')
    dev = torch.device('cuda')
    models = load_models(a.models.split(','), dev)
    t0 = time.time()
    for d in days:
        if all((CACHE / 'feat' / n / 'box' / f'{d}.npz').exists() for n in models):
            continue
        if time.time() - t0 > a.max_minutes * 60:
            print('time budget used, stopping before', d, flush=True)
            break
        n, nm = run(d, models, dev, a.bs, a.workers)
        print(f'{d}: {n} crops ({nm} without a whale mask), {time.time() - t0:.0f} s, peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB', flush=True)


if __name__ == '__main__':
    main()
