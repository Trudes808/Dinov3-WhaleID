"""Canonical tiles -> frozen DINOv3 ViT-B/16 patch tokens (GPU, run under the GPU lock).

    python extract.py dorsal [days...]
Writes CACHE/<part>/<day>.npz: keys, tok11 (N, gh, gw, 768) fp16 (last block, normed), tok8 (N, gh/2, gw/2, 768) fp16
(block 8, 2x2 mean), mask (N, TH, TW) uint8 whale mask in the tile, inside (N, gh, gw) share of the cell inside the
photo, grey (N, TH, TW) uint8, info json.
"""
import json
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, '/home/sat3737/NOAA_whales/Dinov3-WhaleID')
from common import CACHE, TH, TW, bench, dorsal_frame, warp  # noqa: E402
import common  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)


def frame_fn(part):
    if part == 'dorsal':
        return dorsal_frame, (TW, TH)
    import fluke
    return fluke.fluke_frame, (fluke.TW, fluke.TH)


class DS(Dataset):
    def __init__(self, part, items):
        self.part, self.items = part, items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        d, k = self.items[i]
        fn, size = frame_fn(self.part)
        fr = fn(k)
        if fr is None:
            return None
        M, info = fr
        t, m, inside = warp(d, k, M, size)
        x = ((t.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)
        grey = cv2.cvtColor(t, cv2.COLOR_RGB2GRAY)
        return k, torch.from_numpy(x), m, inside, grey, info


def collate(b):
    return [r for r in b if r is not None]


def main(part, days):
    from model_fine_tuning.common.backbone import load_backbone
    import os
    dev = torch.device(os.environ.get('ALIGNED_DEV', 'cuda'))
    if dev.type == 'cpu':
        torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', 8)))
    vit = load_backbone('vitb16').eval().to(dev)
    out_dir = CACHE / part; out_dir.mkdir(parents=True, exist_ok=True)
    for day in days:
        if (out_dir / f'{day}.npz').exists():
            continue
        items = [(day, k) for k, c, a in bench.instances(day, part)]
        dl = DataLoader(DS(part, items), batch_size=24, num_workers=4, collate_fn=collate)
        K, T11, T8, MS, INS, GR, INFO = [], [], [], [], [], [], {}
        with torch.no_grad():
            for b in dl:
                if not b:
                    continue
                x = torch.stack([r[1] for r in b]).to(dev)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
                    f8, f11 = vit.get_intermediate_layers(x, n=[8, 11], reshape=True, norm=True)
                f8 = torch.nn.functional.avg_pool2d(f8.float(), 2, ceil_mode=True)
                T11.append(f11.float().permute(0, 2, 3, 1).cpu().numpy().astype(np.float16))
                T8.append(f8.permute(0, 2, 3, 1).cpu().numpy().astype(np.float16))
                gh, gw = f11.shape[-2:]
                for r in b:
                    K.append(r[0]); MS.append(r[2]); GR.append(r[4]); INFO[r[0]] = r[5]
                    INS.append(cv2.resize((r[3] > 0).astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA))
        np.savez(out_dir / f'{day}.npz', keys=np.array(K), tok11=np.concatenate(T11), tok8=np.concatenate(T8), mask=np.stack(MS),
                 inside=np.stack(INS).astype(np.float16), grey=np.stack(GR))
        json.dump(INFO, open(out_dir / f'{day}_info.json', 'w'))
        print(day, len(K), flush=True)


if __name__ == '__main__':
    part = sys.argv[1]
    days = sys.argv[2:] or bench.TRAIN + bench.VAL
    main(part, days)
