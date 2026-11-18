"""Embed the dorsal instances of the 8 test days (bench.VAL + hold of bench.splits()[0]) with HW-b6, raw or fine-tuned.
Crop exactly as public_reid/extract.py 'box' (square box x 1.15 from the 2560-px image -> 1024 px, ImageNet mean / std);
forward in bfloat16 autocast as in training.

    flock LOCK env OMP_NUM_THREADS=8 .venv/bin/python experiments/moonshot_sizing/fin_tune/embed.py --name raw
    ... embed.py --name s5_0 --ckpt CACHE/runs/s5_0/final.pt
-> CACHE/emb/<name>/<day>.npz (keys, X float32 = the BatchNorm embedding, not normalised)
"""
import argparse
import time

import numpy as np
import torch

from common import CACHE, MEAN, RES, STD, bench, crop, split
from extract import HappyWhaleB6


class Test(torch.utils.data.Dataset):
    def __init__(self, day, keys):
        self.day, self.keys = day, keys

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        return torch.from_numpy(np.asarray(crop(self.day, self.keys[i], 1.15, RES), np.uint8).copy()).permute(2, 0, 1)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--ckpt')
    ap.add_argument('--days', default='test')
    ap.add_argument('--bs', type=int, default=16)
    a = ap.parse_args()
    hold, pool, test = split()
    days = test if a.days == 'test' else a.days.split(',')
    dev = torch.device('cuda')
    torch.cuda.set_per_process_memory_fraction(min(1.0, 10 * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    m = HappyWhaleB6()
    if a.ckpt:
        m.load_state_dict(torch.load(a.ckpt, map_location='cpu', weights_only=True))
    m = m.to(dev).eval()
    mean, std = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1), torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    t0 = time.time()
    for d in days:
        keys = sorted(k for k, c, an in bench.instances(d, 'dorsal'))
        out = []
        for x in torch.utils.data.DataLoader(Test(d, keys), batch_size=a.bs, num_workers=8):
            x = (x.to(dev).float() / 255. - mean) / std
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out.append(m(x).float().cpu().numpy())
        p = CACHE / 'emb' / a.name / f'{d}.npz'
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(p)[:-4] + '.tmp.npz', keys=np.array(keys), X=np.concatenate(out))
        (p.parent / f'{d}.tmp.npz').replace(p)
        print(f'{a.name} {d}: {len(keys)} crops, {time.time() - t0:.0f} s, peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB', flush=True)


if __name__ == '__main__':
    main()
