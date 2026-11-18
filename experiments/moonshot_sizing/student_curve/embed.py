"""Fin and fluke fingerprints of every parts_up cut-out of some days with one identity checkpoint (student_curve).

Same preprocessing as production / model_fine_tuning/cutouts_v3/embed.py (jpg -> 512 px INTER_AREA -> WhaleIdModel.embed
in bfloat16 -> unit vector), but for EVERY key with a DAY/parts_up/charts/<key>__<part>.jpg (not only the assistant's
points), so the bench sees all instances that have the cut-out.

    flock /var/tmp/sat3737-parts-cache/gpu_train.lock env PYTHONPATH=. .venv/bin/python \
        experiments/moonshot_sizing/student_curve/embed.py --net NAME=checkpoint --days d1,d2,...
-> /home/sat3737/NOAA_whales/moonshot_cache/student_curve/emb/NAME/<day>.npz  (fin_keys, fin_E, fluke_keys, fluke_E)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path('/home/sat3737/NOAA_whales/Dinov3-WhaleID')
sys.path.insert(0, str(ROOT))
DAYS = ROOT / 'experiments' / 'days_all'
OUT = Path('/home/sat3737/NOAA_whales/moonshot_cache/student_curve/emb')
EVAL = set(json.load(open(ROOT / 'model_fine_tuning' / 'splits' / 'days.json'))['eval'])
NET = 512


class Files(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        im = cv2.cvtColor(cv2.imread(str(self.files[i])), cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (NET, NET), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(np.ascontiguousarray(im.transpose(2, 0, 1))).float() / 255.0


@torch.no_grad()
def embed_files(model, files, part_idx, dev, bs=32, workers=6):
    out = []
    for x in DataLoader(Files(files), batch_size=bs, num_workers=workers):
        x = x.to(dev)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            e = model.embed(x, torch.full((len(x),), part_idx, dtype=torch.long, device=dev)).float()
        out.append(torch.nn.functional.normalize(e, dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 512), np.float32)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--net', required=True); ap.add_argument('--days', required=True)
    a = ap.parse_args()
    from model_fine_tuning.cluster.train_id import PARTS, WhaleIdModel
    name, ck = a.net.split('=', 1)
    days = [d for d in a.days.split(',') if d]
    bad = [d for d in days if d in EVAL]
    if bad:
        raise SystemExit(f'eval days refused: {bad}')
    dev = torch.device('cuda'); model, _ = WhaleIdModel.from_checkpoint(ck, dev); model.eval()
    for day in days:
        t0 = time.time(); res = {}
        ch = DAYS / day / 'parts_up' / 'charts'
        for part in ('fin', 'fluke'):
            files = sorted(ch.glob(f'*__{part}.jpg'))
            keys = [f.name[: -len(f'__{part}.jpg')] for f in files]
            res[f'{part}_keys'] = np.array(keys); res[f'{part}_E'] = embed_files(model, files, PARTS.index(part), dev).astype(np.float16)
        f = OUT / name / f'{day}.npz'; f.parent.mkdir(parents=True, exist_ok=True)
        np.savez(f.with_suffix('.tmp.npz'), **res); f.with_suffix('.tmp.npz').replace(f)
        print(f"{name} {day}: fin {len(res['fin_keys'])}, fluke {len(res['fluke_keys'])} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == '__main__':
    main()
