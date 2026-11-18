"""Per-point probability maps of the pose-side head B_both for the bench's fluke instances on the train days (val days
already exist in pose_side/maps_Bboth_val), so the fluke tips can be decoded with two peaks (pose_side.decode_tips).
GPU, under the lock.  python fluke_maps.py"""
import sys

import torch

sys.path.insert(0, '/home/sat3737/NOAA_whales/Dinov3-WhaleID')
from common import CACHE, POSE, bench  # noqa: E402
from model_fine_tuning.pose_side.predict import run  # noqa: E402

if __name__ == '__main__':
    out = CACHE / 'fluke_maps'; out.mkdir(parents=True, exist_ok=True)
    have = {p.stem for p in (POSE / 'maps_Bboth_val').glob('*.npz')}
    items = [(d, k) for d in bench.TRAIN for k, c, a in bench.instances(d, 'fluke') if k not in have and not (out / f'{k}.npz').exists()]
    print(len(items), 'fluke instances', flush=True)
    import os
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', 8)))
    run(POSE / 'B_both' / 'head.pt', items, torch.device(os.environ.get('ALIGNED_DEV', 'cuda')), out)
    print('done', flush=True)
