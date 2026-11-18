"""Dense matcher: EfficientLoFTR (HF zju-community/efficientloftr) on the part charts, matches kept only where both ends
fall on (dilated) whale pixels, then the same RANSAC verification as lm_match.py.

    flock ... python lm_dense.py SIZE NPICK PART DAY [DAY ...]
Instances = the first NPICK of the ALIKED run's picks per cluster (largest / sharpest). Writes CACHE/eloftrSIZE/DAY__PART__pairs.npz.
"""
import os
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lm_lib as LM  # noqa: E402
from lm_lib import bench  # noqa: E402
from lm_match import STATS, _verify  # noqa: E402

BATCH = 8
HF = '/home/sat3737/NOAA_whales/moonshot_cache/local_match/models/hf/models--zju-community--efficientloftr/snapshots/face1a79050ffa3e9da28720d1cf93aaf2e8f421'


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    size, npick, part, days = int(args[0]), int(args[1]), args[2], args[3:]
    from transformers import EfficientLoFTRForKeypointMatching as AutoModel
    dev = torch.device(os.environ.get('LM_DEV', 'cuda'))
    model = AutoModel.from_pretrained(HF).to(dev).eval()
    out_dir = LM.CACHE / f'eloftr{size}'
    out_dir.mkdir(parents=True, exist_ok=True)
    pool = Pool(8)
    k = max(1, int(round(6 * size / 1024)))
    ker = np.ones((2 * k + 1, 2 * k + 1), np.uint8)
    for day in days:
        fp = out_dir / f'{day}__{part}__pairs.npz'
        if fp.exists():
            print(day, 'cached'); continue
        t = time.time()
        picked = torch.load(LM.CACHE / 'aliked1024' / f'{day}__{part}.pt', weights_only=False)['picked']
        picked = {c: v[:npick] for c, v in picked.items()}
        keys = sorted({x for v in picked.values() for x in v})
        kidx = {kk: i for i, kk in enumerate(keys)}
        img, msk, area = {}, {}, {}
        for kk in keys:
            im, wh, _ = LM.load_chart(day, kk, part, size)
            g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            img[kk] = g
            msk[kk] = cv2.dilate(wh.astype(np.uint8), ker) > 0
            area[kk] = float(wh.sum())
        jobs = []
        for a, b, same in bench.pairs(day, part):
            for ka in picked.get(a, []):
                for kb in picked.get(b, []):
                    jobs.append((a, b, kidx[ka], kidx[kb], 0))
                    if part == 'dorsal' and '--no-mirror' not in sys.argv:
                        jobs.append((a, b, kidx[ka], kidx[kb], 1))
        tasks = []
        with torch.no_grad():
            for s in range(0, len(jobs), BATCH):
                bj = jobs[s:s + BATCH]
                A = [img[keys[j[2]]] for j in bj]
                Bm = [img[keys[j[3]]][:, ::-1] if j[4] else img[keys[j[3]]] for j in bj]
                x = torch.from_numpy(np.stack([np.stack([a, b]) for a, b in zip(A, Bm)])[:, :, None].repeat(3, 2).copy()).to(dev)
                out = model(pixel_values=x)
                kp = out.keypoints.cpu().numpy() * size                  # (B, 2, N, 2)
                ms = out.matching_scores.cpu().numpy()                   # (B, 2, N)
                mv = out.matches.cpu().numpy() > -1
                for r, j in enumerate(bj):
                    sc = ms[r, 0]
                    ok = (sc > 0.2) & mv[r, 0] & mv[r, 1]
                    pa, pb = kp[r, 0][ok], kp[r, 1][ok]
                    mb = msk[keys[j[3]]][:, ::-1] if j[4] else msk[keys[j[3]]]
                    ia = np.clip(pa.round().astype(int), 0, size - 1); ib = np.clip(pb.round().astype(int), 0, size - 1)
                    on = msk[keys[j[2]]][ia[:, 1], ia[:, 0]] & mb[ib[:, 1], ib[:, 0]]
                    tasks.append((pa[on].astype(np.float32), pb[on].astype(np.float32), sc[ok][on].astype(np.float32), area[keys[j[2]]], size))
        tg = time.time() - t
        rows = pool.map(_verify, tasks, chunksize=32)
        off = np.cumsum([0] + [len(t[0]) for t in tasks])
        np.savez_compressed(str(fp).replace('__pairs.npz', '__matches.npz'), off=off, pa=np.concatenate([t[0] for t in tasks]).reshape(-1, 2),
                            pb=np.concatenate([t[1] for t in tasks]).reshape(-1, 2), sc=np.concatenate([t[2] for t in tasks]))
        torch.save(dict(size=size, picked=picked, feats={kk: dict(plain=dict(kp=np.zeros((1000, 2)))) for kk in keys}), out_dir / f'{day}__{part}.pt')
        np.savez_compressed(fp, jobs=np.array(jobs, np.int64).reshape(-1, 5), stats=np.array(rows, np.float32).reshape(len(jobs), -1),
                            keys=np.array(keys), names=np.array([f'{kk}@{f}' for f in ('t6', 't12') for kk in STATS]))
        print(f'{day} {part}: {len(jobs)} instance pairs, GPU {tg:.0f} s, total {time.time() - t:.0f} s, peak GPU '
              f'{torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else 0:.2f} GB', flush=True)


if __name__ == '__main__':
    main()
