"""Batched LightGlue + RANSAC verification of every instance pair behind the bench's cluster pairs of a day.

    flock ... python lm_match.py VARIANT PART DAY [DAY ...] [--max-sec 1100]
Writes CACHE/VARIANT/DAY__PART__pairs.npz: one row per (cluster pair, instance a, instance b, mirror) with the stats
of lm_lib.verify at two RANSAC thresholds (0.6 % and 1.2 % of the chart size).
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lm_lib as LM  # noqa: E402
from lm_lib import bench  # noqa: E402

BATCH = 48
STATS = ('n', 'sim', 'simc', 'hom', 'aff', 'spread')


def _verify(args):
    pa, pb, conf, area, size = args
    row = []
    for f in (0.006, 0.012):
        st, _ = LM.verify(pa, pb, conf, area, thr=f * size)
        row += [st[k] for k in STATS]
    return row


def lightglue(variant, dev):
    import kornia.feature as KF
    lg = KF.LightGlue('aliked' if variant.startswith('aliked') else 'disk').to(dev).eval()
    lg.conf.depth_confidence = -1
    lg.conf.width_confidence = -1
    lg.conf.filter_threshold = 0.1
    return lg


def todev(f, dev):
    return f['kp'].float().to(dev), f['desc'].float().to(dev)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    variant, part, days = args[0], args[1], args[2:]
    dev = torch.device(os.environ.get('LM_DEV', 'cuda'))
    lg = lightglue(variant, dev)
    pool = Pool(8)
    for day in days:
        fp = LM.CACHE / variant / f'{day}__{part}__pairs.npz'
        if fp.exists():
            print(day, 'cached'); continue
        t = time.time()
        F = torch.load(LM.CACHE / variant / f'{day}__{part}.pt', weights_only=False)
        picked, feats, size = F['picked'], F['feats'], F['size']
        npm = 3 if day in bench.VAL else 2          # train days: the first 2 picks per cluster (cost); val scored 3 x 3
        picked = {c: v[:npm] for c, v in picked.items()}
        keys = sorted(feats)
        kidx = {k: i for i, k in enumerate(keys)}
        jobs = []                                            # (ca, cb, ia, ib, mirror)
        for a, b, same in bench.pairs(day, part):
            for ka in picked.get(a, []):
                for kb in picked.get(b, []):
                    jobs.append((a, b, kidx[ka], kidx[kb], 0))
                    if part == 'dorsal' and '--no-mirror' not in sys.argv:
                        jobs.append((a, b, kidx[ka], kidx[kb], 1))
        # order by keypoint count so that padding is small
        nk = lambda j: max(len(feats[keys[j[2]]]['plain']['kp']), len(feats[keys[j[3]]]['mirror' if j[4] else 'plain']['kp']))
        order = sorted(range(len(jobs)), key=lambda i: nk(jobs[i]))
        tasks = [None] * len(jobs)
        for s in range(0, len(order), BATCH):
            idx = order[s:s + BATCH]
            fa = [feats[keys[jobs[i][2]]]['plain'] for i in idx]
            fb = [feats[keys[jobs[i][3]]]['mirror' if jobs[i][4] else 'plain'] for i in idx]
            L = max(4, max(max(len(x['kp']), len(y['kp'])) for x, y in zip(fa, fb)))
            B, D = len(idx), fa[0]['desc'].shape[1]
            kp0 = torch.zeros(B, L, 2, device=dev); kp1 = torch.zeros(B, L, 2, device=dev)
            d0 = torch.zeros(B, L, D, device=dev); d1 = torch.zeros(B, L, D, device=dev)
            v0 = torch.zeros(B, L, dtype=torch.bool, device=dev); v1 = torch.zeros(B, L, dtype=torch.bool, device=dev)
            for r, (x, y) in enumerate(zip(fa, fb)):
                n0, n1 = len(x['kp']), len(y['kp'])
                kp0[r, :n0], d0[r, :n0], v0[r, :n0] = x['kp'].float(), x['desc'].float(), True
                kp1[r, :n1], d1[r, :n1], v1[r, :n1] = y['kp'].float(), y['desc'].float(), True
            res = LM.lg_batch(lg, kp0, d0, v0, kp1, d1, v1, size)
            for r, i in enumerate(idx):
                m, sc = res[r]
                pa = fa[r]['kp'].float().numpy()[m[:, 0]] if len(m) else np.zeros((0, 2), np.float32)
                pb = fb[r]['kp'].float().numpy()[m[:, 1]] if len(m) else np.zeros((0, 2), np.float32)
                tasks[i] = (pa, pb, sc, feats[keys[jobs[i][2]]]['area'], size)
        tg = time.time() - t
        rows = pool.map(_verify, tasks, chunksize=64)
        J = np.array([(a, b, ia, ib, mi) for a, b, ia, ib, mi in jobs], np.int64).reshape(-1, 5)
        off = np.cumsum([0] + [len(t[0]) for t in tasks])
        np.savez_compressed(str(fp).replace('__pairs.npz', '__matches.npz'), off=off,
                            pa=np.concatenate([t[0] for t in tasks]).astype(np.float32) if len(tasks) else np.zeros((0, 2), np.float32),
                            pb=np.concatenate([t[1] for t in tasks]).astype(np.float32) if len(tasks) else np.zeros((0, 2), np.float32),
                            sc=np.concatenate([t[2] for t in tasks]).astype(np.float32) if len(tasks) else np.zeros(0, np.float32))
        np.savez_compressed(fp, jobs=J, stats=np.array(rows, np.float32).reshape(len(jobs), 2 * len(STATS)), keys=np.array(keys),
                            names=np.array([f'{k}@{f}' for f in ('t6', 't12') for k in STATS]))
        print(f'{day} {part}: {len(jobs)} instance pairs, GPU {tg:.0f} s, total {time.time() - t:.0f} s, '
              f'peak GPU {torch.cuda.max_memory_allocated() / 2**30 if dev.type == 'cuda' else 0:.2f} GB', flush=True)


if __name__ == '__main__':
    main()
