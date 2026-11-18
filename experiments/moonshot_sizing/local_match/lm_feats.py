"""Keypoints + descriptors per instance (chart and, for fins, its mirror), whale pixels only.

    flock ... python lm_feats.py VARIANT PART DAY [DAY ...]
VARIANT  aliked1024 | aliked1536 | disk1024 (extractor + chart size); cache CACHE/VARIANT/DAY__PART.pt
Instances: up to NPICK per safe cluster of the bench (largest / sharpest part), only instances with a parts_up cut-out.
"""
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lm_lib as LM  # noqa: E402
from lm_lib import bench  # noqa: E402

NPICK = 3
FALLBACK = '--fallback' in sys.argv
MAXKP = 1024
DIL = 6        # px at 1024 of whale-mask dilation (outline keypoints kept)


def extractor(variant, dev):
    import kornia.feature as KF
    if variant.startswith('aliked'):
        return KF.ALIKED.from_pretrained('aliked-n16', max_num_keypoints=4 * MAXKP, detection_threshold=0.0, nms_radius=2).to(dev).eval()
    if variant.startswith('disk'):
        return KF.DISK.from_pretrained('depth').to(dev).eval()
    raise ValueError(variant)


@torch.no_grad()
def run(ext, variant, img, mask, dev):
    size = img.shape[0]
    x = torch.from_numpy(img).permute(2, 0, 1)[None].float().div(255.0).to(dev)
    if variant.startswith('aliked'):
        f = ext(x)[0]
        kp, desc, sc = f.keypoints, f.descriptors, f.keypoint_scores
    else:
        f = ext(x, n=4 * MAXKP, window_size=5, score_threshold=0.0, pad_if_not_divisible=True)[0]
        kp, desc, sc = f.keypoints, f.descriptors, f.detection_scores
    k = max(1, int(round(DIL * size / 1024)))
    m = cv2.dilate(mask.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)) > 0
    ij = kp.round().long().clamp(0, size - 1).cpu().numpy()
    keep = np.where(m[ij[:, 1], ij[:, 0]])[0]
    keep = keep[np.argsort(-sc[keep].cpu().numpy())][:MAXKP]
    keep = torch.from_numpy(keep).to(dev)
    return dict(kp=kp[keep].cpu().half(), desc=desc[keep].cpu().half(), sc=sc[keep].cpu().half())


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    variant, part, days = args[0], args[1], args[2:]
    size = int(''.join(c for c in variant if c.isdigit()))
    dev = torch.device(os.environ.get('LM_DEV', 'cuda'))
    ext = extractor(variant, dev)
    out_dir = LM.CACHE / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    for day in days:
        t = time.time()
        fp = out_dir / f'{day}__{part}.pt'
        C = bench.clusters(day, part)
        if fp.exists() and FALLBACK and part == 'fluke':
            old = torch.load(fp, weights_only=False)
            if old.get('fallback'):
                print(day, 'cached'); continue
            n0 = len(old['feats'])
            for c, keys in C.items():
                if old['picked'].get(c):
                    continue
                bxa = {k: (lambda b: (b[2] - b[0]) * (b[3] - b[1]))(bench.box(day, k)) for k in keys}
                old['picked'][c] = sorted(keys, key=lambda k: -bxa[k])[:NPICK]
                for key in old['picked'][c]:
                    img, whale, _ = LM.load_chart(day, key, part, size)
                    old['feats'][key] = dict(area=float(whale.sum()), meta=dict(pixels=int(bxa[key]), fallback=True), plain=run(ext, variant, img, whale, dev))
            old['fallback'] = True
            torch.save(old, fp)
            print(f'{day} {part}: fallback box crops added, {n0} -> {len(old["feats"])} instances', flush=True)
            continue
        if fp.exists():
            print(day, 'cached'); continue
        sharp, feats, picked = {}, {}, {}
        for c, keys in C.items():
            for key in keys:
                if LM.meta(day, key, part):
                    img, whale, _ = LM.load_chart(day, key, part, 512)
                    sharp[key] = LM.sharpness(img, whale)
            picked[c] = LM.pick(day, part, keys, NPICK, sharp)
        for c, keys in picked.items():
            for key in keys:
                img, whale, pm = LM.load_chart(day, key, part, size)
                area = float(whale.sum())
                rec = dict(area=area, meta=LM.meta(day, key, part), plain=run(ext, variant, img, whale, dev))
                if part == 'dorsal' and '--no-mirror' not in sys.argv:
                    rec['mirror'] = run(ext, variant, np.ascontiguousarray(img[:, ::-1]), np.ascontiguousarray(whale[:, ::-1]), dev)
                feats[key] = rec
        torch.save(dict(picked=picked, feats=feats, size=size, sharp=sharp), fp)
        print(f'{day} {part}: {len(C)} clusters, {len(feats)} instances, {time.time() - t:.0f} s, peak GPU {torch.cuda.max_memory_allocated() / 2**30 if dev.type == 'cuda' else 0:.2f} GB', flush=True)


if __name__ == '__main__':
    main()
