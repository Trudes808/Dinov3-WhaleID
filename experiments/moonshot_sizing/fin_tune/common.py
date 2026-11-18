"""fin_tune (2026-11-18): does fine-tuning the public HW-b6 dorsal model on our labelled days beat it raw, and does the gain
grow with the number of days (5 -> 10 -> 16 of bench.splits()[0]'s pool)? Shared helpers.

Crops: the 'box' crop of public_reid/extract.py = square of max(box side) x 1.15 around the instance box, cut from the
2560-px image (outside the photo = black), resized to 1024 px, ImageNet mean / std. Test embeddings use exactly that.
Training crops are cut once (CPU) with more context (x CTX instead of x 1.15) at TRAIN_PX so that the x 1.15 part is
1024 px, then jittered in the loader.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / 'public_reid'))
import bench  # noqa: E402
from size_levers import group_of  # noqa: E402,F401

CACHE = Path('/home/sat3737/NOAA_whales/moonshot_cache/fin_tune')
HW_FEAT = Path('/home/sat3737/NOAA_whales/moonshot_cache/public_reid/feat/hw_b6/box')
RES, CTX = 1024, 1.4
TRAIN_PX = int(round(RES * CTX / 1.15))          # 1247
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def split():
    hold, pool = bench.splits()[0]
    return hold, pool, bench.VAL + hold


def crop(day, key, mult, out_px):
    """Square crop of max(box side) * mult around the box centre, resized to out_px (as extract.py's Crops)."""
    im = Image.open(bench.image_path(day, key)).convert('RGB')
    b = bench.box(day, key)
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    s = max(max(b[2] - b[0], b[3] - b[1]) * mult, 64.0 * mult / 1.15)
    r = (int(round(cx - s / 2)), int(round(cy - s / 2)), int(round(cx + s / 2)), int(round(cy + s / 2)))
    flt = Image.BICUBIC if s < out_px else Image.LANCZOS
    return im.crop(r).resize((out_px, out_px), flt)


def unit(X):
    X = np.asarray(X, np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


def hw_raw(day):
    z = np.load(HW_FEAT / f'{day}.npz')
    return dict(zip([str(k) for k in z['keys']], z['X'].astype(np.float32)))
