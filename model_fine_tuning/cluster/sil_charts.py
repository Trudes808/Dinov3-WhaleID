"""Shape-only charts: the dorsal fin silhouette (and the fluke mask) as images, so an embedding can be trained on shape
alone, the researchers' first cue for backs, without the texture the appearance embeddings already use.

    python -m model_fine_tuning.cluster.sil_charts --day-dir DAY [--size 512]

From parts_hr/<key>__fin.npz: channels R = isolated fin silhouette, G = whale mask, B = fin band (the body map's fin
region), cropped to the silhouette's box with a 35 % margin (the box of the whole mask when the silhouette is empty)
and resized; fluke: R = fluke blades, G = whale mask. Written to parts_sil/charts/<key>__<part>.jpg with a copy of
the npz metadata (so train_id / cluster_day can use parts_sil like any chart directory).
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np


def make(args):
    npz, out_dir, size, use_refined = args
    z = np.load(npz); key, part = npz.name[:-4].rsplit("__", 1)
    whale = z["whale"].astype(np.uint8)
    rm = None
    if use_refined:
        from .reseg import refined_mask

        src_xr = npz.parent.parent / "parts_xr" / npz.name                                # the native-resolution twin of this chart
        if src_xr.exists():
            rm_xr = refined_mask(src_xr, part)
            if rm_xr is not None:
                rm = cv2.resize(rm_xr.astype(np.uint8), (whale.shape[1], whale.shape[0]), interpolation=cv2.INTER_AREA) > 0.5
    if rm is not None:
        whale = rm.astype(np.uint8)
    if part == "fin":
        from .finshape import isolate_fin

        band = z["fin_band"].astype(np.uint8)
        sil = (isolate_fin(whale > 0) if rm is not None else z["silhouette"])
        sil = (sil.astype(np.uint8) if sil is not None else z["silhouette"].astype(np.uint8))
        ref = sil if sil.sum() >= 300 else band if band.sum() >= 300 else whale
    else:
        k = max(int(0.03 * whale.shape[0]), 3)
        sil = ((whale > 0) & (cv2.dilate(z["fluke"].astype(np.uint8), np.ones((k, k), np.uint8)) > 0)).astype(np.uint8) if rm is not None else z["fluke"].astype(np.uint8)
        band = np.zeros_like(whale); ref = sil if sil.sum() >= 300 else whale
    ys, xs = np.nonzero(ref)
    if len(xs) == 0:
        return None
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2; side = max(xs.max() - xs.min(), ys.max() - ys.min()) * 1.7 + 32
    x0, y0 = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0)); x1, y1 = int(min(cx + side / 2, whale.shape[1])), int(min(cy + side / 2, whale.shape[0]))
    img = np.stack([sil[y0:y1, x0:x1], whale[y0:y1, x0:x1], band[y0:y1, x0:x1]], -1) * 255
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_dir / "charts" / f"{key}__{part}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    np.savez_compressed(out_dir / f"{key}__{part}.npz", whale=cv2.resize(whale[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_NEAREST) > 0, part=part, stem=key)
    return key


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--day-dir", type=Path, required=True); p.add_argument("--size", type=int, default=512); p.add_argument("--workers", type=int, default=8)
    p.add_argument("--refined", action="store_true", help="use the native-resolution re-segmentation (cluster/reseg.py) where it exists"); p.add_argument("--out-tag", default="parts_sil")
    a = p.parse_args(argv)
    src = a.day_dir / "parts_hr"; out = a.day_dir / a.out_tag; (out / "charts").mkdir(parents=True, exist_ok=True)
    jobs = [(f, out, a.size, a.refined) for part in ("fin", "fluke") for f in sorted(src.glob(f"*__{part}.npz")) if not f.name.endswith(".reseg.npz")]
    with Pool(a.workers) as pool:
        done = [r for r in pool.imap_unordered(make, jobs, chunksize=8) if r]
    (out / "index.jsonl").write_text("\n".join(json.dumps({"stem": k}) for k in sorted(set(done))) + "\n")
    print(f"{a.day_dir.name}: {len(done)} shape charts -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
