"""Re-segment the fin and fluke charts at native resolution: a tight silhouette for the shape channels.

    python -m model_fine_tuning.cluster.reseg --day-dir DAY [--checkpoint experiments/seg_r04_ft4/checkpoints/best.pt]

The day pipeline segments the 2560-px photo at 1280 px, so the mask on a 3072-px chart is a staircase at fin scale.
For every parts_xr fin / fluke chart the window around the part (silhouette box x 2, or the whale box) is cut from the
native-resolution chart and segmented again at max side 1024 (the fin fills the window, so the edge is resolved 3-6x
finer); the component that overlaps the old mask most is kept and written to parts_xr/<key>__<part>.reseg.npz
(box, mask inside the box, overlap with the old mask). Consumers (cluster/edge_match.py, cluster/sil_charts.py) take
the refined mask where it exists and agrees with the old one (IoU >= min_iou), the old mask otherwise (spray, blur).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from ..segmentation.model import WhaleSegModel
from ..segmentation.predict import segment


def refined_mask(npz_path: Path, part: str, min_iou: float = 0.5) -> np.ndarray | None:
    """The chart-size whale mask with the re-segmented window pasted in, or None when there is no usable refinement."""
    r = npz_path.with_name(npz_path.name[:-4] + ".reseg.npz")
    if not r.exists():
        return None
    z = np.load(r)
    if float(z["iou"]) < min_iou:
        return None
    whale = np.load(npz_path)["whale"].copy(); x0, y0, x1, y1 = [int(v) for v in z["box"]]
    whale[y0:y1, x0:x1] = z["mask"]
    return whale


@torch.no_grad()
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--day-dir", type=Path, required=True); p.add_argument("--checkpoint", type=Path, default=Path("experiments/seg_r04_ft4/checkpoints/best.pt"))
    p.add_argument("--max-side", type=int, default=1024); p.add_argument("--xr-tag", default="parts_xr")
    a = p.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model, _ = WhaleSegModel.from_checkpoint(a.checkpoint, device)
    src = a.day_dir / a.xr_tag; done = 0; ious = []
    for part in ("fin", "fluke"):
        for npz in sorted(src.glob(f"*__{part}.npz")):
            key = npz.name[: -len(f"__{part}.npz")]; out = npz.with_name(npz.name[:-4] + ".reseg.npz")
            if out.exists():
                done += 1; continue
            z = np.load(npz); whale = z["whale"]; ref = z["silhouette"] if part == "fin" and z["silhouette"].sum() > 300 else (z["fluke"] if part == "fluke" else whale)
            ys, xs = np.nonzero(ref)
            if len(xs) == 0:
                continue
            S = whale.shape[0]; cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2; side = max(np.ptp(xs), np.ptp(ys)) * 2.0 + 64
            x0, y0 = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0)); x1, y1 = int(min(cx + side / 2, S)), int(min(cy + side / 2, S))
            img = cv2.cvtColor(cv2.imread(str(src / "charts" / f"{key}__{part}.jpg"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)[y0:y1, x0:x1]
            _, mask = segment(model, img, device, max_side=a.max_side, threshold=0.5)
            old = whale[y0:y1, x0:x1]
            n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
            if n > 1:
                best = 1 + int(np.argmax([(lab == i)[old].sum() for i in range(1, n)])); keep = lab == best
                # keep every component that overlaps the old mask substantially (a back and its fin can be two blobs after a splash)
                for i in range(1, n):
                    if (lab == i)[old].sum() > 0.5 * (lab == i).sum():
                        keep |= lab == i
                mask = keep
            iou = float((mask & old).sum() / max((mask | old).sum(), 1)); ious.append(iou)
            np.savez_compressed(out, box=np.array([x0, y0, x1, y1]), mask=mask, iou=iou); done += 1
    print(f"{a.day_dir.name}: {done} refined masks ({len(ious)} new, IoU with the old mask median {np.median(ious) if ious else float('nan'):.2f}, >= 0.5: {np.mean(np.array(ious) >= 0.5) if ious else float('nan'):.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
