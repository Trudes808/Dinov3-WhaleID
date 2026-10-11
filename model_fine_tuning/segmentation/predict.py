"""Run the segmentation model on a folder of photos (no labels needed).

    python -m model_fine_tuning.segmentation.predict --checkpoint CKPT --images DIR --out OUT

Images are resized so the longer side is --max-side (default 1280, the training scale) and
padded to a multiple of 16. Outputs per image: <name>_mask.png (0/1 at original size),
<name>_overlay.jpg, and predictions.jsonl with whale pixel counts and bounding boxes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from ..common.data import pad_to_multiple, to_tensor
from ..common.viz import seg_overlay
from .model import WhaleSegModel

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def list_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_long_side(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if abs(scale - 1.0) < 1e-6:
        return image, 1.0
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=interp), scale


@torch.no_grad()
def segment(model: WhaleSegModel, image: np.ndarray, device, max_side: int = 1280, threshold: float = 0.5, flip_tta: bool = False):
    """Returns (prob at original size, mask bool at original size)."""
    small, scale = resize_long_side(image, max_side)
    padded, (h, w) = pad_to_multiple(small)
    x = to_tensor(padded)[None].to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model(x).float()
        if flip_tta:
            logits = 0.5 * (logits + model(x.flip(-1)).float().flip(-1))
    prob = torch.sigmoid(logits)[0, :h, :w].cpu().numpy()
    if scale != 1.0:
        prob = cv2.resize(prob, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    return prob, prob >= threshold


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True, help="folder (searched recursively) or a single image")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-side", type=int, default=1280)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--flip-tta", action="store_true")
    args = p.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = WhaleSegModel.from_checkpoint(args.checkpoint, device)
    args.out.mkdir(parents=True, exist_ok=True)
    files = list_images(args.images)
    if not files:
        print("no images found in", args.images)
        return 1
    with open(args.out / "predictions.jsonl", "w") as handle:
        for path in files:
            image = load_rgb(path)
            prob, mask = segment(model, image, device, args.max_side, args.threshold, args.flip_tta)
            stem = path.stem
            cv2.imwrite(str(args.out / f"{stem}_mask.png"), mask.astype(np.uint8))
            cv2.imwrite(str(args.out / f"{stem}_overlay.jpg"), cv2.cvtColor(seg_overlay(image, mask), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
            ys, xs = np.nonzero(mask)
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) else None
            rec = {"image": str(path), "whale_pixels": int(mask.sum()), "fraction": round(float(mask.mean()), 5), "bbox_xyxy": bbox,
                   "max_prob": round(float(prob.max()), 4)}
            handle.write(json.dumps(rec) + "\n")
            print(rec)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
