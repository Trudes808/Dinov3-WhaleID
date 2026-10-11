"""Evaluate a segmentation checkpoint on a split (or another run in the same format).

    python -m model_fine_tuning.segmentation.evaluate --checkpoint experiments/seg_x/checkpoints/best.pt --split test

Writes <out>/metrics.json (overall + per category / sea state / whale size), per_image.jsonl,
and overlay PNGs (TP green, FP red, FN blue) for the worst and a random sample of images.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..common import paths
from ..common.data import IGNORE, SegCropDataset, WhaleRun, groups_for
from ..common.metrics import SegAccumulator
from ..common.train_utils import move
from ..common.viz import seg_overlay
from .model import WhaleSegModel


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, default=paths.WHALE_RUN_DIR)
    p.add_argument("--split-file", type=Path, default=paths.SPLITS_DIR / "run01.json")
    p.add_argument("--split", default="test", help="train | val | test | all")
    p.add_argument("--out", type=Path, default=None, help="default: <experiment>/eval_<split>")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--overlays", type=int, default=24)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--save-masks", action="store_true", help="also write predicted masks as PNG (0/1)")
    return p.parse_args(argv)


@torch.no_grad()
def predict_probs(model, images: torch.Tensor, amp: bool = True) -> np.ndarray:
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and images.is_cuda):
        logits = model(images)
    return torch.sigmoid(logits.float()).cpu().numpy()


def main(argv=None) -> int:
    args = parse(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = WhaleSegModel.from_checkpoint(args.checkpoint, device)
    out = args.out or args.checkpoint.parents[1] / f"eval_{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "overlays").mkdir(exist_ok=True)
    if args.save_masks:
        (out / "masks").mkdir(exist_ok=True)
    run = WhaleRun(args.run_dir, args.split_file if args.split != "all" else None)
    ids = run.ids(args.split)[: args.max_images]
    ds = SegCropDataset(run, ids, train=False)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    acc = SegAccumulator()
    keep = {}
    for batch in loader:
        batch = move(batch, device)
        prob = predict_probs(model, batch["image"])
        masks = batch["mask"].cpu().numpy()
        for i, sid in enumerate(batch["id"]):
            h, w = batch["orig_hw"][i].tolist()
            pred = prob[i, :h, :w] >= args.threshold
            gt = masks[i, :h, :w] == 1
            ignore = masks[i, :h, :w] == IGNORE
            row = acc.add(sid, pred, gt, groups_for(run.by_id[sid]), boundary=True, ignore=ignore)
            keep[sid] = pred
            if args.save_masks:
                cv2.imwrite(str(out / "masks" / f"{sid}.png"), pred.astype(np.uint8))
    summary = acc.summary()
    summary["checkpoint"] = str(args.checkpoint)
    summary["checkpoint_epoch"] = ckpt.get("epoch")
    summary["split"] = args.split
    summary["threshold"] = args.threshold
    (out / "metrics.json").write_text(json.dumps(summary, indent=1))
    with open(out / "per_image.jsonl", "w") as handle:
        for row in acc.rows:
            handle.write(json.dumps(row) + "\n")
    # overlays: worst IoU among images with a whale, worst false positives among negatives, random
    rows = acc.rows
    worst = sorted([r for r in rows if r["gt_pixels"] > 0], key=lambda r: r["iou"])[: args.overlays // 2]
    worst_neg = sorted([r for r in rows if r["gt_pixels"] == 0], key=lambda r: -r["pred_pixels"])[: args.overlays // 4]
    rng = np.random.default_rng(0)
    rand = [rows[i] for i in rng.choice(len(rows), min(args.overlays // 4, len(rows)), replace=False)]
    for tag, group in (("worst", worst), ("worst_negative", worst_neg), ("random", rand)):
        for r in group:
            sid = r["id"]
            over = seg_overlay(run.image(sid), keep[sid], run.visible(sid))
            label = f"{sid} {r['category']} {r['sea_state']} IoU={r['iou'] if r['iou'] is not None else 'n/a'}"
            cv2.putText(over, label[:80], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(str(out / "overlays" / f"{tag}_{sid}.jpg"), cv2.cvtColor(over, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(json.dumps({k: v for k, v in summary.items() if not k.startswith("by_")}, indent=1))
    for key in ("by_category", "by_size_bin", "by_sea_state"):
        if key in summary:
            print(key)
            for name, agg in summary[key].items():
                iou = agg["dataset_iou"]
                print(f"  {name:16s} n={agg['n']:4d} dataset_iou={iou if iou is None else round(iou, 4)} mean_image_iou={agg['mean_image_iou'] if agg['mean_image_iou'] is None else round(agg['mean_image_iou'], 4)} neg_clean={agg['negatives_clean']}")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
