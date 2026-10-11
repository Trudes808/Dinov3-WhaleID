"""Fine-tune DINOv3 for whale segmentation.

    python -m model_fine_tuning.segmentation.train --name seg_vitb16_ft4 --unfreeze-blocks 4

Writes experiments/<name>/{config.json, log.jsonl, checkpoints/{best,last}.pt, val_metrics.json}.
Validation runs on full 1280x720 frames; the selection metric is dataset IoU on val.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..common import paths
from ..common.data import IGNORE, SegCropDataset, WhaleRun, groups_for
from ..common.metrics import SegAccumulator
from ..common.train_utils import Logger, apply_lr, count_params, load_checkpoint, lr_at, move, save_checkpoint, set_seed
from .model import WhaleSegModel, seg_loss


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--run-dir", type=Path, default=paths.WHALE_RUN_DIR)
    p.add_argument("--split", type=Path, default=paths.SPLITS_DIR / "run01.json")
    p.add_argument("--out-root", type=Path, default=paths.EXPERIMENTS_DIR)
    p.add_argument("--arch", default="vitb16")
    p.add_argument("--unfreeze-blocks", type=int, default=4, help="0 frozen, N last blocks, -1 everything")
    p.add_argument("--layers", type=int, nargs="*", default=None)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--scale-min", type=float, default=0.5)
    p.add_argument("--scale-max", type=float, default=1.5)
    p.add_argument("--whale-bias", type=float, default=0.7)
    p.add_argument("--photometric", type=float, default=1.0)
    p.add_argument("--lr-head", type=float, default=3e-4)
    p.add_argument("--lr-backbone", type=float, default=3e-5)
    p.add_argument("--layer-decay", type=float, default=0.8)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--dice-weight", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--eval-every", type=int, default=1, help="epochs")
    p.add_argument("--eval-batch", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-train", type=int, default=None, help="debug: limit training images")
    p.add_argument("--max-val", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None, help="debug: stop after this many steps")
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args(argv)


@torch.no_grad()
def evaluate(model, loader, device, amp: bool, threshold: float = 0.5, run: WhaleRun | None = None) -> dict:
    model.eval()
    acc = SegAccumulator()
    for batch in loader:
        batch = move(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(batch["image"])
        prob = torch.sigmoid(logits.float()).cpu().numpy()
        masks = batch["mask"].cpu().numpy()
        for i, shot_id in enumerate(batch["id"]):
            h, w = batch["orig_hw"][i].tolist()
            pred = prob[i, :h, :w] >= threshold
            gt = masks[i, :h, :w] == 1
            ignore = masks[i, :h, :w] == IGNORE
            acc.add(shot_id, pred, gt, groups_for(run.by_id[shot_id]) if run else None, boundary=False, ignore=ignore)
    return acc.summary()


def main(argv=None) -> int:
    args = parse(argv)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = not args.no_amp and device.type == "cuda"
    out = args.out_root / args.name
    out.mkdir(parents=True, exist_ok=True)
    run = WhaleRun(args.run_dir, args.split)
    train_ids = run.ids("train")[: args.max_train]
    val_ids = run.ids("val")[: args.max_val]
    train_ds = SegCropDataset(run, train_ids, crop=args.crop, scale_range=(args.scale_min, args.scale_max), whale_bias=args.whale_bias,
                              photometric_strength=args.photometric, train=True, seed=args.seed)
    val_ds = SegCropDataset(run, val_ids, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch, shuffle=False, num_workers=min(args.workers, 4), pin_memory=True)

    model = WhaleSegModel(arch=args.arch, unfreeze_blocks=args.unfreeze_blocks, layers=args.layers).to(device)
    config = {"task": "segmentation", "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "model": model.config, "train_images": len(train_ds), "val_images": len(val_ds), "params": count_params(model)}
    (out / "config.json").write_text(json.dumps(config, indent=1))
    logger = Logger(out)
    logger.log(event="start", **{k: v for k, v in config.items() if k != "args"})

    optimizer = torch.optim.AdamW(model.param_groups(args.lr_head, args.lr_backbone, args.layer_decay, args.weight_decay), betas=(0.9, 0.999))
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    step, start_epoch, best = 0, 0, -1.0
    if args.resume:
        ckpt = load_checkpoint(args.resume)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        step, start_epoch, best = ckpt.get("step", 0), ckpt.get("epoch", 0) + 1, ckpt.get("best", -1.0)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_ds.set_epoch(epoch)
        t0 = time.time()
        running = []
        for batch in train_loader:
            factor = lr_at(step, total_steps, args.warmup_steps, 1.0)
            apply_lr(optimizer, factor)
            batch = move(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits = model(batch["image"])
            loss, parts = seg_loss(logits.float(), batch["mask"], args.dice_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], 1.0)
            optimizer.step()
            running.append(loss.item())
            step += 1
            if step % 50 == 0:
                logger.log(event="train", epoch=epoch, step=step, loss=float(np.mean(running[-50:])), **parts, lr_factor=factor)
            if args.max_steps and step >= args.max_steps:
                break
        epoch_loss = float(np.mean(running)) if running else None
        logger.log(event="epoch", epoch=epoch, step=step, loss=epoch_loss, seconds=round(time.time() - t0, 1))
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1 or (args.max_steps and step >= args.max_steps):
            metrics = evaluate(model, val_loader, device, amp, run=run)
            score = metrics["dataset_iou"] if metrics["dataset_iou"] is not None else -1.0
            logger.log(event="val", epoch=epoch, step=step, dataset_iou=metrics["dataset_iou"], mean_image_iou=metrics["mean_image_iou"],
                       negatives_clean=metrics["negatives_clean"], recall=metrics["recall"], precision=metrics["precision"])
            extra = {"optimizer": optimizer.state_dict(), "step": step, "epoch": epoch, "best": best, "val": metrics}
            if score > best:
                best = score
                extra["best"] = best
                save_checkpoint(out / "checkpoints" / "best.pt", model, config, {"step": step, "epoch": epoch, "val": metrics})
                (out / "val_metrics.json").write_text(json.dumps(metrics, indent=1))
            save_checkpoint(out / "checkpoints" / "last.pt", model, config, extra)
        if args.max_steps and step >= args.max_steps:
            break
    logger.log(event="done", best_val_dataset_iou=best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
