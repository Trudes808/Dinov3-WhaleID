"""Whale / water segmentation trained on run05 renders plus real partial labels (seg_real.data).

    flock /var/tmp/sat3737-parts-cache/gpu_train.lock env PYTHONPATH=. .venv/bin/python -m model_fine_tuning.seg_real.train \
        --name seg_splash_s1 --real-p 0.3 [--paste-p 0.5] [--rw-p 0.3] --init experiments/seg_r05_ft4/checkpoints/best.pt --epochs 10

The recipe of experiments/seg_r05_ft4/config.json (ViT-B/16, last 4 blocks, crop 512, scale 0.5-1.5, whale bias 0.7,
photometric 1.0, AdamW lr head 3e-4 / backbone 3e-5, layer decay 0.8, wd 0.05, warm-up 200 steps then cosine, BCE +
Dice with ignore, batch 8, bf16) with three additions: each sample is a real crop with probability --real-p (S1),
real splash pasted into the synthetic crop with probability --paste-p (S2), a real-water composite (synth_aug,
unchanged) with probability --rw-p (S3). --init starts from a checkpoint (fine-tuning) instead of the DINOv3
weights. Validation = the run05 validation renders (dataset IoU, as the recipe); no real validation-day pixel is read.
Writes experiments/<name>/{config.json, log.jsonl, checkpoints/{best,last}.pt, val_metrics.json}.
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
from ..common.data import SegCropDataset, open_runs
from ..common.train_utils import Logger, apply_lr, count_params, load_checkpoint, lr_at, move, save_checkpoint, set_seed
from ..segmentation.model import WhaleSegModel, seg_loss
from ..segmentation.train import evaluate
from . import common as C
from .data import Mixed, RealBlobCrops, SynthCrops


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--run-dir", type=Path, default=C.RUN05)
    p.add_argument("--split", type=Path, default=paths.SPLITS_DIR / "run05.json")
    p.add_argument("--out-root", type=Path, default=paths.EXPERIMENTS_DIR)
    p.add_argument("--init", type=Path, default=None, help="fine-tune from this checkpoint (default: DINOv3 weights, as seg_r05_ft4)")
    p.add_argument("--real-p", type=float, default=0.3)
    p.add_argument("--real-dir", type=Path, default=None, help="real crops + labels (default CACHE/real; S4: CACHE/real_strict)")
    p.add_argument("--paste-p", type=float, default=0.0)
    p.add_argument("--rw-p", type=float, default=0.0)
    p.add_argument("--unfreeze-blocks", type=int, default=4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--lr-head", type=float, default=3e-4)
    p.add_argument("--lr-backbone", type=float, default=3e-5)
    p.add_argument("--layer-decay", type=float, default=0.8)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--dice-weight", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument("--eval-batch", type=int, default=4)
    p.add_argument("--max-val", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse(argv)
    set_seed(a.seed); dev = torch.device("cuda"); out = a.out_root / a.name; out.mkdir(parents=True, exist_ok=True)
    run = open_runs(a.run_dir, a.split)
    synth = SynthCrops(run, run.ids("train"), paste_p=a.paste_p, rw_p=a.rw_p, crop=a.crop, scale_range=(0.5, 1.5), whale_bias=0.7,
                       photometric_strength=1.0, train=True, seed=a.seed)
    real = RealBlobCrops(crop=a.crop, seed=a.seed, real_dir=a.real_dir) if a.real_p > 0 else None
    train_ds = Mixed(synth, real, a.real_p, seed=a.seed)
    val_ds = SegCropDataset(run, run.ids("val")[: a.max_val], train=False)
    tl = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, num_workers=a.workers, pin_memory=True, drop_last=True, persistent_workers=True)
    vl = DataLoader(val_ds, batch_size=a.eval_batch, shuffle=False, num_workers=4, pin_memory=True)
    model = WhaleSegModel(arch="vitb16", unfreeze_blocks=a.unfreeze_blocks).to(dev)
    if a.init:
        model.load_state_dict(load_checkpoint(a.init)["model"])
    config = {"task": "segmentation", "variant": "seg_real", "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}, "model": model.config,
              "train_synthetic": len(synth), "train_real_centres": len(real) if real else 0, "real_photos": len(real.index) if real else 0,
              "val_images": len(val_ds), "params": count_params(model), "recipe": "experiments/seg_r05_ft4/config.json"}
    (out / "config.json").write_text(json.dumps(config, indent=1))
    log = Logger(out); log.log(event="start", **{k: v for k, v in config.items() if k != "args"})
    opt = torch.optim.AdamW(model.param_groups(a.lr_head, a.lr_backbone, a.layer_decay, a.weight_decay), betas=(0.9, 0.999))
    total = a.epochs * len(tl); step, best = 0, -1.0
    for epoch in range(a.epochs):
        model.train(); train_ds.set_epoch(epoch); t0 = time.time(); run_l, n_real = [], 0
        for batch in tl:
            f = lr_at(step, total, a.warmup_steps, 1.0); apply_lr(opt, f)
            n_real += sum(str(i).startswith("real:") for i in batch["id"])
            batch = move(batch, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(batch["image"])
            loss, parts = seg_loss(logits.float(), batch["mask"], a.dice_weight)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0); opt.step()
            run_l.append(loss.item()); step += 1
            if step % 50 == 0:
                log.log(event="train", epoch=epoch, step=step, loss=float(np.mean(run_l[-50:])), **parts, lr_factor=f, sec_per_step=round((time.time() - t0) / max(1, len(run_l)), 3))
            if a.max_steps and step >= a.max_steps:
                break
        log.log(event="epoch", epoch=epoch, step=step, loss=float(np.mean(run_l)), seconds=round(time.time() - t0, 1), real_share=round(n_real / max(1, len(run_l) * a.batch_size), 3))
        last = epoch == a.epochs - 1 or (a.max_steps and step >= a.max_steps)
        if (epoch + 1) % a.eval_every == 0 or last:
            m = evaluate(model, vl, dev, True, run=run); sc = m["dataset_iou"] or -1.0
            log.log(event="val", epoch=epoch, step=step, dataset_iou=m["dataset_iou"], mean_image_iou=m["mean_image_iou"], negatives_clean=m["negatives_clean"],
                    recall=m["recall"], precision=m["precision"])
            if sc > best:
                best = sc; save_checkpoint(out / "checkpoints" / "best.pt", model, config, {"step": step, "epoch": epoch, "val": m})
                (out / "val_metrics.json").write_text(json.dumps({k: v for k, v in m.items() if k not in ("by_category", "by_sea_state", "by_size_bin")}, indent=1))
            if last:
                save_checkpoint(out / "checkpoints" / "last.pt", model, config, {"step": step, "epoch": epoch, "val": m})
                (out / "last_val_metrics.json").write_text(json.dumps({k: v for k, v in m.items() if k not in ("by_category", "by_sea_state", "by_size_bin")}, indent=1))
        if a.max_steps and step >= a.max_steps:
            break
    log.log(event="done", best_val_dataset_iou=best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
