"""Fine-tune DINOv3 for whale PART segmentation on factory renders (run05).

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.parts_seg.train --name parts_seg_r05_a \
        --run-dir ../Whale-Image-Data-Factory/data/renders/run05 --split model_fine_tuning/splits/run05.json \
        --init-from experiments/seg_r05_ft4/checkpoints/best.pt --epochs 24 --eval-every 3

Writes experiments/<name>/{config.json, log.jsonl, checkpoints/{best,last}.pt, val_metrics.json}. Validation on full
1280x720 frames of the split's val shots; the selection metric is the mean IoU of fluke and dorsal fin.
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
from ..common.data import IGNORE, WhaleRun
from ..common.train_utils import Logger, apply_lr, count_params, load_checkpoint, lr_at, move, save_checkpoint, set_seed
from .data import DEFAULT_CACHE, PartsCropDataset
from .labels import CLASSES, COARSE
from .model import PartsSegModel, parts_loss

K = len(CLASSES)


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--run-dir", type=Path, default=paths.WHALE_FACTORY_ROOT / "data" / "renders" / "run05")
    p.add_argument("--split", type=Path, default=paths.SPLITS_DIR / "run05.json")
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--out-root", type=Path, default=paths.EXPERIMENTS_DIR)
    p.add_argument("--init-from", type=Path, default=None, help="whale / background segmentation checkpoint to start from")
    p.add_argument("--init-weights", type=Path, default=None, help="a parts_seg checkpoint to continue from (weights only, fresh schedule)")
    p.add_argument("--arch", default="vitb16")
    p.add_argument("--unfreeze-blocks", type=int, default=4)
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--scale-min", type=float, default=0.5)
    p.add_argument("--scale-max", type=float, default=1.5)
    p.add_argument("--whale-bias", type=float, default=0.7)
    p.add_argument("--rare-bias", type=float, default=0.5)
    p.add_argument("--photometric", type=float, default=1.0)
    p.add_argument("--lr-head", type=float, default=3e-4)
    p.add_argument("--lr-backbone", type=float, default=3e-5)
    p.add_argument("--layer-decay", type=float, default=0.8)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--dice-weight", type=float, default=1.0)
    p.add_argument("--class-weights", type=float, nargs="*", default=None, help="CE weights per class (default: from label frequencies)")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--eval-every", type=int, default=3)
    p.add_argument("--eval-batch", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-val", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args(argv)


def class_frequencies(cache: Path, ids: list[str], n: int = 600, seed: int = 0) -> np.ndarray:
    from PIL import Image

    rng = np.random.default_rng(seed)
    counts = np.zeros(K, np.float64)
    for i in rng.choice(len(ids), min(n, len(ids)), replace=False):
        p = cache / f"{ids[i]}.png"
        if p.exists():
            counts += np.bincount(np.asarray(Image.open(p)).ravel(), minlength=256)[:K]
    return counts / counts.sum()


def iou_table(conf: np.ndarray, names: list[str]) -> dict:
    tp = np.diag(conf).astype(np.float64)
    union = conf.sum(0) + conf.sum(1) - tp
    out = {n: (float(tp[k] / union[k]) if union[k] > 0 else None) for k, n in enumerate(names)}
    out["pixels"] = {n: int(conf[k].sum()) for k, n in enumerate(names)}
    return out


def collapse(conf: np.ndarray) -> np.ndarray:
    c = np.zeros((4, 4), np.int64)
    for i in range(K):
        for j in range(K):
            c[COARSE[i], COARSE[j]] += conf[i, j]
    return c


@torch.no_grad()
def evaluate(model, loader, device, amp: bool, run: WhaleRun) -> dict:
    model.eval()
    total = np.zeros((K, K), np.int64)
    by_cat: dict[str, np.ndarray] = {}
    for batch in loader:
        batch = move(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(batch["image"])
        pred = logits.float().argmax(1).cpu().numpy()
        lab = batch["label"].cpu().numpy()
        for i, sid in enumerate(batch["id"]):
            h, w = batch["orig_hw"][i].tolist()
            t, p = lab[i, :h, :w].ravel(), pred[i, :h, :w].ravel()
            ok = t != IGNORE
            c = np.bincount(t[ok] * K + p[ok], minlength=K * K).reshape(K, K)
            total += c
            cat = run.by_id[sid]["category"]
            by_cat[cat] = by_cat.get(cat, np.zeros((K, K), np.int64)) + c
    coarse_names = ["background", "fluke", "dorsal_fin", "other_whale"]
    res = {"iou": iou_table(total, CLASSES), "coarse_iou": iou_table(collapse(total), coarse_names), "confusion": total.tolist(),
           "by_category": {cat: {"iou": iou_table(c, CLASSES), "coarse_iou": iou_table(collapse(c), coarse_names)} for cat, c in sorted(by_cat.items())}}
    fin, fl = res["iou"]["dorsal_fin"], res["iou"]["fluke"]
    whale_iou = [res["iou"][n] for n in CLASSES[1:] if res["iou"][n] is not None]
    res["mean_iou_whale_classes"] = float(np.mean(whale_iou)) if whale_iou else None
    res["score"] = float(np.mean([v for v in (fin, fl) if v is not None])) if (fin is not None or fl is not None) else -1.0
    return res


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
    train_ds = PartsCropDataset(run, train_ids, args.cache, crop=args.crop, scale_range=(args.scale_min, args.scale_max), whale_bias=args.whale_bias,
                                rare_bias=args.rare_bias, photometric_strength=args.photometric, train=True, seed=args.seed)
    val_ds = PartsCropDataset(run, val_ids, args.cache, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch, shuffle=False, num_workers=min(args.workers, 4), pin_memory=True)

    freq = class_frequencies(args.cache, train_ids)
    if args.class_weights:
        cw = np.asarray(args.class_weights, np.float64)
    else:                                   # inverse square root of the pixel frequency, background = 1, capped at 10
        cw = np.clip(np.sqrt(freq[0] / np.maximum(freq, 1e-9)), 1.0, 10.0)
    class_weight = torch.tensor(cw, dtype=torch.float32, device=device)

    model = PartsSegModel(arch=args.arch, unfreeze_blocks=args.unfreeze_blocks).to(device)
    init = model.init_from_binary(args.init_from) if args.init_from else None
    if args.init_weights:
        model.load_state_dict(load_checkpoint(args.init_weights)["model"], strict=True)
        init = {"loaded": "all", "not_loaded": [], "from_parts_checkpoint": str(args.init_weights)}
    config = {"task": "parts_segmentation", "classes": CLASSES,
              "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "model": model.config, "train_images": len(train_ds), "val_images": len(val_ds), "params": count_params(model),
              "class_frequency": freq.round(6).tolist(), "class_weight": [round(float(x), 3) for x in cw],
              "init": None if init is None else {"from": str(args.init_weights or args.init_from), "loaded_tensors": init["loaded"], "not_loaded": init["not_loaded"]}}
    (out / "config.json").write_text(json.dumps(config, indent=1))
    logger = Logger(out)
    logger.log(event="start", **{k: v for k, v in config.items() if k not in ("args", "init")})

    optimizer = torch.optim.AdamW(model.param_groups(args.lr_head, args.lr_backbone, args.layer_decay, args.weight_decay), betas=(0.9, 0.999))
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    step, best = 0, -1.0
    for epoch in range(args.epochs):
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
            loss, parts = parts_loss(logits.float(), batch["label"], class_weight, args.dice_weight)
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
        logger.log(event="epoch", epoch=epoch, step=step, loss=float(np.mean(running)) if running else None, seconds=round(time.time() - t0, 1))
        last = epoch == args.epochs - 1 or bool(args.max_steps and step >= args.max_steps)
        if (epoch + 1) % args.eval_every == 0 or last:
            t1 = time.time()
            metrics = evaluate(model, val_loader, device, amp, run)
            logger.log(event="val", epoch=epoch, step=step, score=metrics["score"], mean_iou_whale=metrics["mean_iou_whale_classes"],
                       iou={k: (round(v, 4) if v is not None else None) for k, v in metrics["iou"].items() if k != "pixels"},
                       coarse={k: (round(v, 4) if v is not None else None) for k, v in metrics["coarse_iou"].items() if k != "pixels"},
                       seconds=round(time.time() - t1, 1))
            if metrics["score"] > best:
                best = metrics["score"]
                save_checkpoint(out / "checkpoints" / "best.pt", model, config, {"step": step, "epoch": epoch, "val": metrics})
                (out / "val_metrics.json").write_text(json.dumps({"epoch": epoch, **metrics}, indent=1))
            save_checkpoint(out / "checkpoints" / "last.pt", model, config, {"step": step, "epoch": epoch, "best": best})
        if args.max_steps and step >= args.max_steps:
            break
    logger.log(event="done", best_score=best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
