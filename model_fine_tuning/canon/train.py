"""Fine-tune DINOv3 for canonical surface points (+ rigid pose).

    python -m model_fine_tuning.canon.train --name canon_r02_ft4 --run-dir .../run02 --split model_fine_tuning/splits/run02.json

Needs the canon/<id>.npz labels in the run (factory.postprocess --canon-only) and the folded
table asset (build with `python -m model_fine_tuning.canon.train --build-table`).
Selection metric: 1 - min(10 * mean surface error in body lengths, 1), on val whale pixels.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..common import geometry, paths
from ..common.data import CanonCropDataset, RealCropDataset, WhaleRun
from ..common.metrics import summarize_values
from ..common.train_utils import Logger, apply_lr, count_params, load_checkpoint, lr_at, move, save_checkpoint, set_seed
from .model import WhaleCanonModel, canon_loss, consistency_loss


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default=None)
    p.add_argument("--build-table", action="store_true", help="only build the folded table asset from the mesh and exit")
    p.add_argument("--run-dir", type=Path, default=paths.WHALE_RUN_DIR)
    p.add_argument("--split", type=Path, default=paths.SPLITS_DIR / "run02.json")
    p.add_argument("--out-root", type=Path, default=paths.EXPERIMENTS_DIR)
    p.add_argument("--arch", default="vitb16")
    p.add_argument("--unfreeze-blocks", type=int, default=4)
    p.add_argument("--layers", type=int, nargs="*", default=None)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--min-visible", type=int, default=300)
    p.add_argument("--margin-min", type=float, default=1.25)
    p.add_argument("--margin-max", type=float, default=2.2)
    p.add_argument("--jitter", type=float, default=0.12)
    p.add_argument("--photometric", type=float, default=1.0)
    p.add_argument("--lr-head", type=float, default=3e-4)
    p.add_argument("--lr-backbone", type=float, default=3e-5)
    p.add_argument("--layer-decay", type=float, default=0.8)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--w-vis", type=float, default=1.0)
    p.add_argument("--w-xyz", type=float, default=10.0)
    p.add_argument("--w-rot", type=float, default=2.0)
    p.add_argument("--w-scale", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-val", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--init-from", type=Path, default=None, help="initialise weights from a checkpoint (fresh optimizer)")
    p.add_argument("--real-images", type=Path, default=None, help="unlabelled real photos (jpg) for the mirror-consistency loss")
    p.add_argument("--real-masks", type=Path, default=None, help="their predicted masks (<stem>_mask.png)")
    p.add_argument("--w-consistency", type=float, default=5.0)
    p.add_argument("--w-consistency-rot", type=float, default=1.0)
    p.add_argument("--real-batch-size", type=int, default=4)
    return p.parse_args(argv)


def build_table() -> dict:
    vertices = geometry.load_obj_vertices(paths.WHALE_MESH_OBJ)
    table = geometry.build_folded_table(vertices, geometry.load_obj_faces(paths.WHALE_MESH_OBJ))
    np.savez_compressed(paths.FOLDED_TABLE, **table)
    print(f"folded table: {len(table['folded_m'])} vertices, length {table['length_m']:.2f} m -> {paths.FOLDED_TABLE}")
    return table


@torch.no_grad()
def evaluate(model, loader, device, amp: bool, table: dict) -> dict:
    """Dense surface error (metres, on valid pixels, by region), visibility IoU, direct rotation
    error. No side resolution / PnP here (evaluate.py does that)."""
    model.eval()
    per_image = []
    for batch in loader:
        b = move(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out = model(b["image"])
        xyz = out["xyz"].float().cpu().numpy()
        vis = (out["vis_logit"].float() > 0).cpu().numpy()
        R_pred = out["R"].float().cpu().numpy()
        logppm = out["log_ppm"].float().cpu().numpy()
        valid = batch["valid"].numpy()
        gt = batch["xyz"].numpy()
        mask = batch["mask"].numpy()
        R_gt, logppm_gt = batch["R"].numpy(), batch["log_ppm"].numpy()
        for i, sid in enumerate(batch["id"]):
            v = valid[i]
            if v.sum() == 0:
                continue
            pred_m = geometry.unfold_norm(np.moveaxis(xyz[i], 0, -1), table)
            gt_m = geometry.unfold_norm(np.moveaxis(gt[i], 0, -1), table)
            err = np.linalg.norm(pred_m - gt_m, axis=-1)
            region = geometry.mesh_region(gt_m, table)
            row = {"id": sid, "surface_err_m": float(err[v].mean()), "surface_err_median_m": float(np.median(err[v])), "valid_pixels": int(v.sum())}
            for rid, name in ((0, "torso"), (1, "fluke"), (2, "fin")):
                sel = v & (region == rid)
                row[f"surface_err_{name}_m"] = float(err[sel].mean()) if sel.sum() > 20 else None
            gtm = mask[i] == 1
            inter = np.sum(vis[i] & gtm)
            union = np.sum((vis[i] | gtm) & (mask[i] != 255))
            row["vis_iou"] = float(inter / max(union, 1))
            row["rot_direct_deg"] = geometry.geodesic_deg(R_pred[i], R_gt[i])
            row["logscale_direct"] = float(abs(logppm[i] - logppm_gt[i]))
            per_image.append(row)
    summary = {"images": len(per_image)}
    for key in ("surface_err_m", "surface_err_median_m", "surface_err_torso_m", "surface_err_fluke_m", "surface_err_fin_m", "vis_iou", "rot_direct_deg", "logscale_direct"):
        summary.update(summarize_values([r.get(key) for r in per_image], key + "_"))
    rd = np.asarray([r["rot_direct_deg"] for r in per_image])
    summary["rot_direct_under_15deg"] = float(np.mean(rd < 15)) if len(rd) else None
    summary["rot_direct_under_30deg"] = float(np.mean(rd < 30)) if len(rd) else None
    summary["score"] = float(1 - min(10 * summary.get("surface_err_m_mean", 1.0) / table["length_m"], 1.0))
    summary["per_image"] = per_image
    return summary


def main(argv=None) -> int:
    args = parse(argv)
    if args.build_table:
        build_table()
        return 0
    if not args.name:
        raise SystemExit("--name is required")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = not args.no_amp and device.type == "cuda"
    out = args.out_root / args.name
    out.mkdir(parents=True, exist_ok=True)
    run = WhaleRun(args.run_dir, args.split)
    table = geometry.load_folded_table(paths.FOLDED_TABLE) if paths.FOLDED_TABLE.exists() else build_table() | {"tree": None}
    if table.get("tree") is None:
        table = geometry.load_folded_table(paths.FOLDED_TABLE)
    train_ds = CanonCropDataset(run, run.ids("train")[: args.max_train], table, crop=args.crop, min_visible=args.min_visible,
                                margin_range=(args.margin_min, args.margin_max), jitter=args.jitter, photometric_strength=args.photometric,
                                train=True, seed=args.seed)
    val_ds = CanonCropDataset(run, run.ids("val")[: args.max_val], table, crop=args.crop, min_visible=args.min_visible, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=min(args.workers, 4), pin_memory=True)
    model = WhaleCanonModel(arch=args.arch, unfreeze_blocks=args.unfreeze_blocks, layers=args.layers).to(device)
    if args.init_from:
        model.load_state_dict(load_checkpoint(args.init_from)["model"], strict=True)
    real_loader, real_iter = None, None
    if args.real_images:
        real_ds = RealCropDataset(args.real_images, args.real_masks or args.real_images, crop=args.crop, margin_range=(args.margin_min, args.margin_max),
                                  jitter=args.jitter, photometric_strength=args.photometric, seed=args.seed)
        real_loader = DataLoader(real_ds, batch_size=args.real_batch_size, shuffle=True, num_workers=min(args.workers, 4), pin_memory=True, drop_last=True,
                                 persistent_workers=True)
        print(f"mirror-consistency on {len(real_ds)} real crops", flush=True)
    config = {"task": "canon", "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "model": model.config, "train_images": len(train_ds), "val_images": len(val_ds), "params": count_params(model),
              "real_images": len(real_loader.dataset) if real_loader is not None else 0}
    (out / "config.json").write_text(json.dumps(config, indent=1))
    logger = Logger(out)
    logger.log(event="start", **{k: v for k, v in config.items() if k != "args"})
    weights = {"vis": args.w_vis, "xyz": args.w_xyz, "rot": args.w_rot, "scale": args.w_scale}
    optimizer = torch.optim.AdamW(model.param_groups(args.lr_head, args.lr_backbone, args.layer_decay, args.weight_decay))
    total_steps = args.epochs * len(train_loader)
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
            b = move(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                out_ = model(b["image"])
            out_ = {k: v.float() for k, v in out_.items()}
            loss, parts = canon_loss(out_, b, weights)
            if real_loader is not None:
                try:
                    rb = next(real_iter)
                except (StopIteration, TypeError):
                    real_loader.dataset.set_epoch(epoch)
                    real_iter = iter(real_loader)
                    rb = next(real_iter)
                rb = move(rb, device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    ra = model(rb["image"])
                    rbm = model(rb["image"].flip(-1))
                ra = {k: v.float() for k, v in ra.items()}
                rbm = {k: v.float() for k, v in rbm.items()}
                closs, cparts = consistency_loss(ra, rbm, rb["whale"], w_rot=args.w_consistency_rot)
                loss = loss + args.w_consistency * closs
                parts.update(cparts)
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
        last = epoch == args.epochs - 1 or (args.max_steps and step >= args.max_steps)
        if (epoch + 1) % args.eval_every == 0 or last:
            metrics = evaluate(model, val_loader, device, amp, table)
            per_image = metrics.pop("per_image")
            logger.log(event="val", epoch=epoch, step=step, **{k: v for k, v in metrics.items() if isinstance(v, (int, float)) and ("mean" in k or "median" in k or "score" in k or "under" in k)})
            score = metrics["score"]
            extra = {"optimizer": optimizer.state_dict(), "step": step, "epoch": epoch, "best": best, "val": metrics}
            if score > best:
                best = score
                extra["best"] = best
                save_checkpoint(out / "checkpoints" / "best.pt", model, config, {"step": step, "epoch": epoch, "val": metrics})
                (out / "val_metrics.json").write_text(json.dumps(metrics, indent=1))
                with open(out / "val_per_image.jsonl", "w") as handle:
                    for row in per_image:
                        handle.write(json.dumps(row) + "\n")
            save_checkpoint(out / "checkpoints" / "last.pt", model, config, extra)
        if last:
            break
    logger.log(event="done", best_val_score=best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
