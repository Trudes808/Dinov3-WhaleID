"""Fine-tune DINOv3 for dense body coordinates + rigid pose.

    python -m model_fine_tuning.bodycoords.train --name body_vitb16_ft4 --unfreeze-blocks 4

Training crops are centred on the visible whale (ground-truth mask bbox with a random
margin and jitter); validation crops use the GT bbox with a fixed margin ("oracle crops").
End-to-end evaluation with crops from the segmentation model is in evaluate.py.
Selection metric: mean of (1 - u MAE * 4) and (1 - angle MAE / 90) on val whale pixels,
so both coordinates count; details in val_metrics.json.
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
from ..common.data import BodyCropDataset, WhaleRun
from ..common.metrics import angle_diff_deg, summarize_values
from ..common.train_utils import Logger, apply_lr, count_params, load_checkpoint, lr_at, move, save_checkpoint, set_seed
from .model import WhaleBodyModel, body_loss


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--run-dir", type=Path, default=paths.WHALE_RUN_DIR)
    p.add_argument("--split", type=Path, default=paths.SPLITS_DIR / "run01.json")
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
    p.add_argument("--w-u", type=float, default=10.0)
    p.add_argument("--w-angle", type=float, default=5.0)
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
    return p.parse_args(argv)


@torch.no_grad()
def evaluate(model, loader, device, amp: bool, table=None, run: WhaleRun | None = None, pnp: bool = True) -> dict:
    """Dense errors on valid pixels, visibility IoU, direct rotation error, and (optionally)
    PnP rotation / scale error from the dense predictions."""
    model.eval()
    u_err, ang_err, rot_direct, rot_pnp, scale_direct, scale_pnp, vis_iou, pnp_inl = [], [], [], [], [], [], [], []
    per_image = []
    for batch in loader:
        b = move(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out = model(b["image"])
        u = out["u"].float().cpu().numpy()
        sc = out["sincos"].float().cpu().numpy()
        theta = np.degrees(np.arctan2(sc[:, 0], sc[:, 1])) % 360.0
        vis = (out["vis_logit"].float() > 0).cpu().numpy()
        R_pred = out["R"].float().cpu().numpy()
        logppm = out["log_ppm"].float().cpu().numpy()
        valid = batch["valid"].numpy()
        gt_u, gt_theta, mask = batch["u"].numpy(), batch["theta_deg"].numpy(), batch["mask"].numpy()
        R_gt, logppm_gt = batch["R"].numpy(), batch["log_ppm"].numpy()
        for i, sid in enumerate(batch["id"]):
            v = valid[i]
            if v.sum() == 0:
                continue
            ue = float(np.abs(u[i][v] - gt_u[i][v]).mean())
            ae = float(angle_diff_deg(theta[i][v], gt_theta[i][v]).mean())
            gtm = mask[i] == 1
            inter = np.sum(vis[i] & gtm)
            union = np.sum((vis[i] | gtm) & (mask[i] != 255))
            viou = float(inter / max(union, 1))
            rd = geometry.geodesic_deg(R_pred[i], R_gt[i])
            sd = float(abs(logppm[i] - logppm_gt[i]))
            row = {"id": sid, "u_mae": ue, "angle_mae_deg": ae, "vis_iou": viou, "rot_direct_deg": rd, "logscale_direct": sd, "valid_pixels": int(v.sum())}
            u_err.append(ue); ang_err.append(ae); vis_iou.append(viou); rot_direct.append(rd); scale_direct.append(sd)
            if pnp and table is not None and run is not None:
                meta = run.meta(sid)
                x0, y0, s = batch["crop_x0y0s"][i].numpy()
                Kc = geometry.intrinsics_for_crop(geometry.intrinsics_from_meta(meta), x0, y0, s)
                use = vis[i] & (mask[i] != 255)
                fit = geometry.fit_pose_pnp(u[i], theta[i], use, Kc, table)
                if fit is not None:
                    rp = geometry.geodesic_deg(fit["R"], R_gt[i])
                    sp = float(abs(np.log(fit["ppm"]) - logppm_gt[i]))
                    row.update({"rot_pnp_deg": rp, "logscale_pnp": sp, "pnp_inlier_fraction": fit["inlier_fraction"]})
                    rot_pnp.append(rp); scale_pnp.append(sp); pnp_inl.append(fit["inlier_fraction"])
                else:
                    row.update({"rot_pnp_deg": None})
            per_image.append(row)
    summary = {"images": len(per_image)}
    summary.update(summarize_values(u_err, "u_mae_"))
    summary.update(summarize_values(ang_err, "angle_mae_deg_"))
    summary.update(summarize_values(vis_iou, "vis_iou_"))
    summary.update(summarize_values(rot_direct, "rot_direct_deg_"))
    summary.update(summarize_values(scale_direct, "logscale_direct_"))
    if rot_pnp:
        summary.update(summarize_values(rot_pnp, "rot_pnp_deg_"))
        summary.update(summarize_values(scale_pnp, "logscale_pnp_"))
        summary["pnp_success_rate"] = len(rot_pnp) / max(len(per_image), 1)
        summary["rot_pnp_under_15deg"] = float(np.mean(np.asarray(rot_pnp) < 15))
        summary["rot_pnp_under_30deg"] = float(np.mean(np.asarray(rot_pnp) < 30))
    summary["rot_direct_under_15deg"] = float(np.mean(np.asarray(rot_direct) < 15)) if rot_direct else None
    summary["rot_direct_under_30deg"] = float(np.mean(np.asarray(rot_direct) < 30)) if rot_direct else None
    summary["score"] = float(np.mean([1 - min(summary.get("u_mae_mean", 1) * 4, 1), 1 - min(summary.get("angle_mae_deg_mean", 90) / 90, 1)]))
    summary["per_image"] = per_image
    return summary


def main(argv=None) -> int:
    args = parse(argv)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = not args.no_amp and device.type == "cuda"
    out = args.out_root / args.name
    out.mkdir(parents=True, exist_ok=True)
    run = WhaleRun(args.run_dir, args.split)
    table = geometry.load_body_table(paths.BODY_TABLE) if paths.BODY_TABLE.exists() else None
    train_ds = BodyCropDataset(run, run.ids("train")[: args.max_train], crop=args.crop, min_visible=args.min_visible,
                               margin_range=(args.margin_min, args.margin_max), jitter=args.jitter, photometric_strength=args.photometric,
                               train=True, seed=args.seed)
    val_ds = BodyCropDataset(run, run.ids("val")[: args.max_val], crop=args.crop, min_visible=args.min_visible, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=min(args.workers, 4), pin_memory=True)
    model = WhaleBodyModel(arch=args.arch, unfreeze_blocks=args.unfreeze_blocks, layers=args.layers).to(device)
    config = {"task": "bodycoords", "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "model": model.config, "train_images": len(train_ds), "val_images": len(val_ds), "params": count_params(model)}
    (out / "config.json").write_text(json.dumps(config, indent=1))
    logger = Logger(out)
    logger.log(event="start", **{k: v for k, v in config.items() if k != "args"})
    weights = {"vis": args.w_vis, "u": args.w_u, "angle": args.w_angle, "rot": args.w_rot, "scale": args.w_scale}
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
            loss, parts = body_loss(out_, b, weights)
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
            metrics = evaluate(model, val_loader, device, amp, table, run, pnp=True)
            per_image = metrics.pop("per_image")
            logger.log(event="val", epoch=epoch, step=step, **{k: v for k, v in metrics.items() if isinstance(v, (int, float)) and ("mean" in k or "median" in k or "score" in k or "under" in k or "rate" in k)})
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
