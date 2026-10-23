"""Train the continuous-surface-embedding head (canon/cse.py): which template vertex every whale pixel shows.

    python -m model_fine_tuning.canon.train_cse --name cse_r04_frozen --run-dir RUN --split SPLIT --unfreeze-blocks 0
    python -m model_fine_tuning.canon.train_cse --name cse_r04_ft2 --run-dir RUN --split SPLIT --unfreeze-blocks 2

Needs vertex/<id>.png labels in the run (common.vertex_labels). Selection metric: the fraction of labelled val
pixels whose predicted vertex lies within 30 cm (geodesic) of the true one.
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
from ..common.data import IGNORE, CanonCropDataset, open_runs
from ..common.metrics import summarize_values
from ..common.train_utils import Logger, apply_lr, count_params, load_checkpoint, lr_at, move, save_checkpoint, set_seed
from .cse import Template, WhaleCSEModel, crop_metrics, cse_loss, predict_vertices, side_vote


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--run-dir", type=Path, default=paths.WHALE_RUN_DIR); p.add_argument("--split", type=Path, required=True)
    p.add_argument("--extra-run-dir", type=Path, nargs="*", default=None); p.add_argument("--extra-split", type=Path, nargs="*", default=None)
    p.add_argument("--out-root", type=Path, default=paths.EXPERIMENTS_DIR)
    p.add_argument("--arch", default="vitb16"); p.add_argument("--unfreeze-blocks", type=int, default=0); p.add_argument("--layers", type=int, nargs="*", default=None)
    p.add_argument("--emb-dim", type=int, default=32); p.add_argument("--tau", type=float, default=0.07); p.add_argument("--pixels-per-crop", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=30); p.add_argument("--batch-size", type=int, default=8); p.add_argument("--crop", type=int, default=512)
    p.add_argument("--min-visible", type=int, default=300); p.add_argument("--margin-min", type=float, default=1.25); p.add_argument("--margin-max", type=float, default=2.2)
    p.add_argument("--jitter", type=float, default=0.12); p.add_argument("--photometric", type=float, default=1.0)
    p.add_argument("--lr-head", type=float, default=3e-4); p.add_argument("--lr-backbone", type=float, default=3e-5); p.add_argument("--layer-decay", type=float, default=0.8)
    p.add_argument("--weight-decay", type=float, default=0.05); p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--w-vis", type=float, default=1.0); p.add_argument("--w-vertex", type=float, default=1.0); p.add_argument("--w-rot", type=float, default=2.0); p.add_argument("--w-scale", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=6); p.add_argument("--eval-every", type=int, default=1); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-train", type=int, default=None); p.add_argument("--max-val", type=int, default=None); p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--resume", type=Path, default=None); p.add_argument("--no-amp", action="store_true")
    return p.parse_args(argv)


@torch.no_grad()
def evaluate(model: WhaleCSEModel, loader, device, amp: bool, template: Template) -> dict:
    model.eval()
    rows = []
    for batch in loader:
        b = move(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out = model(b["image"])
        emb = out["emb"].float(); vis = out["vis_logit"].float() > 0
        R = out["R"].float().cpu().numpy()
        for i, sid in enumerate(batch["id"]):
            gtv = b["vertex"][i]
            labelled = gtv > 0
            if labelled.sum() < 20:
                continue
            use_pred = vis[i] & (b["mask"][i] != IGNORE)
            ids_all, conf_all, mirror_p = predict_vertices(model, emb[i], use_pred | labelled, template)
            yx = torch.nonzero(use_pred | labelled, as_tuple=True)
            side, ids_snapped = side_vote(ids_all, conf_all, template, R[i], mirror_p)
            lab = labelled[yx]
            row = {"id": sid, "side_vote": side}
            row.update(crop_metrics(ids_snapped[lab], gtv[yx][lab] - 1, use_pred, gtv, template))
            raw = crop_metrics(ids_all[lab], gtv[yx][lab] - 1, use_pred, gtv, template)
            row["side_acc_raw"] = raw.get("side_acc"); row["within_30cm_raw"] = raw.get("within_30cm"); row["within_60cm_raw"] = raw.get("within_60cm")
            gtm = b["mask"][i] == 1
            row["vis_iou"] = float((vis[i] & gtm).sum() / max(((vis[i] | gtm) & (b["mask"][i] != IGNORE)).sum(), 1))
            rows.append(row)
    summary = {"images": len(rows)}
    for key in ("geodesic_cm_median", "geodesic_cm_mean", "within_30cm", "within_30cm_raw", "within_60cm", "within_60cm_raw", "part_acc", "side_acc", "side_acc_raw",
                "coverage_err_mean", "vis_iou"):
        summary.update(summarize_values([r.get(key) for r in rows], key + "_"))
    summary["score"] = summary.get("within_30cm_raw_mean", 0.0)          # selection on the raw map (the vote is an inference-time option)
    summary["per_image"] = rows
    return summary


def main(argv=None) -> int:
    args = parse(argv)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = not args.no_amp and device.type == "cuda"
    out = args.out_root / args.name
    out.mkdir(parents=True, exist_ok=True)
    template = Template(device=device)
    mirror = template.mirror.cpu().numpy()
    run = open_runs(args.run_dir, args.split, args.extra_run_dir, args.extra_split)
    from ..common import geometry
    table = geometry.load_folded_table(paths.FOLDED_TABLE)
    train_ds = CanonCropDataset(run, run.ids("train")[: args.max_train], table, crop=args.crop, min_visible=args.min_visible, margin_range=(args.margin_min, args.margin_max),
                                jitter=args.jitter, photometric_strength=args.photometric, train=True, seed=args.seed, vertex_mirror=mirror)
    val_ds = CanonCropDataset(run, run.ids("val")[: args.max_val], table, crop=args.crop, min_visible=args.min_visible, train=False, vertex_mirror=mirror)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=min(args.workers, 4), pin_memory=True)
    model = WhaleCSEModel(template.n, arch=args.arch, unfreeze_blocks=args.unfreeze_blocks, layers=args.layers, emb_dim=args.emb_dim, tau=args.tau).to(device)
    config = {"task": "cse", "args": {k: (str(v) if isinstance(v, Path) else [str(x) for x in v] if isinstance(v, list) and v and isinstance(v[0], Path) else v) for k, v in vars(args).items()},
              "model": model.config, "train_images": len(train_ds), "val_images": len(val_ds), "params": count_params(model), "template": str(template.mesh)}
    (out / "config.json").write_text(json.dumps(config, indent=1))
    logger = Logger(out)
    logger.log(event="start", **{k: v for k, v in config.items() if k != "args"})
    weights = {"vis": args.w_vis, "vertex": args.w_vertex, "rot": args.w_rot, "scale": args.w_scale}
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
        t0 = time.time(); running = []
        for batch in train_loader:
            factor = lr_at(step, total_steps, args.warmup_steps, 1.0)
            apply_lr(optimizer, factor)
            b = move(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                out_ = model(b["image"])
            out_ = {k: v.float() for k, v in out_.items()}
            loss, parts = cse_loss(out_, b, model, template, weights, args.pixels_per_crop)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], 1.0)
            optimizer.step()
            running.append(loss.item()); step += 1
            if step % 50 == 0:
                logger.log(event="train", epoch=epoch, step=step, loss=float(np.mean(running[-50:])), **parts, lr_factor=factor)
            if args.max_steps and step >= args.max_steps:
                break
        logger.log(event="epoch", epoch=epoch, step=step, loss=float(np.mean(running)) if running else None, seconds=round(time.time() - t0, 1))
        last = epoch == args.epochs - 1 or (args.max_steps and step >= args.max_steps)
        if (epoch + 1) % args.eval_every == 0 or last:
            metrics = evaluate(model, val_loader, device, amp, template)
            per_image = metrics.pop("per_image")
            logger.log(event="val", epoch=epoch, step=step, **{k: v for k, v in metrics.items() if isinstance(v, (int, float)) and ("mean" in k or "median" in k or "score" in k)})
            score = metrics["score"]
            extra = {"optimizer": optimizer.state_dict(), "step": step, "epoch": epoch, "best": best, "val": metrics}
            if score > best:
                best = score; extra["best"] = best
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
