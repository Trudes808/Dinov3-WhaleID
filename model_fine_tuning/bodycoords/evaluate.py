"""Evaluate a body-coordinate checkpoint on a split, with oracle crops (GT mask bbox) and,
if a segmentation checkpoint is given, end-to-end crops from the predicted mask.

    python -m model_fine_tuning.bodycoords.evaluate --checkpoint experiments/body_x/checkpoints/best.pt \
        --seg-checkpoint experiments/seg_x/checkpoints/best.pt --split test

Reports dense errors (u MAE in body fractions and metres, angle MAE), visibility IoU,
rotation error from the direct head and from PnP, scale error, and landmark reprojection
error (PCK: fraction of visible landmarks reprojected within 10 % of the whale's image
length). Writes metrics.json, per_image.jsonl and overlay images.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..common import geometry, paths
from ..common.data import BodyCropDataset, SegCropDataset, WhaleRun, bbox_of
from ..common.metrics import summarize_values
from ..common.train_utils import move
from ..common.viz import bodycoord_overlay, bodycoord_rgb, draw_points, hstack
from .train import evaluate as evaluate_dense
from .model import WhaleBodyModel


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--seg-checkpoint", type=Path, default=None)
    p.add_argument("--run-dir", type=Path, default=paths.WHALE_RUN_DIR)
    p.add_argument("--split-file", type=Path, default=paths.SPLITS_DIR / "run01.json")
    p.add_argument("--split", default="test")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--min-visible", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--overlays", type=int, default=24)
    p.add_argument("--max-images", type=int, default=None)
    return p.parse_args(argv)


@torch.no_grad()
def predicted_bboxes(seg_ckpt: Path, run: WhaleRun, ids: list[str], device, batch_size: int, workers: int) -> dict:
    from ..segmentation.model import WhaleSegModel

    model, _ = WhaleSegModel.from_checkpoint(seg_ckpt, device)
    ds = SegCropDataset(run, ids, train=False)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=workers)
    boxes = {}
    for batch in loader:
        b = move(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prob = torch.sigmoid(model(b["image"]).float()).cpu().numpy()
        for i, sid in enumerate(batch["id"]):
            h, w = batch["orig_hw"][i].tolist()
            boxes[sid] = bbox_of(prob[i, :h, :w] >= 0.5)
    return boxes


def landmark_metrics(run: WhaleRun, per_image: list[dict], ds: BodyCropDataset, preds: dict, table, landmarks) -> None:
    """PCK of visible landmarks reprojected with the PnP pose, normalised by whale image length."""
    for row in per_image:
        sid = row["id"]
        meta = run.meta(sid)
        fit = preds.get(sid)
        if fit is None:
            row["pck_0.1"] = None
            continue
        K = geometry.intrinsics_from_meta(meta)
        names = [n for n in landmarks if meta["keypoints"][n]["in_frame"] and not meta["keypoints"][n]["occluded"]]
        if not names:
            row["pck_0.1"] = None
            continue
        uv, _ = geometry.project_ue(np.stack([landmarks[n] for n in names]), fit["R"], fit["t_m"], K)
        gt = np.array([meta["keypoints"][n]["uv"] for n in names])
        length_px = meta["pixels_per_metre_at_whale"] * meta["whale_length_m"]
        err = np.linalg.norm(uv - gt, axis=1) / length_px
        row["pck_0.1"] = float(np.mean(err < 0.1))
        row["landmark_err_frac_median"] = float(np.median(err))


@torch.no_grad()
def run_eval(model, run, ids, device, table, landmarks, bboxes, args, tag: str, out: Path) -> dict:
    ds = BodyCropDataset(run, ids, crop=512, min_visible=args.min_visible, train=False, bboxes=bboxes)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers)
    metrics = evaluate_dense(model, loader, device, amp=device.type == "cuda", table=table, run=run, pnp=True)
    per_image = metrics.pop("per_image")
    # re-run PnP on full-frame intrinsics to get poses for landmark reprojection + overlays
    preds, keep = {}, {}
    for batch in loader:
        b = move(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            o = model(b["image"])
        u = o["u"].float().cpu().numpy()
        sc = o["sincos"].float().cpu().numpy()
        theta = np.degrees(np.arctan2(sc[:, 0], sc[:, 1])) % 360.0
        vis = (o["vis_logit"].float() > 0).cpu().numpy()
        for i, sid in enumerate(batch["id"]):
            meta = run.meta(sid)
            x0, y0, s = batch["crop_x0y0s"][i].numpy()
            Kc = geometry.intrinsics_for_crop(geometry.intrinsics_from_meta(meta), x0, y0, s)
            use = vis[i] & (batch["mask"][i].numpy() != 255)
            preds[sid] = geometry.fit_pose_pnp(u[i], theta[i], use, Kc, table)
            keep[sid] = (batch["image"][i].permute(1, 2, 0).numpy() * 255).astype(np.uint8), u[i], theta[i], vis[i], batch["u"][i].numpy(), batch["theta_deg"][i].numpy(), batch["valid"][i].numpy(), (x0, y0, s)
    landmark_metrics(run, per_image, ds, preds, table, landmarks)
    pck = [r["pck_0.1"] for r in per_image if r.get("pck_0.1") is not None]
    metrics["pck_0.1_mean"] = float(np.mean(pck)) if pck else None
    metrics.update(summarize_values([r.get("landmark_err_frac_median") for r in per_image], "landmark_err_frac_"))
    metrics["u_mae_metres_mean"] = metrics["u_mae_mean"] * table["length_m"] if "u_mae_mean" in metrics else None
    metrics["crops"] = tag
    metrics["images_evaluated"] = len(per_image)
    (out / f"metrics_{tag}.json").write_text(json.dumps(metrics, indent=1))
    with open(out / f"per_image_{tag}.jsonl", "w") as handle:
        for row in per_image:
            handle.write(json.dumps(row) + "\n")
    # overlays: worst by angle error + random
    rows = sorted(per_image, key=lambda r: -(r["angle_mae_deg"] + 90 * r["u_mae"]))
    rng = np.random.default_rng(0)
    chosen = rows[: args.overlays // 2] + [rows[i] for i in rng.choice(len(rows), min(args.overlays // 2, len(rows)), replace=False)]
    (out / f"overlays_{tag}").mkdir(exist_ok=True)
    for r in chosen:
        sid = r["id"]
        img, u, theta, vis, gu, gth, gv, (x0, y0, s) = keep[sid]
        pred_over = bodycoord_overlay(img, u, theta, vis)
        gt_over = bodycoord_overlay(img, gu, gth, gv)
        fit = preds.get(sid)
        if fit is not None:
            meta = run.meta(sid)
            Kc = geometry.intrinsics_for_crop(geometry.intrinsics_from_meta(meta), x0, y0, s)
            uv, _ = geometry.project_ue(np.stack(list(landmarks.values())), fit["R"], fit["t_m"], Kc)
            pred_over = draw_points(pred_over, {n: tuple(p) for n, p in zip(landmarks, uv)}, (255, 255, 0), 3)
        panel = hstack([img, pred_over, gt_over])
        text = f"{sid} u_mae={r['u_mae']:.3f} ang={r['angle_mae_deg']:.1f} rot_direct={r['rot_direct_deg']:.1f} rot_pnp={r.get('rot_pnp_deg')}"
        cv2.putText(panel, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out / f"overlays_{tag}" / f"{sid}.jpg"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return metrics


def main(argv=None) -> int:
    args = parse(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = WhaleBodyModel.from_checkpoint(args.checkpoint, device)
    out = args.out or args.checkpoint.parents[1] / f"eval_{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    run = WhaleRun(args.run_dir, args.split_file if args.split != "all" else None)
    table = geometry.load_body_table(paths.BODY_TABLE)
    landmarks = geometry.load_landmarks(paths.ASSETS_DIR / "landmarks_mesh_frame_cm.json")
    ids = [i for i in run.ids(args.split) if run.by_id[i]["visible_pixels"] >= args.min_visible][: args.max_images]
    results = {"checkpoint": str(args.checkpoint), "epoch": ckpt.get("epoch"), "split": args.split}
    results["oracle_crops"] = run_eval(model, run, ids, device, table, landmarks, None, args, "oracle", out)
    if args.seg_checkpoint:
        boxes = predicted_bboxes(args.seg_checkpoint, run, ids, device, 4, args.workers)
        found = {k: v for k, v in boxes.items() if v is not None}
        results["seg_crops"] = run_eval(model, run, ids, device, table, landmarks, boxes, args, "segcrops", out)
        results["seg_crops"]["images_with_predicted_whale"] = len(found)
        results["seg_crops"]["images_total"] = len(ids)
    (out / "metrics.json").write_text(json.dumps(results, indent=1))
    for tag in ("oracle_crops", "seg_crops"):
        if tag in results:
            m = results[tag]
            print(f"[{tag}] n={m['images_evaluated']} u_mae={m.get('u_mae_mean'):.4f} ({m.get('u_mae_metres_mean'):.2f} m) angle_mae={m.get('angle_mae_deg_mean'):.1f} deg "
                  f"vis_iou={m.get('vis_iou_mean'):.3f} rot_direct med={m.get('rot_direct_deg_median'):.1f} <15:{m.get('rot_direct_under_15deg')} "
                  f"rot_pnp med={m.get('rot_pnp_deg_median')} <15:{m.get('rot_pnp_under_15deg')} pck0.1={m.get('pck_0.1_mean')} logscale_pnp med={m.get('logscale_pnp_median')}")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
