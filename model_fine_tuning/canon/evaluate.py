"""Evaluate a canonical-surface checkpoint on a split: oracle crops (GT mask bbox) and, with a
segmentation checkpoint, end-to-end crops from the predicted mask.

    python -m model_fine_tuning.canon.evaluate --checkpoint experiments/canon_x/checkpoints/best.pt \
        --seg-checkpoint experiments/seg_x/checkpoints/best.pt --run-dir RUN --split-file SPLIT --split test

Per image: surface error (metres between the predicted and true canonical point, by region),
mirror consistency (the same crop mirrored must give the same folded points and, after side
resolution, the same side), side accuracy after resolution, rotation error (direct head and
PnP with the resolved side), and the points of interest read off the dense map against the
rendered keypoints (PCK at 5 % and 10 % of the whale's image length; PnP-reprojected
landmarks reported too for comparison with the old pipeline).
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
from ..common.data import IGNORE, CanonCropDataset, WhaleRun
from ..common.metrics import summarize_values
from ..common.train_utils import move
from ..common.viz import canon_rgb, draw_points, hstack, side_rgb
from ..bodycoords.evaluate import predicted_bboxes
from .model import WhaleCanonModel


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--seg-checkpoint", type=Path, default=None)
    p.add_argument("--run-dir", type=Path, default=paths.WHALE_RUN_DIR)
    p.add_argument("--split-file", type=Path, default=paths.SPLITS_DIR / "run02.json")
    p.add_argument("--split", default="test")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--min-visible", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--overlays", type=int, default=24)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--no-tta", action="store_true", help="do not average the prediction with its mirror")
    return p.parse_args(argv)


@torch.no_grad()
def predict_crop(model, image_t: torch.Tensor, device, tta: bool = True) -> dict:
    """image_t (B, 3, H, W) in [0, 1] -> folded xyz (B, 3, H, W) numpy, vis bool, R, log_ppm, and
    the mirror-consistency: folded prediction of the mirrored crop mapped back."""
    amp = device.type == "cuda"
    x = image_t.to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        a = model(x)
        b = model(x.flip(-1))
    xyz_a = a["xyz"].float().cpu().numpy()
    xyz_b = b["xyz"].float().cpu().numpy()[..., ::-1]              # un-mirror: same folded target
    vis_a = (a["vis_logit"].float() > 0).cpu().numpy()
    vis_b = (b["vis_logit"].float() > 0).cpu().numpy()[..., ::-1]
    R_a = a["R"].float().cpu().numpy()
    R_b = np.stack([geometry.flip_rotation(r) for r in b["R"].float().cpu().numpy()])
    xyz = 0.5 * (xyz_a + xyz_b) if tta else xyz_a
    vis = (vis_a & vis_b) if tta else vis_a
    out = {"xyz": xyz, "xyz_a": xyz_a, "xyz_b": xyz_b, "vis": vis, "vis_a": vis_a, "vis_b": vis_b, "R": R_a, "R_b": R_b,
           "log_ppm": a["log_ppm"].float().cpu().numpy()}
    if "kp_heat" in a:
        from ..common.run03_labels import FLIP_PAIRS, KEYPOINTS

        swap = list(range(len(KEYPOINTS)))
        for p, q in FLIP_PAIRS:
            i, j = KEYPOINTS.index(p), KEYPOINTS.index(q)
            swap[i], swap[j] = j, i
        heat_a = a["kp_heat"].float().cpu().numpy()
        heat_b = b["kp_heat"].float().cpu().numpy()[:, swap][..., ::-1]
        st_a = torch.softmax(a["kp_state_logit"].float(), -1).cpu().numpy()
        st_b = torch.softmax(b["kp_state_logit"].float(), -1).cpu().numpy()[:, swap]
        out["kp_heat"] = 0.5 * (heat_a + heat_b) if tta else heat_a
        out["kp_state_prob"] = 0.5 * (st_a + st_b) if tta else st_a
        out["deform"] = a["deform"].float().cpu().numpy()
    return out


def landmark_pck(pred_uv: dict, meta: dict, names: list[str], x0: float, y0: float, s: float, length_px: float) -> dict:
    gt = {n: (np.array(meta["keypoints"][n]["uv"], float) - (x0, y0)) * s for n in names}
    errs = []
    for n in names:
        p = pred_uv.get(n, {}).get("uv")
        errs.append(np.inf if p is None else float(np.linalg.norm(np.array(p) - gt[n]) / length_px))
    errs = np.array(errs)
    return {"pck_0.05": float(np.mean(errs < 0.05)), "pck_0.1": float(np.mean(errs < 0.1)), "err_frac_median": float(np.median(errs[np.isfinite(errs)])) if np.isfinite(errs).any() else None,
            "found": int(np.isfinite(errs).sum()), "n": len(names)}


def keypoint_rows(heat, state_prob, deform_pred, uv_gt, state_gt, deform_gt, crop: int, length_px: float, Kc, landmarks: dict, R_gt) -> dict:
    """Keypoint-head metrics for one crop (heatmap PCK on the visible keypoints, state accuracy, visible
    F1, deformation error) and the rotation from a PnP on the detected keypoints alone, which is the
    head's value for views where the dense map is ambiguous (fin tips, rolled whales)."""
    from ..common.run03_labels import KEYPOINTS
    from .model import DEFORM_SCALE_DEG, decode_heatmaps

    uv, conf = decode_heatmaps(heat, stride=crop // heat.shape[-1])
    state_pred = state_prob.argmax(-1)
    vis = state_gt == 0
    row = {}
    if vis.any() and length_px > 1:
        err = np.linalg.norm(uv[vis] - uv_gt[vis], axis=1) / length_px
        row.update({"kp_pck_0.05": float(np.mean(err < 0.05)), "kp_pck_0.1": float(np.mean(err < 0.1)), "kp_visible_n": int(vis.sum())})
    known = state_gt >= 0
    if known.any():
        row["kp_state_acc"] = float(np.mean(state_pred[known] == state_gt[known]))
        tp = np.sum((state_pred == 0) & vis); fp = np.sum((state_pred == 0) & known & ~vis); fn = np.sum((state_pred != 0) & vis)
        row["kp_visible_f1"] = float(2 * tp / max(2 * tp + fp + fn, 1))
    row["deform_err_deg"] = float(np.abs(deform_pred - deform_gt).mean() * DEFORM_SCALE_DEG)
    detected = (state_pred == 0) & (conf > 0.3) & np.array([n in landmarks for n in KEYPOINTS])
    row["kp_detected_n"] = int(detected.sum())
    if detected.sum() >= 5:
        pts = np.stack([landmarks[KEYPOINTS[k]] for k in np.nonzero(detected)[0]])
        fit = geometry.fit_pose_pnp_points(np.repeat(pts, 3, 0), np.repeat(uv[detected], 3, 0), Kc, reproj_px=max(0.03 * length_px, 4.0), iterations=200)
        if fit is not None:
            row["rot_kppnp_deg"] = geometry.geodesic_deg(fit["R"], R_gt)
    return row


@torch.no_grad()
def run_eval(model, run: WhaleRun, ids, device, table, args, tag: str, out: Path, bboxes=None) -> dict:
    ds = CanonCropDataset(run, ids, table, crop=512, min_visible=args.min_visible, train=False, bboxes=bboxes, keypoints=bool(getattr(model, "keypoints", 0)))
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers)
    L = table["length_m"]
    landmarks = run.landmarks_m
    per_image, keep = [], {}
    for batch in loader:
        pred = predict_crop(model, batch["image"], device, tta=not args.no_tta)
        for i, sid in enumerate(batch["id"]):
            meta = run.meta(sid)
            v = batch["valid"][i].numpy()
            if v.sum() == 0:
                continue
            mask = batch["mask"][i].numpy()
            x0, y0, s = batch["crop_x0y0s"][i].numpy()
            Kc = geometry.intrinsics_for_crop(geometry.intrinsics_from_meta(meta), x0, y0, s)
            gt_m = geometry.unfold_norm(np.moveaxis(batch["xyz"][i].numpy(), 0, -1), table)
            gt_side = batch["side"][i].numpy()
            pm = geometry.unfold_norm(np.moveaxis(pred["xyz"][i], 0, -1), table)
            pa = geometry.unfold_norm(np.moveaxis(pred["xyz_a"][i], 0, -1), table)
            pb = geometry.unfold_norm(np.moveaxis(pred["xyz_b"][i], 0, -1), table)
            use = pred["vis"][i] & (mask != IGNORE)
            err = np.linalg.norm(pm - gt_m, axis=-1)
            region = geometry.mesh_region(gt_m, table)
            row = {"id": sid, "valid_pixels": int(v.sum()), "surface_err_m": float(err[v].mean()), "surface_err_median_m": float(np.median(err[v]))}
            for rid, name in ((0, "torso"), (1, "fluke"), (2, "fin")):
                sel = v & (region == rid)
                row[f"surface_err_{name}_m"] = float(err[sel].mean()) if sel.sum() > 20 else None
            both = pred["vis_a"][i] & pred["vis_b"][i] & (mask != IGNORE)
            row["flip_surface_disagreement_m"] = float(np.linalg.norm(pa - pb, axis=-1)[both].mean()) if both.sum() else None
            gtm = mask == 1
            row["vis_iou"] = float(np.sum(pred["vis"][i] & gtm) / max(np.sum((pred["vis"][i] | gtm) & (mask != IGNORE)), 1))
            R_gt = batch["R"][i].numpy()
            row["rot_direct_deg"] = geometry.geodesic_deg(pred["R"][i], R_gt)
            row["rot_direct_flip_consistency_deg"] = geometry.geodesic_deg(pred["R"][i], pred["R_b"][i])
            # side resolution (once per whale), on the TTA map and on each single-pass map for the consistency check
            prior, pconf = geometry.choose_prior_side(pred["R"][i], pred["xyz"][i], use)
            prior_a, pconf_a = geometry.choose_prior_side(pred["R"][i], pred["xyz_a"][i], use)
            prior_b, pconf_b = geometry.choose_prior_side(pred["R_b"][i], pred["xyz_b"][i], use)
            res = geometry.resolve_side(pm, use, Kc, R_init=pred["R"][i], table=table, prior_side=prior, prior_confidence=pconf)
            res_a = geometry.resolve_side(pa, use, Kc, R_init=pred["R"][i], table=table, prior_side=prior_a, prior_confidence=pconf_a)
            res_b = geometry.resolve_side(pb, use, Kc, R_init=pred["R_b"][i], table=table, prior_side=prior_b, prior_confidence=pconf_b)
            cue, cue_conf = geometry.global_side_cue(pred["xyz"][i], use)
            row["cue_side"] = cue; row["cue_confidence"] = cue_conf; row["head_side"] = geometry.initial_side_from_rotation(pred["R"][i])
            row["facing_gt"] = geometry.initial_side_from_rotation(R_gt); row["prior_side"] = prior
            side = res["side"]
            scored = use & v & (gt_side != 0) & (np.abs(gt_m[..., 1]) > 0.1)
            row["side_acc"] = float(np.mean(side[scored] == gt_side[scored])) if scored.sum() > 20 else None
            agree = use & (res_a["side"] != 0) & (res_b["side"] != 0)
            row["side_flip_agreement"] = float(np.mean(res_a["side"][agree] == res_b["side"][agree])) if agree.sum() > 20 else None
            row["side_purity"] = res["purity"]
            row["side_margin"] = res["margin"]
            row["side_agrees_with_prior"] = res.get("agrees_with_prior")
            fit = res["fit"]
            if fit is not None:
                row["rot_pnp_deg"] = geometry.geodesic_deg(fit["R"], R_gt)
                row["logscale_pnp"] = float(abs(np.log(fit["ppm"]) - batch["log_ppm"][i].item()))
                row["pnp_inlier_fraction"] = fit["inlier_fraction"]
            # points of interest: from the dense map (signed) and, for comparison, PnP-reprojected
            names = [n for n in landmarks if n in meta["keypoints"] and geometry.keypoint_visible(meta["keypoints"][n])]
            length_px = meta["pixels_per_metre_at_whale"] * meta["whale_length_m"] * s
            signed = geometry.signed_points(pm, side)
            lm_map = geometry.landmarks_from_map(signed, use & (side != 0), landmarks, max_dist_m=1.0)
            if names:
                m = landmark_pck(lm_map, meta, names, x0, y0, s, length_px)
                row.update({"lm_map_pck_0.05": m["pck_0.05"], "lm_map_pck_0.1": m["pck_0.1"], "lm_map_err_frac_median": m["err_frac_median"], "lm_map_found": m["found"], "lm_n": m["n"],
                            "lm_map_found_frac": m["found"] / m["n"]})
                if fit is not None:
                    uv, _ = geometry.project_ue(np.stack([landmarks[n] for n in names]), fit["R"], fit["t_m"], Kc)
                    m2 = landmark_pck({n: {"uv": tuple(p)} for n, p in zip(names, uv)}, meta, names, x0, y0, s, length_px)
                    row.update({"lm_pnp_pck_0.05": m2["pck_0.05"], "lm_pnp_pck_0.1": m2["pck_0.1"]})
            if "kp_heat" in pred and "kp_uv" in batch:
                row.update(keypoint_rows(pred["kp_heat"][i], pred["kp_state_prob"][i], pred["deform"][i], batch["kp_uv"][i].numpy(), batch["kp_state"][i].numpy(),
                                         batch["deform"][i].numpy(), mask.shape[-1], length_px, Kc, landmarks, R_gt))
            row["category"] = meta.get("category")
            per_image.append(row)
            keep[sid] = {"img": (batch["image"][i].permute(1, 2, 0).numpy() * 255).astype(np.uint8), "pred": pred["xyz"][i], "use": use, "side": side,
                         "gt": batch["xyz"][i].numpy(), "gt_valid": v, "gt_side": gt_side, "lm_map": lm_map, "names": names, "x0y0s": (x0, y0, s)}
    summary = {"crops": tag, "images_evaluated": len(per_image)}
    keys = ["surface_err_m", "surface_err_median_m", "surface_err_torso_m", "surface_err_fluke_m", "surface_err_fin_m", "flip_surface_disagreement_m", "vis_iou",
            "rot_direct_deg", "rot_direct_flip_consistency_deg", "rot_pnp_deg", "logscale_pnp", "pnp_inlier_fraction", "side_acc", "side_flip_agreement", "side_purity",
            "lm_map_pck_0.05", "lm_map_pck_0.1", "lm_map_err_frac_median", "lm_pnp_pck_0.05", "lm_pnp_pck_0.1", "side_margin", "lm_map_found_frac", "cue_confidence",
            "kp_pck_0.05", "kp_pck_0.1", "kp_state_acc", "kp_visible_f1", "deform_err_deg", "rot_kppnp_deg", "kp_detected_n"]
    for key in keys:
        summary.update(summarize_values([r.get(key) for r in per_image], key + "_"))
    for key, thr in (("rot_direct_deg", 15), ("rot_pnp_deg", 15), ("rot_direct_deg", 30), ("rot_pnp_deg", 30), ("rot_kppnp_deg", 15), ("rot_kppnp_deg", 30)):
        vals = np.asarray([r[key] for r in per_image if r.get(key) is not None])
        summary[f"{key}_under_{thr}"] = float(np.mean(vals < thr)) if len(vals) else None
    summary["rot_kppnp_images"] = sum(r.get("rot_kppnp_deg") is not None for r in per_image)
    # per pose category: the hard views (close-ups, rolled whales) are what run03 is for
    by_cat = {}
    for cat in sorted({r.get("category") for r in per_image} - {None}):
        rs = [r for r in per_image if r.get("category") == cat]
        entry = {"n": len(rs)}
        for key in ("surface_err_median_m", "side_acc", "lm_map_pck_0.1", "kp_pck_0.1", "kp_state_acc", "deform_err_deg"):
            vals = [r[key] for r in rs if r.get(key) is not None]
            entry[key] = float(np.median(vals)) if vals else None
        for key in ("rot_direct_deg", "rot_pnp_deg", "rot_kppnp_deg"):
            vals = np.asarray([r[key] for r in rs if r.get(key) is not None])
            entry[f"{key}_under_15"] = float(np.mean(vals < 15)) if len(vals) else None
            entry[f"{key}_n"] = int(len(vals))
        by_cat[cat] = entry
    summary["by_category"] = by_cat
    summary["surface_err_frac_of_length_mean"] = summary.get("surface_err_m_mean", np.nan) / L
    summary["facing_acc_head"] = float(np.mean([r["head_side"] == r["facing_gt"] for r in per_image if "facing_gt" in r])) if per_image else None
    summary["facing_acc_prior"] = float(np.mean([r["prior_side"] == r["facing_gt"] for r in per_image if "facing_gt" in r])) if per_image else None
    summary["side_unstable_crops_frac"] = float(np.mean([r["side_flip_agreement"] < 0.8 for r in per_image if r.get("side_flip_agreement") is not None])) if per_image else None
    (out / f"metrics_{tag}.json").write_text(json.dumps(summary, indent=1))
    with open(out / f"per_image_{tag}.jsonl", "w") as handle:
        for row in per_image:
            handle.write(json.dumps(row) + "\n")
    # overlays: worst by surface error + random
    rows = sorted(per_image, key=lambda r: -r["surface_err_m"])
    rng = np.random.default_rng(0)
    chosen = rows[: args.overlays // 2] + [rows[i] for i in rng.choice(len(rows), min(args.overlays // 2, len(rows)), replace=False)]
    (out / f"overlays_{tag}").mkdir(exist_ok=True)
    for r in chosen:
        k = keep[r["id"]]
        img = k["img"]
        pred_rgb = canon_rgb(k["pred"], k["side"], k["use"])
        gt_rgb = canon_rgb(k["gt"], k["gt_side"], k["gt_valid"])
        blend = lambda c, m: np.where(m[..., None], (0.35 * img + 0.65 * c), img).astype(np.uint8)
        pred_panel = blend(pred_rgb, k["use"])
        pts = {n: v["uv"] for n, v in k["lm_map"].items() if v["uv"] is not None and n in k["names"]}
        pred_panel = draw_points(pred_panel, pts, (255, 255, 0), 3)
        meta = run.meta(r["id"]); x0, y0, s = k["x0y0s"]
        gt_pts = {n: tuple((np.array(meta["keypoints"][n]["uv"]) - (x0, y0)) * s) for n in k["names"]}
        gt_panel = draw_points(blend(gt_rgb, k["gt_valid"]), gt_pts, (0, 255, 0), 3)
        panel = hstack([img, pred_panel, gt_panel, blend(side_rgb(k["side"], k["use"]), k["use"]), blend(side_rgb(k["gt_side"], k["gt_valid"]), k["gt_valid"])])
        text = (f"{r['id']} surf={r['surface_err_m']:.2f}m side_acc={r.get('side_acc')} flipagree={r.get('side_flip_agreement')} "
                f"rot_direct={r['rot_direct_deg']:.0f} rot_pnp={r.get('rot_pnp_deg')} pck.05={r.get('lm_map_pck_0.05')}")
        cv2.putText(panel, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out / f"overlays_{tag}" / f"{r['id']}.jpg"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return summary


def main(argv=None) -> int:
    args = parse(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = WhaleCanonModel.from_checkpoint(args.checkpoint, device)
    out = args.out or args.checkpoint.parents[1] / f"eval_{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    run = WhaleRun(args.run_dir, args.split_file if args.split != "all" else None)
    table = geometry.load_folded_table(paths.FOLDED_TABLE)
    ids = [i for i in run.ids(args.split) if run.by_id[i]["visible_pixels"] >= args.min_visible and run.has_canon(i)][: args.max_images]
    results = {"checkpoint": str(args.checkpoint), "epoch": ckpt.get("epoch"), "split": args.split, "tta": not args.no_tta}
    results["oracle_crops"] = run_eval(model, run, ids, device, table, args, "oracle", out)
    if args.seg_checkpoint:
        boxes = predicted_bboxes(args.seg_checkpoint, run, ids, device, 4, args.workers)
        results["seg_crops"] = run_eval(model, run, ids, device, table, args, "segcrops", out, bboxes=boxes)
        results["seg_crops"]["images_with_predicted_whale"] = sum(v is not None for v in boxes.values())
        results["seg_crops"]["images_total"] = len(ids)
    (out / "metrics.json").write_text(json.dumps(results, indent=1))
    for tag in ("oracle_crops", "seg_crops"):
        if tag in results:
            m = results[tag]
            print(f"[{tag}] n={m['images_evaluated']} surface err mean={m.get('surface_err_m_mean'):.3f} m (median {m.get('surface_err_median_m_median'):.3f}; "
                  f"torso {m.get('surface_err_torso_m_mean')} fluke {m.get('surface_err_fluke_m_mean')} fin {m.get('surface_err_fin_m_mean')}) "
                  f"flip-disagree={m.get('flip_surface_disagreement_m_mean'):.3f} m  side_acc={m.get('side_acc_mean')} side_flip_agree={m.get('side_flip_agreement_mean')} "
                  f"unstable={m.get('side_unstable_crops_frac')}  rot_direct med={m.get('rot_direct_deg_median'):.1f} rot_pnp med={m.get('rot_pnp_deg_median')} "
                  f"PCK map .05/.1={m.get('lm_map_pck_0.05_mean')}/{m.get('lm_map_pck_0.1_mean')}  PCK pnp .1={m.get('lm_pnp_pck_0.1_mean')}")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
