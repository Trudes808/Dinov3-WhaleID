"""Segment photos, then map every whale pixel to its point on the canonical whale surface.

    python -m model_fine_tuning.canon.predict --seg-checkpoint SEG --checkpoint CANON --images DIR --out OUT \
        [--focal-mm 400 --sensor-width-mm 36 | --hfov-deg 10]

Per image: <name>_mask.png (segmentation), <name>_canon.npz (crop-resolution signed canonical
points in mm, int16 (H, W, 3), the use mask, the side map, the mirror-disagreement map in mm,
and the crop transform x0, y0, s that maps full-image pixels to crop pixels), <name>_overlay.jpg
(crop | canonical colours with the resolved side | side map | mirror disagreement), and
points.jsonl with, per image, the points of interest read off the map (full-image pixels),
the mirror-consistency and side statistics (confidence), the direct rotation and the PnP pose
with the resolved side.

Every crop is also run mirrored; the two folded predictions are averaged (they should be
identical) and their disagreement is reported as a confidence. The side (left/right) is
decided once per whale by geometry.resolve_side. Without calibration the rotation is still
usable; scale/translation assume the guessed intrinsics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from ..common import geometry, paths
from ..common.data import bbox_of, square_crop_params, to_tensor, warp_image, warp_params
from ..common.viz import canon_rgb, draw_points, hstack, side_rgb
from ..segmentation.model import WhaleSegModel
from ..segmentation.predict import segment
from .evaluate import predict_crop
from .model import WhaleCanonModel


def keep_main_components(mask: np.ndarray, keep_frac: float = 0.25) -> np.ndarray:
    """Drop small stray blobs of a whale mask (false positives on an island or the horizon pull the crop
    away from the whale): keep the largest connected component and any component at least keep_frac of
    its size (a fluke and a back can be separate blobs)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 2:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.nonzero(areas >= keep_frac * areas.max())[0] + 1
    return np.isin(labels, keep)


@torch.no_grad()
def estimate(model: WhaleCanonModel, image: np.ndarray, mask: np.ndarray, device, K: np.ndarray, table: dict, landmarks: dict,
             crop: int = 512, margin: float = 1.6, tta: bool = True) -> dict | None:
    bbox = bbox_of(mask)
    if bbox is None:
        return None
    x0, y0, s = square_crop_params(bbox, crop, margin, 0.0, None)
    M = warp_params(x0, y0, s)
    crop_img = warp_image(image, M, (crop, crop))
    pred = predict_crop(model, to_tensor(crop_img)[None], device, tta=tta)
    seg_in_crop = cv2.warpAffine(mask.astype(np.uint8), M, (crop, crop), flags=cv2.INTER_NEAREST) > 0
    use = pred["vis"][0] & seg_in_crop
    pm = geometry.unfold_norm(np.moveaxis(pred["xyz"][0], 0, -1), table)
    pa = geometry.unfold_norm(np.moveaxis(pred["xyz_a"][0], 0, -1), table)
    pb = geometry.unfold_norm(np.moveaxis(pred["xyz_b"][0], 0, -1), table)
    disagree = np.linalg.norm(pa - pb, axis=-1)
    Kc = geometry.intrinsics_for_crop(K, x0, y0, s)
    prior, pconf = geometry.choose_prior_side(pred["R"][0], pred["xyz"][0], use)
    prior_b, pconf_b = geometry.choose_prior_side(pred["R_b"][0], pred["xyz_b"][0], use)
    res = geometry.resolve_side(pm, use, Kc, R_init=pred["R"][0], table=table, prior_side=prior, prior_confidence=pconf)
    res_b = geometry.resolve_side(pb, use, Kc, R_init=pred["R_b"][0], table=table, prior_side=prior_b, prior_confidence=pconf_b)
    cue, cue_conf = geometry.global_side_cue(pred["xyz"][0], use)
    side = res["side"]
    agree = use & (side != 0) & (res_b["side"] != 0)
    side_flip_agreement = float(np.mean(side[agree] == res_b["side"][agree])) if agree.sum() > 20 else None
    signed = geometry.signed_points(pm, side)
    lm = geometry.landmarks_from_map(signed, use & (side != 0), landmarks, max_dist_m=1.0)
    Minv = cv2.invertAffineTransform(M)
    for name, rec in lm.items():
        if rec["uv"] is not None:
            u, v = rec["uv"]
            rec["uv_full"] = (float(Minv[0, 0] * u + Minv[0, 1] * v + Minv[0, 2]), float(Minv[1, 0] * u + Minv[1, 1] * v + Minv[1, 2]))
    keypoints = None
    if "kp_heat" in pred:
        from ..common.run03_labels import KEYPOINTS
        from .model import decode_heatmaps

        uv, conf = decode_heatmaps(pred["kp_heat"][0], stride=crop // pred["kp_heat"].shape[-1])
        prob = pred["kp_state_prob"][0]
        keypoints = {"names": KEYPOINTS, "uv": uv, "conf": conf, "state": prob.argmax(-1), "state_prob": prob, "deform": pred["deform"][0],
                     "uv_full": np.stack([Minv[0, 0] * uv[:, 0] + Minv[0, 1] * uv[:, 1] + Minv[0, 2],
                                          Minv[1, 0] * uv[:, 0] + Minv[1, 1] * uv[:, 1] + Minv[1, 2]], 1)}
    return {"crop": crop_img, "x0y0s": (float(x0), float(y0), float(s)), "signed_m": signed, "folded_m": pm, "use": use, "side": side, "keypoints": keypoints,
            "disagree_m": disagree, "fit": res["fit"], "side_purity": res["purity"], "side_margin": res["margin"],
            "cue_side": cue, "cue_confidence": cue_conf, "head_side": geometry.initial_side_from_rotation(pred["R"][0]), "prior_side": prior, "side_flip_agreement": side_flip_agreement,
            "flip_disagreement_median_m": float(np.median(disagree[use])) if use.any() else None,
            "R_direct": pred["R"][0], "R_direct_flip_consistency_deg": geometry.geodesic_deg(pred["R"][0], pred["R_b"][0]),
            "ppm_direct": float(np.exp(pred["log_ppm"][0])) / s, "Kc": Kc, "landmarks": lm}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seg-checkpoint", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-side", type=int, default=1280)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--focal-mm", type=float, default=None)
    p.add_argument("--sensor-width-mm", type=float, default=36.0)
    p.add_argument("--hfov-deg", type=float, default=None)
    p.add_argument("--min-pixels", type=int, default=200)
    p.add_argument("--no-tta", action="store_true")
    args = p.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg_model, _ = WhaleSegModel.from_checkpoint(args.seg_checkpoint, device)
    model, _ = WhaleCanonModel.from_checkpoint(args.checkpoint, device)
    table = geometry.load_folded_table(paths.FOLDED_TABLE)
    landmarks = geometry.load_landmarks(paths.ASSETS_DIR / "landmarks_mesh_frame_cm.json")
    args.out.mkdir(parents=True, exist_ok=True)
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    files = [args.images] if args.images.is_file() else sorted(f for f in args.images.rglob("*") if f.suffix.lower() in exts)
    with open(args.out / "points.jsonl", "w") as handle:
        for path in files:
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            H, W = image.shape[:2]
            stem = path.stem
            prob, mask = segment(seg_model, image, device, args.max_side, args.threshold)
            raw_pixels = int(mask.sum())
            mask = keep_main_components(mask)
            rec = {"image": str(path), "whale_pixels": int(mask.sum()), "whale_pixels_before_cleanup": raw_pixels}
            cv2.imwrite(str(args.out / f"{stem}_mask.png"), mask.astype(np.uint8))
            if mask.sum() < args.min_pixels:
                rec["status"] = "no_whale"
                handle.write(json.dumps(rec) + "\n")
                print(rec)
                continue
            K = geometry.intrinsics_guess(W, H, args.focal_mm, args.sensor_width_mm, args.hfov_deg)
            est = estimate(model, image, mask, device, K, table, landmarks, tta=not args.no_tta)
            extra = {}
            if est.get("keypoints"):
                kp = est["keypoints"]
                extra = {"kp_uv": kp["uv"].astype(np.float32), "kp_conf": kp["conf"].astype(np.float32), "kp_state": kp["state"].astype(np.int8),
                         "kp_state_prob": kp["state_prob"].astype(np.float32), "deform": kp["deform"].astype(np.float32)}
            np.savez_compressed(args.out / f"{stem}_canon.npz", signed_mm=np.round(np.nan_to_num(est["signed_m"]) * 1000).astype(np.int16),
                                use=est["use"], side=est["side"], disagree_mm=np.round(est["disagree_m"] * 1000).astype(np.uint16), x0y0s=np.array(est["x0y0s"]), **extra)
            crop = est["crop"]
            colour = np.where(est["use"][..., None], 0.35 * crop + 0.65 * canon_rgb(np.moveaxis((est["folded_m"] - [table["x_min_m"], 0, 0]) / table["length_m"], -1, 0), est["side"], est["use"]), crop).astype(np.uint8)
            pts = {n: v["uv"] for n, v in est["landmarks"].items() if v["uv"] is not None}
            colour = draw_points(colour, pts, (255, 255, 0), 3)
            if est.get("keypoints"):
                kp = est["keypoints"]
                colour = draw_points(colour, {n: tuple(kp["uv"][k]) for k, n in enumerate(kp["names"]) if kp["state"][k] == 0 and kp["conf"][k] > 0.3}, (255, 0, 255), 4)
            heat = np.clip(est["disagree_m"] / 0.5, 0, 1)
            heat_rgb = cv2.cvtColor(cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            heat_panel = np.where(est["use"][..., None], heat_rgb, crop).astype(np.uint8)
            side_panel = np.where(est["use"][..., None], 0.4 * crop + 0.6 * side_rgb(est["side"], est["use"]), crop).astype(np.uint8)
            panel = hstack([crop, colour, side_panel, heat_panel])
            fd = est["flip_disagreement_median_m"]
            cv2.putText(panel, f"flip-disagree med={'n/a' if fd is None else f'{fd:.2f}m'} side agree={est['side_flip_agreement']} purity={est['side_purity']}",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(str(args.out / f"{stem}_overlay.jpg"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
            fit = est["fit"]
            rec.update({"status": "ok", "crop_x0y0s": est["x0y0s"], "flip_disagreement_median_m": est["flip_disagreement_median_m"],
                        "side_flip_agreement": est["side_flip_agreement"], "side_purity": est["side_purity"], "side_margin": est["side_margin"],
                        "cue_side": est["cue_side"], "cue_confidence": round(est["cue_confidence"], 3), "head_side": est["head_side"], "prior_side": est["prior_side"],
                        "rotation_direct": np.round(est["R_direct"], 4).tolist(), "rot_direct_flip_consistency_deg": round(est["R_direct_flip_consistency_deg"], 1),
                        "ppm_direct": round(est["ppm_direct"], 2), "intrinsics_fx": round(float(K[0, 0]), 1),
                        "landmarks": {n: {"uv_full": v.get("uv_full"), "dist_m": None if v["dist_m"] is None else round(v["dist_m"], 3)} for n, v in est["landmarks"].items()}})
            if est.get("keypoints"):
                kp = est["keypoints"]
                rec["keypoints"] = {n: {"uv_full": [round(float(x), 1) for x in kp["uv_full"][k]], "conf": round(float(kp["conf"][k]), 3), "state": int(kp["state"][k])}
                                    for k, n in enumerate(kp["names"])}
                rec["deform_deg"] = [round(float(v) * 30.0, 1) for v in kp["deform"]]
            if fit is not None:
                rec.update({"rotation_pnp": np.round(fit["R"], 4).tolist(), "translation_m_pnp": np.round(fit["t_m"], 2).tolist(), "ppm_pnp": round(fit["ppm"], 2),
                            "pnp_inliers": fit["inliers"], "pnp_inlier_fraction": round(fit["inlier_fraction"], 3),
                            "rot_direct_vs_pnp_deg": round(geometry.geodesic_deg(fit["R"], est["R_direct"]), 1)})
            handle.write(json.dumps(rec) + "\n")
            print({k: rec[k] for k in ("image", "whale_pixels", "flip_disagreement_median_m", "side_flip_agreement", "rot_direct_vs_pnp_deg") if k in rec})
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
