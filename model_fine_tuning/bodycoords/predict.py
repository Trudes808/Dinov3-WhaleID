"""Segment photos, then estimate body coordinates and the rigid 3D pose of the whale.

    python -m model_fine_tuning.bodycoords.predict --seg-checkpoint SEG --checkpoint BODY --images DIR --out OUT \
        [--focal-mm 400 --sensor-width-mm 36 | --hfov-deg 10]

Per image: <name>_mask.png, <name>_bodycoords.png (same encoding as the training labels:
R = fraction along body, G = angle/360, B = 255 where whale), <name>_overlay.jpg (crop |
predicted coordinates with reprojected landmarks), and poses.jsonl with the rotation
(camera-from-whale, Unreal camera frame: x forward, y right, z up), the translation in
metres (needs the focal length to be meaningful), pixels-per-metre, and PnP inlier stats.
Without calibration the rotation is still usable; the scale/translation assume the guessed
intrinsics.
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
from ..common.viz import bodycoord_overlay, bodycoord_rgb, draw_points, hstack
from ..segmentation.model import WhaleSegModel
from ..segmentation.predict import list_images, load_rgb, segment
from .model import WhaleBodyModel


@torch.no_grad()
def estimate(body_model: WhaleBodyModel, image: np.ndarray, mask: np.ndarray, device, K: np.ndarray, table, crop: int = 512, margin: float = 1.6):
    bbox = bbox_of(mask)
    if bbox is None:
        return None
    x0, y0, s = square_crop_params(bbox, crop, margin, 0.0, None)
    M = warp_params(x0, y0, s)
    crop_img = warp_image(image, M, (crop, crop))
    x = to_tensor(crop_img)[None].to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        o = body_model(x)
    u = o["u"][0].float().cpu().numpy()
    sc = o["sincos"][0].float().cpu().numpy()
    theta = np.degrees(np.arctan2(sc[0], sc[1])) % 360.0
    vis = (o["vis_logit"][0].float() > 0).cpu().numpy()
    seg_in_crop = cv2.warpAffine(mask.astype(np.uint8), M, (crop, crop), flags=cv2.INTER_NEAREST) > 0
    use = vis & seg_in_crop
    Kc = geometry.intrinsics_for_crop(K, x0, y0, s)
    fit = geometry.fit_pose_pnp(u, theta, use, Kc, table)
    # map dense coordinates back to the full image
    Minv = cv2.invertAffineTransform(M)
    H, W = mask.shape
    u_full = cv2.warpAffine(u, Minv, (W, H), flags=cv2.INTER_LINEAR)
    theta_full = cv2.warpAffine(theta, Minv, (W, H), flags=cv2.INTER_NEAREST)
    valid_full = cv2.warpAffine(use.astype(np.uint8), Minv, (W, H), flags=cv2.INTER_NEAREST) > 0
    return {"crop": crop_img, "x0y0s": (x0, y0, s), "u": u, "theta": theta, "use": use, "fit": fit, "Kc": Kc,
            "R_direct": o["R"][0].float().cpu().numpy(), "ppm_direct": float(np.exp(o["log_ppm"][0].float().item())) / s,
            "u_full": u_full, "theta_full": theta_full, "valid_full": valid_full}


def describe_rotation(R: np.ndarray) -> dict:
    """Readable summary: where the whale's nose, right side and back point in camera terms."""
    nose, right, back = R[:, 0], R[:, 1], R[:, 2]
    return {"nose_toward_camera": round(float(-nose[0]), 3), "nose_right": round(float(nose[1]), 3), "nose_up": round(float(nose[2]), 3),
            "back_up": round(float(back[2]), 3), "right_side_toward_camera": round(float(-right[0]), 3)}


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
    args = p.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg_model, _ = WhaleSegModel.from_checkpoint(args.seg_checkpoint, device)
    body_model, _ = WhaleBodyModel.from_checkpoint(args.checkpoint, device)
    table = geometry.load_body_table(paths.BODY_TABLE)
    landmarks = geometry.load_landmarks(paths.ASSETS_DIR / "landmarks_mesh_frame_cm.json")
    args.out.mkdir(parents=True, exist_ok=True)
    files = list_images(args.images)
    if not files:
        print("no images in", args.images)
        return 1
    with open(args.out / "poses.jsonl", "w") as handle:
        for path in files:
            image = load_rgb(path)
            H, W = image.shape[:2]
            prob, mask = segment(seg_model, image, device, args.max_side, args.threshold)
            rec = {"image": str(path), "whale_pixels": int(mask.sum())}
            stem = path.stem
            cv2.imwrite(str(args.out / f"{stem}_mask.png"), mask.astype(np.uint8))
            if mask.sum() < args.min_pixels:
                rec["status"] = "no_whale"
                handle.write(json.dumps(rec) + "\n")
                print(rec)
                continue
            K = geometry.intrinsics_guess(W, H, args.focal_mm, args.sensor_width_mm, args.hfov_deg)
            est = estimate(body_model, image, mask, device, K, table)
            cv2.imwrite(str(args.out / f"{stem}_bodycoords.png"), cv2.cvtColor(bodycoord_rgb(est["u_full"], est["theta_full"], est["valid_full"]), cv2.COLOR_RGB2BGR))
            over = bodycoord_overlay(est["crop"], est["u"], est["theta"], est["use"])
            fit = est["fit"]
            rec.update({"status": "ok", "rotation_direct": np.round(est["R_direct"], 4).tolist(), "direct": describe_rotation(est["R_direct"]),
                        "ppm_direct": round(est["ppm_direct"], 2), "intrinsics_fx": round(float(K[0, 0]), 1)})
            if fit is not None:
                uv, _ = geometry.project_ue(np.stack(list(landmarks.values())), fit["R"], fit["t_m"], est["Kc"])
                over = draw_points(over, {n: tuple(pt) for n, pt in zip(landmarks, uv)}, (255, 255, 0), 3)
                rec.update({"rotation_pnp": np.round(fit["R"], 4).tolist(), "pnp": describe_rotation(fit["R"]), "translation_m_pnp": np.round(fit["t_m"], 2).tolist(),
                            "ppm_pnp": round(fit["ppm"], 2), "pnp_inliers": fit["inliers"], "pnp_inlier_fraction": round(fit["inlier_fraction"], 3),
                            "rot_direct_vs_pnp_deg": round(geometry.geodesic_deg(fit["R"], est["R_direct"]), 1)})
            cv2.imwrite(str(args.out / f"{stem}_overlay.jpg"), cv2.cvtColor(hstack([est["crop"], over]), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
            handle.write(json.dumps(rec) + "\n")
            print({k: v for k, v in rec.items() if k not in ("rotation_direct", "rotation_pnp")})
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
