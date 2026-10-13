"""Label-free pose-quality metric on real photos: how well the humpback mesh, posed by PnP on
the model's dense canonical map, matches the segmentation above the water line.

    python -m model_fine_tuning.canon.outline_metric --canon-dir experiments/real_2024-12-18/canon_r02_ft4_v2 \
        --images experiments/real_2024-12-18/images --timestamps experiments/real_2024-12-18/timestamps.json [--out metrics.json]

Per photo: RANSAC PnP (geometry.fit_pose_pnp_points) from every whale pixel's predicted point to
the pixel, the part of the mesh above the water plane (level camera, plane at the image row where the
visible whale meets the water, at the fitted distance) rendered with that pose, and its
intersection over union with the predicted whale mask: a pose that raises parts the photo does
not show, or hides parts it does, scores badly. Also the inlier fraction and reprojection error of the fit.
No labels are used, so it can be run for any checkpoint on any day's photos.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from ..common import geometry, paths

SENSOR_WIDTH_MM = {"Canon EOS 7D Mark II": 22.3, "Canon EOS R5": 36.0}
SIL = 128


def silhouette(V, F, R, t, K, crop: int, size: int = SIL, face_ok=None) -> np.ndarray:
    P, depth = geometry.project_ue(V, R, t, K)
    P = P * (size / crop)
    ok = (depth[F] > 0.5).all(1) & (np.abs(P[F]).max(axis=(1, 2)) < 20 * size)
    if face_ok is not None:
        ok &= face_ok
    img = np.zeros((size, size), np.uint8)
    tris = np.round(P[F[ok]]).astype(np.int32)
    if len(tris):
        cv2.fillPoly(img, tris, 255)
    return img > 0


def above_water_silhouette(V, F, R, t, K, water_row_crop: float, crop: int, size: int = SIL):
    """Silhouette of the part of the posed mesh above the water plane (level camera; the plane's
    image row at the whale's distance is where the visible whale meets the water), and of all of it.
    Same definition as Whale-Review-Web/backend/wire.py."""
    cam = V @ R.T + t
    z_w = -(water_row_crop - K[1, 2]) * float(t[0]) / K[1, 1]
    above = cam[:, 2] > z_w
    face_above = above[F].mean(1) >= 0.5
    return silhouette(V, F, R, t, K, crop, size, face_ok=face_above), silhouette(V, F, R, t, K, crop, size)


def keypoint_correspondences(z, landmarks: dict, min_conf: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """Detected visible keypoints of a run03 prediction (kp_* in the npz) as mesh-point / pixel pairs."""
    from ..common.run03_labels import KEYPOINTS

    if "kp_uv" not in z.files:
        return np.zeros((0, 3)), np.zeros((0, 2))
    uv, conf, state = z["kp_uv"], z["kp_conf"], z["kp_state"]
    keep = [k for k, n in enumerate(KEYPOINTS) if state[k] == 0 and conf[k] >= min_conf and n in landmarks]
    if not keep:
        return np.zeros((0, 3)), np.zeros((0, 2))
    return np.stack([landmarks[KEYPOINTS[k]] for k in keep]).astype(np.float64), uv[keep].astype(np.float64)


def score_photo(z, K, V, F, landmarks: dict | None = None, kp_weight: int = 0) -> dict:
    use = z["use"]; signed = z["signed_mm"].astype(np.float32) / 1000.0
    crop = use.shape[0]
    ys, xs = np.nonzero(use)
    if len(xs) < 50:
        return {"ok": False, "reason": "too few whale pixels"}
    pts = signed[ys, xs].astype(np.float64); pix = np.stack([xs + 0.5, ys + 0.5], 1).astype(np.float64)
    if len(pts) > 3000:
        pick = np.random.default_rng(0).choice(len(pts), 3000, replace=False)
        pts, pix = pts[pick], pix[pick]
    n_kp = 0
    if kp_weight > 0 and landmarks is not None:
        kp_pts, kp_pix = keypoint_correspondences(z, landmarks)
        n_kp = len(kp_pts)
        if n_kp:
            pts = np.concatenate([pts, np.repeat(kp_pts, kp_weight, 0)]); pix = np.concatenate([pix, np.repeat(kp_pix, kp_weight, 0)])
    fit = geometry.fit_pose_pnp_points(pts, pix, K, reproj_px=8.0, max_points=len(pts))
    if fit is None:
        return {"ok": False, "reason": "pnp failed"}
    proj, _ = geometry.project_ue(pts, fit["R"], fit["t_m"], K)
    err = np.linalg.norm(proj - pix, axis=1)
    mask = cv2.resize(use.astype(np.uint8), (SIL, SIL), interpolation=cv2.INTER_AREA) > 0
    rows = np.nonzero(use.any(1))[0]
    water_row = float(np.percentile(rows, 99)) + 3.0 * (crop / SIL)
    sil_above, sil_all = above_water_silhouette(V, F, fit["R"], fit["t_m"], K, water_row, crop)
    iou = float((sil_above & mask).sum() / max((sil_above | mask).sum(), 1))
    cov = float((sil_above & mask).sum() / max(mask.sum(), 1))
    exc = float((sil_above & ~mask).sum() / max(sil_above.sum(), 1))
    return {"ok": True, "iou_above_water": round(iou, 4), "coverage": round(cov, 4), "excess_above_water": round(exc, 4), "inlier_fraction": round(fit["inlier_fraction"], 4),
            "median_reproj_px": round(float(np.median(err)), 2), "distance_m": round(float(fit["t_m"][0]), 1), "whale_pixels": int(len(xs)), "keypoints_used": int(n_kp)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--canon-dir", type=Path, required=True); p.add_argument("--images", type=Path, required=True); p.add_argument("--timestamps", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None); p.add_argument("--mesh", type=Path, default=paths.WHALE_MESH_OBJ); p.add_argument("--limit", type=int, default=0)
    p.add_argument("--kp-weight", type=int, default=0, help="run03 keypoint heads: add each detected visible keypoint this many times to the PnP points (0 = dense map only)")
    p.add_argument("--landmarks", type=Path, default=paths.ASSETS_DIR / "landmarks_run03_mesh_frame_cm.json", help="mesh-frame positions of the 18 keypoints")
    a = p.parse_args(argv)
    V = geometry.load_obj_vertices(a.mesh) / 100.0; F = geometry.load_obj_faces(a.mesh)[::3]
    landmarks = geometry.load_landmarks(a.landmarks) if a.kp_weight > 0 else None
    ts = json.loads(a.timestamps.read_text())
    rows = {}
    files = sorted(a.canon_dir.glob("*_canon.npz"))
    if a.limit:
        files = files[: a.limit]
    for npz in files:
        stem = npz.name[: -len("_canon.npz")]
        z = np.load(npz); x0, y0, s = z["x0y0s"]
        img = cv2.imread(str(a.images / f"{stem}.jpg")); H, W = img.shape[:2]
        meta = ts.get(stem, {}); focal = float(meta["focal_mm"]) if meta.get("focal_mm") else None
        K = geometry.intrinsics_for_crop(geometry.intrinsics_guess(W, H, focal, SENSOR_WIDTH_MM.get(meta.get("camera"), 36.0)), x0, y0, s)
        rows[stem] = score_photo(z, K, V, F, landmarks, a.kp_weight)
    ok = [r for r in rows.values() if r["ok"]]
    summary = {"photos": len(rows), "scored": len(ok), "canon_dir": str(a.canon_dir), "kp_weight": a.kp_weight,
               "photos_with_keypoints": int(sum(r.get("keypoints_used", 0) > 0 for r in ok))}
    for k in ("iou_above_water", "coverage", "excess_above_water", "inlier_fraction", "median_reproj_px"):
        v = np.array([r[k] for r in ok]); summary[k] = {"mean": round(float(v.mean()), 4), "median": round(float(np.median(v)), 4), "p25": round(float(np.percentile(v, 25)), 4), "p75": round(float(np.percentile(v, 75)), 4)}
    summary["iou_above_0.5"] = round(float(np.mean([r["iou_above_water"] >= 0.5 for r in ok])), 4)
    print(json.dumps(summary, indent=1))
    if a.out:
        a.out.write_text(json.dumps({"summary": summary, "photos": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
