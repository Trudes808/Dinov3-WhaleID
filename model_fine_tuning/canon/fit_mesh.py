"""Bendable mesh fit: the pose (rotation, translation) and the body bend (arch, tail curl, head pitch, lateral sweep)
of the canonical whale mesh, fitted to a canonical-surface prediction and the whale mask, instead of a rigid PnP.

    python -m model_fine_tuning.canon.fit_mesh --canon-dir experiments/real_2024-12-18/canon_canon_r04_ft4 \
        --images experiments/real_2024-12-18/images --timestamps experiments/real_2024-12-18/timestamps.json \
        --out experiments/real_2024-12-18/fit_canon_r04_ft4 [--mesh OBJ] [--limit N] [--overlays 24] [--workers 8]

Per photo: (1) rigid start: the direct rotation head and the PnP translation of the prediction (points.jsonl);
(2) bend fit: robust least squares on the dense correspondences (predicted canonical point -> bent -> posed ->
projected -> its pixel) over 10 parameters (rotation increment, translation, arch, tail curl, head pitch, lateral
sweep); (3) silhouette refinement (Powell) on the same residual plus the top-edge terms of the rendered above-water
silhouette against the mask: the per-column top edge of the mesh must follow the mask's top edge (the whale's own
silhouette) and not stick out beyond it; the bottom edge (waterline) is never matched.
Outputs: per photo the above-water IoU, coverage, excess and the top-edge error for the rigid PnP pose, the rigid
direct-rotation pose, the bend fit and the refined fit; overlays with the silhouettes; a summary json. The bend
model is the factory's (factory/deform.py), imported from the sibling repository.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from ..common import geometry, paths
from .outline_metric import SENSOR_WIDTH_MM, SIL, above_water_silhouette

sys.path.insert(0, str(paths.WHALE_FACTORY_ROOT))
from factory import deform as _deform  # noqa: E402

BEND_KEYS = ("arch_deg", "tail_curl_deg", "head_pitch_deg", "lateral_deg")
BEND_BOUNDS = {"arch_deg": (-90.0, 60.0), "tail_curl_deg": (-70.0, 70.0), "head_pitch_deg": (-45.0, 45.0), "lateral_deg": (-45.0, 45.0)}
N_CORR = 1500          # correspondences used in the fits
TOP_W = 3.0            # weight of the top-edge terms in the refinement, in units of the correspondence residual
DEG_SCALE = 10.0       # a step of one in the bend parameters = 10 degrees (least-squares scaling)
PRIOR_DEG = 25.0       # soft prior: a bend of this many degrees costs as much as one correspondence off by PRIOR_PX
PRIOR_PX = 6.0
PRIOR_ROWS = 40        # the prior is repeated this many times among the N_CORR correspondences (its weight)
FIT_SIL = 256          # silhouette resolution inside the refinement objective (the metric itself stays at SIL)
EXCESS_W = 1.5         # weight of the mesh-above-water-outside-the-mask fraction in the refinement (raised fins / flukes the photo lacks)
DEPTH_RANGE = (0.6, 1.6)   # the fitted distance stays within this factor of the start (a huge whale right in front of the camera is a degenerate optimum)


class Mesh:
    def __init__(self, obj: Path):
        self.V_cm = geometry.load_obj_vertices(obj)
        self.F = geometry.load_obj_faces(obj)
        self.x_min, self.x_max = float(self.V_cm[:, 0].min()), float(self.V_cm[:, 0].max())
        self.anchors = np.array([[self.x_min, 0.0, 0.0], [self.x_max, 0.0, 0.0]])


def params_from(theta_bend: np.ndarray) -> _deform.DeformParams:
    return _deform.DeformParams(**{k: float(v) * DEG_SCALE for k, v in zip(BEND_KEYS, theta_bend)})


def bend_points_m(points_m: np.ndarray, params: _deform.DeformParams, mesh: Mesh) -> np.ndarray:
    """Bend arbitrary canonical surface points (metres) with the mesh's body range (the factory's deform takes the
    range from the array it bends, so the two extreme mesh vertices ride along as anchors)."""
    if params.is_identity():
        return points_m
    pts = np.concatenate([points_m * 100.0, mesh.anchors], 0)
    return _deform.deform(pts, params)[:-2] / 100.0


def pose_from(theta: np.ndarray, R0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R_inc, _ = cv2.Rodrigues(np.asarray(theta[:3], float))
    return R_inc @ R0, np.asarray(theta[3:6], float)


def corr_residual(theta, pts_m, pix, K, R0, mesh: Mesh, prior: bool = True) -> np.ndarray:
    R, t = pose_from(theta, R0)
    bent = bend_points_m(pts_m, params_from(theta[6:]), mesh)
    proj, depth = geometry.project_ue(bent, R, t, K)
    r = proj - pix
    r[depth <= 0.5] = 100.0
    if not prior:
        return r.ravel()
    pri = np.asarray(theta[6:], float) * DEG_SCALE / PRIOR_DEG * PRIOR_PX          # px-equivalent cost of each bend
    return np.concatenate([r.ravel(), np.tile(pri, PRIOR_ROWS)])


def bend_bounds(t0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nb = len(BEND_KEYS)
    lo = np.full(6 + nb, -np.inf); hi = np.full(6 + nb, np.inf)
    lo[:3], hi[:3] = -0.6, 0.6                         # rotation increment (rad) about the direct head's rotation
    lo[3], hi[3] = max(float(t0[0]) * DEPTH_RANGE[0], 1.0), max(float(t0[0]) * DEPTH_RANGE[1], 2.0)
    lo[4:6], hi[4:6] = -0.6 * float(t0[0]), 0.6 * float(t0[0])
    for i, k in enumerate(BEND_KEYS):
        lo[6 + i], hi[6 + i] = BEND_BOUNDS[k][0] / DEG_SCALE, BEND_BOUNDS[k][1] / DEG_SCALE
    return lo, hi


def init_translation(pts_m: np.ndarray, pix: np.ndarray, K: np.ndarray, R: np.ndarray, iters: int = 3) -> np.ndarray | None:
    """Translation for a fixed rotation by linear least squares: with cam = R p + t (UE camera frame, x forward),
    u = cx + fx cam_y / cam_x and v = cy - fy cam_z / cam_x are linear in t once multiplied out. A few reweighted
    passes (Huber on the reprojection error) take the weight off the wrong side / outliers. None if degenerate."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    rp = pts_m @ R.T
    u, v = pix[:, 0] - cx, pix[:, 1] - cy
    # u (rp_x + t_x) - fx (rp_y + t_y) = 0 ;  v (rp_x + t_x) + fy (rp_z + t_z) = 0
    A = np.concatenate([np.stack([u, -fx * np.ones_like(u), np.zeros_like(u)], 1), np.stack([v, np.zeros_like(v), fy * np.ones_like(v)], 1)], 0)
    b = np.concatenate([fx * rp[:, 1] - u * rp[:, 0], -fy * rp[:, 2] - v * rp[:, 0]], 0)
    w = np.ones(len(b))
    t = None
    for _ in range(iters):
        sol, *_ = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)
        t = sol
        proj, depth = geometry.project_ue(pts_m, R, t, K)
        err = np.linalg.norm(proj - pix, axis=1)
        wp = np.where(err < 8.0, 1.0, 8.0 / np.maximum(err, 1e-6))
        w = np.concatenate([wp, wp])
    if t is None or not np.isfinite(t).all() or t[0] <= 1.0:
        return None
    return t


def fit_bend(pts_m, pix, K, R0, t0, mesh: Mesh, rigid_only: bool = False) -> np.ndarray:
    from scipy.optimize import least_squares

    nb = len(BEND_KEYS)
    n = 6 if rigid_only else 6 + nb
    theta0 = np.zeros(n); theta0[3:6] = t0
    lo, hi = bend_bounds(t0); lo, hi = lo[:n], hi[:n]
    theta0 = np.clip(theta0, lo + 1e-6, hi - 1e-6)                           # a start with a large lateral offset would sit outside
    x_scale = np.ones(n); x_scale[:3] = 0.05; x_scale[3:6] = max(float(t0[0]) * 0.02, 0.5)
    full = (lambda th: th) if not rigid_only else (lambda th: np.concatenate([th, np.zeros(nb)]))
    sol = least_squares(lambda th: corr_residual(full(th), pts_m, pix, K, R0, mesh), theta0, loss="huber", f_scale=6.0,
                        bounds=(lo, hi), x_scale=x_scale, max_nfev=80)
    return full(sol.x)


def top_profile(mask: np.ndarray) -> np.ndarray:
    """Row of the topmost mask pixel per column (nan where the column is empty)."""
    has = mask.any(0)
    return np.where(has, mask.argmax(0), np.nan).astype(float)


def silhouette_terms(sil_above: np.ndarray, mask: np.ndarray) -> dict:
    """Above-water IoU, coverage, excess, and the top-edge terms: median |top(sil) - top(mask)| in pixels over the
    columns where both exist, the mean amount by which the mesh's top edge sits above the whale's, and the fraction
    of silhouette columns with no mask at all (the mesh sticks out sideways)."""
    inter = (sil_above & mask).sum(); union = (sil_above | mask).sum()
    ts, tm = top_profile(sil_above), top_profile(mask)
    both = np.isfinite(ts) & np.isfinite(tm)
    top_err = float(np.median(np.abs(ts[both] - tm[both]))) if both.any() else float(sil_above.shape[0])
    top_above = float(np.mean(np.clip(tm[both] - ts[both], 0, None))) if both.any() else 0.0
    only_sil = float(np.mean(np.isfinite(ts) & ~np.isfinite(tm)))
    return {"iou_above_water": float(inter / max(union, 1)), "coverage": float(inter / max(mask.sum(), 1)),
            "excess_above_water": float((sil_above & ~mask).sum() / max(sil_above.sum(), 1)),
            "top_edge_err_px": top_err, "top_edge_above_px": top_above, "columns_beyond_mask": only_sil}


def render_above(theta, R0, mesh: Mesh, K, crop: int, water_row: float, size: int = SIL):
    R, t = pose_from(theta, R0)
    params = params_from(theta[6:])
    V = mesh.V_cm if params.is_identity() else _deform.deform(mesh.V_cm, params)
    return above_water_silhouette(V / 100.0, mesh.F, R, t, K, water_row, crop, size)


def refine_silhouette(theta, pts_m, pix, K, R0, mesh: Mesh, mask_fit: np.ndarray, crop: int, water_row: float) -> np.ndarray:
    """Powell on the correspondence residual plus the top-edge and excess terms, a few iterations from the bend fit;
    mask_fit is the whale mask at FIT_SIL."""
    from scipy.optimize import minimize

    lo, hi = bend_bounds(np.asarray(theta[3:6], float))

    def objective(th):
        r = corr_residual(th, pts_m, pix, K, R0, mesh, prior=False).reshape(-1, 2)
        d = np.linalg.norm(r, axis=1)
        corr = float(np.mean(np.where(d < 6.0, 0.5 * d ** 2, 6.0 * (d - 3.0)))) / 36.0       # Huber, in units of (6 px)^2
        prior = float(np.sum((np.asarray(th[6:]) * DEG_SCALE / PRIOR_DEG) ** 2)) * PRIOR_ROWS / len(pts_m)
        sil_above, _ = render_above(th, R0, mesh, K, crop, water_row, size=FIT_SIL)
        s = silhouette_terms(sil_above, mask_fit)
        top = (s["top_edge_err_px"] + 2.0 * s["top_edge_above_px"]) * (crop / FIT_SIL) / 6.0 + 2.0 * s["columns_beyond_mask"] + (1.0 - s["coverage"])
        return corr + prior + TOP_W * top + EXCESS_W * s["excess_above_water"]

    sol = minimize(objective, np.clip(theta, lo + 1e-6, hi - 1e-6), method="Powell", bounds=list(zip(lo, hi)),
                   options={"maxiter": 3, "xtol": 1e-2, "ftol": 1e-3})
    return np.clip(sol.x, lo, hi)


def _corr(z, seed: int = 0):
    use = z["use"]; signed = z["signed_mm"].astype(np.float32) / 1000.0
    ys, xs = np.nonzero(use)
    pts = signed[ys, xs].astype(np.float64); pix = np.stack([xs + 0.5, ys + 0.5], 1).astype(np.float64)
    if len(pts) > N_CORR:
        pick = np.random.default_rng(seed).choice(len(pts), N_CORR, replace=False)
        pts, pix = pts[pick], pix[pick]
    return pts, pix


def part_mix(pts_m: np.ndarray, mesh: Mesh) -> dict:
    """Share of the correspondences on the flukes / peduncle (body fraction < 0.25), the back (0.25-0.75) and the head."""
    frac = (pts_m[:, 0] * 100.0 - mesh.x_min) / (mesh.x_max - mesh.x_min)
    return {"fluke": float(np.mean(frac < 0.25)), "back": float(np.mean((frac >= 0.25) & (frac < 0.75))), "head": float(np.mean(frac >= 0.75))}


def pose_class(mix: dict) -> str:
    return "fluke" if mix["fluke"] > 0.5 else ("head" if mix["head"] > 0.5 else "back")


def fit_photo(z, record: dict, K: np.ndarray, mesh: Mesh, gt_mask: np.ndarray | None = None, R_gt: np.ndarray | None = None) -> dict:
    """z: the *_canon.npz of a photo (signed_mm, use); record: its points.jsonl row (rotation_direct, translation_m_pnp).
    gt_mask / R_gt (synthetic data): every variant is also scored against the true visible mask and rotation."""
    use = z["use"]; crop = use.shape[0]
    if use.sum() < 50 or not record.get("rotation_direct"):
        return {"ok": False, "reason": "too few whale pixels or no direct rotation"}
    pts, pix = _corr(z)
    gt_small = cv2.resize(gt_mask.astype(np.uint8), (SIL, SIL), interpolation=cv2.INTER_AREA) > 0 if gt_mask is not None else None

    def score(sil, R=None):
        d = silhouette_terms(sil, mask_small)
        if gt_small is not None:
            d["iou_gt"] = float((sil & gt_small).sum() / max((sil | gt_small).sum(), 1))
            d["top_edge_err_gt_px"] = silhouette_terms(sil, gt_small)["top_edge_err_px"]
        if R_gt is not None and R is not None:
            d["rot_err_gt_deg"] = geometry.geodesic_deg(R, R_gt)
        return d
    R0 = np.asarray(record["rotation_direct"], float)
    t0 = init_translation(pts, pix, K, R0)                                    # start independent of the PnP (whose rotation may be wrong)
    if t0 is None:
        t0 = np.asarray(record.get("translation_m_pnp") or [float(K[0, 0]) / max(float(record.get("ppm_direct") or 1.0), 1e-6), 0.0, 0.0], float)
    rows = np.nonzero(use.any(1))[0]
    water_row = float(np.percentile(rows, 99)) + 3.0 * (crop / SIL)
    mask_small = cv2.resize(use.astype(np.uint8), (SIL, SIL), interpolation=cv2.INTER_AREA) > 0
    mask_fit = cv2.resize(use.astype(np.uint8), (FIT_SIL, FIT_SIL), interpolation=cv2.INTER_AREA) > 0
    out = {"ok": True, "whale_pixels": int(use.sum()), "water_row": water_row, "theta": {}, "R": {}, "part_mix": part_mix(pts, mesh)}
    out["pose_class"] = pose_class(out["part_mix"])
    nb = len(BEND_KEYS)
    fit = geometry.fit_pose_pnp_points(pts, pix, K, reproj_px=8.0, max_points=len(pts))
    if fit is not None:                                                     # the outline_metric reference: rigid PnP of the map
        th = np.concatenate([[0, 0, 0], fit["t_m"], np.zeros(nb)])
        sil, _ = render_above(th, fit["R"], mesh, K, crop, water_row)
        out["rigid_pnp"] = score(sil, fit["R"]) | {"rot_vs_direct_deg": geometry.geodesic_deg(fit["R"], R0)}
        out["theta"]["rigid_pnp"] = th.tolist(); out["R"]["rigid_pnp"] = fit["R"].tolist()
    th_r = fit_bend(pts, pix, K, R0, t0, mesh, rigid_only=True)             # rigid, rotation from the direct head
    sil, _ = render_above(th_r, R0, mesh, K, crop, water_row)
    out["rigid_direct"] = score(sil, pose_from(th_r, R0)[0]); out["theta"]["rigid_direct"] = th_r.tolist()
    th_b = fit_bend(pts, pix, K, R0, t0, mesh)                              # bend fit on the correspondences
    sil, _ = render_above(th_b, R0, mesh, K, crop, water_row)
    out["bend"] = score(sil, pose_from(th_b, R0)[0]) | {k: round(float(v) * DEG_SCALE, 1) for k, v in zip(BEND_KEYS, th_b[6:])}
    out["theta"]["bend"] = th_b.tolist()
    th_s = refine_silhouette(th_b, pts, pix, K, R0, mesh, mask_fit, crop, water_row)   # silhouette refinement
    sil, _ = render_above(th_s, R0, mesh, K, crop, water_row)
    R_s, t_s = pose_from(th_s, R0)
    out["bend_silhouette"] = score(sil, R_s) | {k: round(float(v) * DEG_SCALE, 1) for k, v in zip(BEND_KEYS, th_s[6:])}
    out["theta"]["bend_silhouette"] = th_s.tolist()
    out["pose_silhouette"] = {"R": np.round(R_s, 4).tolist(), "t_m": np.round(t_s, 2).tolist(), "rot_vs_direct_deg": geometry.geodesic_deg(R_s, R0)}
    return out


def overlay(image_crop: np.ndarray, use: np.ndarray, sils: dict, label: str) -> np.ndarray:
    """Crop with the mask tinted and the above-water silhouette outlines: rigid PnP orange, bend green, refined white."""
    img = image_crop.copy()
    img[use] = (0.6 * img[use] + 0.4 * np.array([255, 60, 60])).astype(np.uint8)
    H = img.shape[0]
    for name, colour in (("rigid_pnp", (255, 150, 0)), ("bend", (60, 220, 60)), ("bend_silhouette", (255, 255, 255))):
        s = sils.get(name)
        if s is None:
            continue
        big = cv2.resize(s.astype(np.uint8), (H, H), interpolation=cv2.INTER_NEAREST)
        big = cv2.morphologyEx(big, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))        # the face-by-face render leaves hairline gaps
        edge = (cv2.dilate(big, np.ones((3, 3), np.uint8)) - big) > 0
        img[edge] = colour
    cv2.putText(img, label[:95], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


_MESH_CACHE: dict = {}


def _job(args):
    stem, npz_path, record, K, mesh_path, image_path, out_dir, want_overlay = args
    mesh = _MESH_CACHE.setdefault(mesh_path, Mesh(Path(mesh_path)))
    z = np.load(npz_path)
    res = fit_photo(z, record, K, mesh)
    if res.get("ok") and want_overlay:
        x0, y0, s = z["x0y0s"]                       # crop window in original-image pixels (as outline_metric reads it)
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        M = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], np.float64)
        crop = cv2.warpAffine(img, M, (z["use"].shape[1], z["use"].shape[0]), flags=cv2.INTER_AREA)
        R0 = np.asarray(record["rotation_direct"], float)
        sils = {}
        for name in ("rigid_pnp", "bend", "bend_silhouette"):
            th = res["theta"].get(name)
            if th is None:
                continue
            R = np.asarray(res["R"]["rigid_pnp"]) if name == "rigid_pnp" else R0
            sils[name], _ = render_above(np.asarray(th), R, mesh, K, crop.shape[0], res["water_row"], size=crop.shape[0])
        label = (f"{stem[-8:]} IoU pnp {res.get('rigid_pnp', {}).get('iou_above_water', 0):.2f} bend {res['bend']['iou_above_water']:.2f} "
                 f"refined {res['bend_silhouette']['iou_above_water']:.2f}  top err px {res['bend_silhouette']['top_edge_err_px']:.0f}")
        cv2.imwrite(str(Path(out_dir) / "overlays" / f"{stem}.jpg"), cv2.cvtColor(overlay(crop, z["use"], sils, label), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return stem, res


def summarize(rows: dict) -> dict:
    ok = [r for r in rows.values() if r.get("ok")]
    summary = {"photos": len(rows), "scored": len(ok)}
    for variant in ("rigid_pnp", "rigid_direct", "bend", "bend_silhouette"):
        vals = [r[variant] for r in ok if variant in r]
        if not vals:
            continue
        entry = {"n": len(vals)}
        for key in ("iou_above_water", "coverage", "excess_above_water", "top_edge_err_px", "top_edge_above_px", "columns_beyond_mask",
                    "iou_gt", "top_edge_err_gt_px", "rot_err_gt_deg"):
            v = np.array([x[key] for x in vals if key in x], float)
            if len(v):
                entry[key] = {"median": round(float(np.median(v)), 4), "mean": round(float(v.mean()), 4)}
        if any("rot_err_gt_deg" in x for x in vals):
            entry["rot_err_gt_under_15"] = round(float(np.mean([x["rot_err_gt_deg"] < 15 for x in vals if "rot_err_gt_deg" in x])), 4)
        entry["iou_above_0.5"] = round(float(np.mean([x["iou_above_water"] >= 0.5 for x in vals])), 4)
        if variant in ("bend", "bend_silhouette"):
            entry["bend_abs_median_deg"] = {k: round(float(np.median([abs(x[k]) for x in vals])), 1) for k in BEND_KEYS}
        summary[variant] = entry
    return summary


def _synthetic_job(args):
    """One synthetic test crop: the model's prediction packaged like a real photo's, fitted, scored against the truth."""
    sid, category, z, record, K, mesh_path, gt_mask, R_gt, crop_rgb, out_dir = args
    mesh = _MESH_CACHE.setdefault(mesh_path, Mesh(Path(mesh_path)))
    res = fit_photo(z, record, K, mesh, gt_mask=gt_mask, R_gt=R_gt)
    res["category"] = category
    if res.get("ok") and crop_rgb is not None:
        R0 = np.asarray(record["rotation_direct"], float)
        sils = {}
        for name in ("rigid_pnp", "bend", "bend_silhouette"):
            th = res["theta"].get(name)
            if th is None:
                continue
            R = np.asarray(res["R"]["rigid_pnp"]) if name == "rigid_pnp" else R0
            sils[name], _ = render_above(np.asarray(th), R, mesh, K, crop_rgb.shape[0], res["water_row"], size=crop_rgb.shape[0])
        label = (f"synthetic {category} {sid}  IoU(gt) pnp {res.get('rigid_pnp', {}).get('iou_gt', 0):.2f} bend {res['bend']['iou_gt']:.2f} "
                 f"refined {res['bend_silhouette']['iou_gt']:.2f}  rot err {res['bend_silhouette']['rot_err_gt_deg']:.0f} deg")
        img = overlay(crop_rgb, gt_mask, sils, label)
        pred_edge = (cv2.dilate(z["use"].astype(np.uint8), np.ones((3, 3), np.uint8)) - z["use"].astype(np.uint8)) > 0
        img[pred_edge] = (0, 220, 255)                                     # the model's own mask outline, cyan
        cv2.imwrite(str(Path(out_dir) / "overlays" / f"{category}_{sid}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return sid, res


def main_synthetic(a) -> int:
    """--synthetic: the model on crops of the run's test split (oracle crops), fitted like real photos, scored against the
    ground-truth visible mask and rotation; overlays tint the true mask and outline the predicted one in cyan."""
    import torch
    from torch.utils.data import DataLoader
    from ..common.data import IGNORE, CanonCropDataset, WhaleRun
    from .evaluate import predict_crop
    from .model import WhaleCanonModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = WhaleCanonModel.from_checkpoint(a.checkpoint, device)
    run = WhaleRun(a.run_dir, a.split_file)
    table = geometry.load_folded_table(a.table)
    rng = np.random.default_rng(0)
    ids = [i for i in run.ids(a.split) if run.by_id[i]["visible_pixels"] >= 800 and run.has_canon(i)]
    cats = a.categories or sorted({run.by_id[i]["category"] for i in ids})
    per = max(1, a.max_images // len(cats))
    chosen = []
    for c in cats:
        pool = [i for i in ids if run.by_id[i]["category"] == c]
        chosen += [pool[k] for k in rng.permutation(len(pool))[:per]]
    ds = CanonCropDataset(run, chosen, table, crop=512, min_visible=300, train=False)
    loader = DataLoader(ds, batch_size=8, num_workers=4)
    a.out.mkdir(parents=True, exist_ok=True); (a.out / "overlays").mkdir(exist_ok=True)
    jobs = []
    for batch in loader:
        pred = predict_crop(model, batch["image"], device)
        for i, sid in enumerate(batch["id"]):
            meta = run.meta(sid)
            mask = batch["mask"][i].numpy(); x0, y0, s = batch["crop_x0y0s"][i].numpy()
            Kc = geometry.intrinsics_for_crop(geometry.intrinsics_from_meta(meta), x0, y0, s)
            pm = geometry.unfold_norm(np.moveaxis(pred["xyz"][i], 0, -1), table)
            use = pred["vis"][i] & (mask != IGNORE)
            prior, pconf = geometry.choose_prior_side(pred["R"][i], pred["xyz"][i], use)
            res = geometry.resolve_side(pm, use, Kc, R_init=pred["R"][i], table=table, prior_side=prior, prior_confidence=pconf)
            signed = geometry.signed_points(pm, res["side"])
            z = {"use": use, "signed_mm": np.round(np.nan_to_num(signed) * 1000).astype(np.int16), "x0y0s": np.array([x0, y0, s])}
            fit = res.get("fit")
            record = {"rotation_direct": pred["R"][i].tolist(), "translation_m_pnp": fit["t_m"].tolist() if fit else None, "ppm_direct": float(np.exp(pred["log_ppm"][i]))}
            crop_rgb = (batch["image"][i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            jobs.append((sid, meta["category"], z, record, Kc, str(a.mesh), mask == 1, batch["R"][i].numpy(), crop_rgb if len(jobs) < a.overlays else None, str(a.out)))
    rows = {}
    if a.workers > 1:
        from multiprocessing import Pool
        with Pool(a.workers) as pool:
            for sid, res in pool.imap_unordered(_synthetic_job, jobs):
                rows[sid] = res
    else:
        for job in jobs:
            sid, res = _synthetic_job(job); rows[sid] = res
    summary = summarize(rows) | {"checkpoint": str(a.checkpoint), "mesh": str(a.mesh), "split": a.split, "categories": cats}
    by_cat = {}
    for c in cats:
        rs = {k: v for k, v in rows.items() if v.get("category") == c}
        by_cat[c] = {k: v for k, v in summarize(rs).items() if k in ("scored", "rigid_pnp", "bend_silhouette")}
    summary["by_category"] = by_cat
    (a.out / "fit.json").write_text(json.dumps({"summary": summary, "photos": rows}, indent=1))
    for v in ("rigid_pnp", "rigid_direct", "bend", "bend_silhouette"):
        e = summary.get(v)
        if e:
            print(f"{v:16s} n={e['n']:3d} IoU(gt) med {e.get('iou_gt', {}).get('median')}  top-edge err(gt) px {e.get('top_edge_err_gt_px', {}).get('median')}  "
                  f"rot err med {e.get('rot_err_gt_deg', {}).get('median')} deg  <15: {e.get('rot_err_gt_under_15')}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--canon-dir", type=Path, default=None); p.add_argument("--images", type=Path, default=None); p.add_argument("--timestamps", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True); p.add_argument("--mesh", type=Path, default=paths.WHALE_MESH_OBJ)
    p.add_argument("--limit", type=int, default=0); p.add_argument("--overlays", type=int, default=24); p.add_argument("--workers", type=int, default=8)
    p.add_argument("--synthetic", action="store_true", help="fit the model's predictions on synthetic test crops and score them against the truth")
    p.add_argument("--checkpoint", type=Path, default=None); p.add_argument("--run-dir", type=Path, default=paths.WHALE_RUN_DIR)
    p.add_argument("--split-file", type=Path, default=None); p.add_argument("--split", default="test"); p.add_argument("--table", type=Path, default=paths.FOLDED_TABLE)
    p.add_argument("--categories", nargs="*", default=None); p.add_argument("--max-images", type=int, default=48)
    a = p.parse_args(argv)
    if a.synthetic:
        return main_synthetic(a)
    a.out.mkdir(parents=True, exist_ok=True); (a.out / "overlays").mkdir(exist_ok=True)
    ts = json.loads(a.timestamps.read_text())
    records = {Path(r["image"]).stem: r for r in (json.loads(l) for l in open(a.canon_dir / "points.jsonl"))}
    files = sorted(a.canon_dir.glob("*_canon.npz"))
    if a.limit:
        files = files[: a.limit]
    jobs = []
    for i, npz in enumerate(files):
        stem = npz.name[: -len("_canon.npz")]
        rec = records.get(stem)
        if rec is None:
            continue
        z = np.load(npz); x0, y0, s = z["x0y0s"]
        img = cv2.imread(str(a.images / f"{stem}.jpg")); H, W = img.shape[:2]
        meta = ts.get(stem, {}); focal = float(meta["focal_mm"]) if meta.get("focal_mm") else None
        K = geometry.intrinsics_for_crop(geometry.intrinsics_guess(W, H, focal, SENSOR_WIDTH_MM.get(meta.get("camera"), 36.0)), x0, y0, s)
        jobs.append((stem, str(npz), rec, K, str(a.mesh), str(a.images / f"{stem}.jpg"), str(a.out), i < a.overlays))
    rows = {}
    if a.workers > 1:
        from multiprocessing import Pool
        with Pool(a.workers) as pool:
            for stem, res in pool.imap_unordered(_job, jobs):
                rows[stem] = res
                if len(rows) % 25 == 0:
                    print(f"{len(rows)}/{len(jobs)}", flush=True)
    else:
        for job in jobs:
            stem, res = _job(job); rows[stem] = res
    summary = summarize(rows) | {"canon_dir": str(a.canon_dir), "mesh": str(a.mesh)}
    (a.out / "fit.json").write_text(json.dumps({"summary": summary, "photos": rows}, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k in ("photos", "scored", "rigid_pnp", "rigid_direct", "bend", "bend_silhouette")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
