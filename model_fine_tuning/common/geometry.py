"""Body coordinates <-> 3D, rotations, and the PnP pose fit.

Frames (matching Whale-Image-Data-Factory/factory/project.py):
  * mesh / whale frame: +X tail->snout along the body axis, +Y the whale's right, +Z up; cm.
  * UE camera frame: +X forward (depth), +Y right, +Z up.
  * OpenCV camera frame: +X right, +Y down, +Z forward.  UE_TO_CV converts UE -> CV.
Body coordinates of a surface point: u = fraction along the body (tail 0, snout 1) and
theta = atan2(y, z) (0 = back/top, 90 = right side, 180 = belly, 270 = left side).

The ground-truth pose in meta["pose_in_camera"] stores ``rotation`` with rows = whale axes
in UE camera coordinates, so  X_cam = rotation.T @ X_whale + t.  We call R = rotation.T
the "camera-from-whale" matrix throughout.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

UE_TO_CV = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
# Unreal frames are left-handed, OpenCV's is right-handed, so UE_TO_CV has determinant -1.
# For PnP the *object* frame must be mirrored too (whale y -> -y), giving a proper rotation:
#   X_cv = (UE_TO_CV @ R @ RH_WHALE) @ (RH_WHALE @ X_whale) + UE_TO_CV @ t
RH_WHALE = np.diag([1.0, -1.0, 1.0])
# Mirroring the image horizontally: flips camera Y (UE) and, for a bilaterally symmetric
# whale, is the same as flipping the whale's own Y.  R' = S_CAM @ R @ S_WHALE is a proper rotation.
S_CAM = np.diag([1.0, -1.0, 1.0])
S_WHALE = np.diag([1.0, -1.0, 1.0])


# ---- rotations ------------------------------------------------------------------------------
def gt_camera_from_whale(meta: dict) -> tuple[np.ndarray, np.ndarray]:
    """(R 3x3 camera-from-whale in the UE camera frame, t in metres) from a run meta json."""
    pose = meta["pose_in_camera"]
    R = np.asarray(pose["rotation"], float).T
    t = np.asarray(pose["translation_cm"], float) / 100.0
    return R, t


def geodesic_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    cos = (np.trace(R1.T @ R2) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def flip_rotation(R: np.ndarray) -> np.ndarray:
    return S_CAM @ R @ S_WHALE


def flip_rotation_torch(R):
    """Batched torch version of flip_rotation."""
    import torch

    s_cam = torch.as_tensor(S_CAM, dtype=R.dtype, device=R.device)
    s_whale = torch.as_tensor(S_WHALE, dtype=R.dtype, device=R.device)
    return s_cam @ R @ s_whale


def rotation_to_6d(R: np.ndarray) -> np.ndarray:
    """First two columns, Zhou et al. 2019."""
    return np.concatenate([R[:, 0], R[:, 1]]).astype(np.float64)


def rotation_from_6d_np(d6: np.ndarray) -> np.ndarray:
    a1, a2 = d6[:3], d6[3:6]
    b1 = a1 / max(np.linalg.norm(a1), 1e-8)
    a2 = a2 - b1 * (b1 @ a2)
    b2 = a2 / max(np.linalg.norm(a2), 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def rotation_from_6d_torch(d6):
    import torch
    import torch.nn.functional as F

    a1, a2 = d6[..., :3], d6[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    a2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(a2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


# ---- the body table: (u, theta) -> 3D point on the canonical mesh -----------------------------
def load_obj_vertices(path: Path) -> np.ndarray:
    verts = []
    with open(path) as handle:
        for line in handle:
            if line.startswith("v "):
                verts.append([float(t) for t in line.split()[1:4]])
    return np.asarray(verts, float)


def body_coordinates(vertices_cm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = vertices_cm[:, 0]
    u = (x - x.min()) / max(x.max() - x.min(), 1e-6)
    theta = np.degrees(np.arctan2(vertices_cm[:, 1], vertices_cm[:, 2])) % 360.0
    return u, theta


def build_body_table(vertices_cm: np.ndarray) -> dict:
    """Every mesh vertex with its body coordinates; lookups are nearest-vertex in (u, theta)
    space, which inverts the label rasterisation to within the vertex spacing (~10 cm)."""
    u, theta = body_coordinates(vertices_cm)
    x = vertices_cm[:, 0]
    return {"u": u.astype(np.float32), "theta_deg": theta.astype(np.float32), "points_m": (vertices_cm / 100.0).astype(np.float32),
            "length_m": float((x.max() - x.min()) / 100.0), "x_min_m": float(x.min() / 100.0)}


def _embed(u: np.ndarray, theta_deg: np.ndarray, u_weight: float = 4.0) -> np.ndarray:
    """(u, theta) -> 3D embedding where Euclidean distance respects theta's periodicity. One
    unit of u (the whole body) is weighted like u_weight units of the unit circle."""
    rad = np.radians(np.asarray(theta_deg, float))
    return np.stack([np.asarray(u, float) * u_weight, np.sin(rad), np.cos(rad)], axis=-1)


def save_body_table(table: dict, path: Path) -> None:
    np.savez_compressed(path, **table)


def load_body_table(path: Path) -> dict:
    from scipy.spatial import cKDTree

    with np.load(path) as data:
        table = {k: (data[k].item() if data[k].ndim == 0 else data[k]) for k in data.files}
    table["tree"] = cKDTree(_embed(table["u"], table["theta_deg"]))
    return table


def body_to_3d(table: dict, u: np.ndarray, theta_deg: np.ndarray) -> np.ndarray:
    """(u, theta) arrays -> (N, 3) metres in the mesh frame (nearest mesh vertex)."""
    _, idx = table["tree"].query(_embed(np.ravel(u), np.ravel(theta_deg)), k=1)
    return table["points_m"][idx].reshape(np.shape(u) + (3,))


# ---- cameras ---------------------------------------------------------------------------------
def intrinsics_from_meta(meta: dict) -> np.ndarray:
    k = meta["camera_intrinsics"]
    return np.array([[k["fx_px"], 0.0, k["cx_px"]], [0.0, k["fy_px"], k["cy_px"]], [0.0, 0.0, 1.0]])


def intrinsics_for_crop(K: np.ndarray, x0: float, y0: float, scale: float) -> np.ndarray:
    """Intrinsics after cropping at (x0, y0) in the original image and resizing by ``scale``."""
    K2 = K.copy()
    K2[0, 0] *= scale
    K2[1, 1] *= scale
    K2[0, 2] = (K[0, 2] - x0) * scale
    K2[1, 2] = (K[1, 2] - y0) * scale
    return K2


def intrinsics_guess(width: int, height: int, focal_mm: float | None = None, sensor_width_mm: float = 36.0, hfov_deg: float | None = None) -> np.ndarray:
    """For photos without calibration: from EXIF focal + sensor width, or a horizontal FOV."""
    if focal_mm is not None:
        fx = focal_mm / sensor_width_mm * width
    elif hfov_deg is not None:
        fx = width / (2.0 * np.tan(np.radians(hfov_deg) / 2.0))
    else:
        fx = 1.2 * width
    return np.array([[fx, 0.0, width / 2.0], [0.0, fx, height / 2.0], [0.0, 0.0, 1.0]])


def project_ue(X_whale_m: np.ndarray, R: np.ndarray, t_m: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Whale-frame points -> pixels, with a camera-from-whale pose in the UE camera frame."""
    cam = X_whale_m @ R.T + t_m
    depth = cam[:, 0]
    safe = np.where(depth > 1e-6, depth, 1e-6)
    u = K[0, 2] + K[0, 0] * cam[:, 1] / safe
    v = K[1, 2] - K[1, 1] * cam[:, 2] / safe
    return np.stack([u, v], 1), depth


# ---- PnP from dense body coordinates --------------------------------------------------------
def fit_pose_pnp(u: np.ndarray, theta_deg: np.ndarray, valid: np.ndarray, K: np.ndarray, table: dict,
                 max_points: int = 4000, reproj_px: float = 6.0, iterations: int = 300, seed: int = 0) -> dict | None:
    """Recover the whale's rigid pose from per-pixel body coordinates.

    u, theta_deg: (H, W) predicted coordinates; valid: (H, W) bool whale pixels to use;
    K: intrinsics of that image. Returns R (camera-from-whale, UE camera frame), t (m),
    pixels-per-metre at the whale origin, the inlier count, and the inlier fraction."""
    import cv2

    ys, xs = np.nonzero(valid)
    if len(xs) < 12:
        return None
    rng = np.random.default_rng(seed)
    if len(xs) > max_points:
        pick = rng.choice(len(xs), max_points, replace=False)
        ys, xs = ys[pick], xs[pick]
    obj = (body_to_3d(table, u[ys, xs], theta_deg[ys, xs]) @ RH_WHALE).astype(np.float64)
    img = np.stack([xs, ys], 1).astype(np.float64) + 0.5
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(obj, img, K.astype(np.float64), None, iterationsCount=iterations,
                                                 reprojectionError=reproj_px, confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers) < 6:
        return None
    inl = inliers[:, 0]
    ok, rvec, tvec = cv2.solvePnP(obj[inl], img[inl], K.astype(np.float64), None, rvec=rvec, tvec=tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R_cv, _ = cv2.Rodrigues(rvec)
    t_cv = tvec[:, 0]
    R = UE_TO_CV.T @ R_cv @ RH_WHALE
    t = UE_TO_CV.T @ t_cv
    if t[0] <= 0:
        return None
    return {"R": R, "t_m": t, "ppm": float(K[0, 0] / t[0]), "inliers": int(len(inl)), "inlier_fraction": float(len(inl) / len(xs)),
            "n_points": int(len(xs))}


def load_landmarks(path: Path) -> dict[str, np.ndarray]:
    data = json.loads(Path(path).read_text())
    return {k: np.asarray(v, float) / 100.0 for k, v in data.items()}


# ---- canonical surface points (folded) ------------------------------------------------------
def fold_points_m(points_m: np.ndarray) -> np.ndarray:
    """(N, 3) mesh-frame metres -> folded (x, |y|, z)."""
    out = np.asarray(points_m, float).copy()
    out[:, 1] = np.abs(out[:, 1])
    return out


def load_obj_faces(path: Path) -> np.ndarray:
    """Triangle indices (M, 3) of an OBJ (polygons fan-triangulated)."""
    faces = []
    for line in open(path):
        if line.startswith("f "):
            idx = [int(c.split("/")[0]) - 1 for c in line.split()[1:]]
            for k in range(1, len(idx) - 1):
                faces.append((idx[0], idx[k], idx[k + 1]))
    return np.asarray(faces, np.int64)


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted outward vertex normals (unit), assuming consistently wound faces; the
    overall orientation is fixed so that normals point away from the body axis on average."""
    v = np.asarray(vertices, float)
    n = np.zeros_like(v)
    a, b, c = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    fn = np.cross(b - a, c - a)
    for k in range(3):
        np.add.at(n, faces[:, k], fn)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(norm, 1e-9)
    radial = v - v.mean(0)
    radial[:, 0] = 0.0
    if np.mean(np.sum(n * radial, axis=1)) < 0:
        n = -n
    return n


def build_folded_table(vertices_cm: np.ndarray, faces: np.ndarray | None = None) -> dict:
    """Mesh vertices as folded canonical points (metres) for nearest-surface lookups, with the
    normalisation the dense head uses: x_hat = (x - x_min) / L, y_hat = |y| / L, z_hat = z / L.
    With faces, also the outward vertex normals folded to the +y half ("normals_folded": for a
    vertex at y < 0 the mirrored normal), so a candidate side can be checked for facing the
    camera (resolve_side)."""
    pts = np.asarray(vertices_cm, float) / 100.0
    x = pts[:, 0]
    table = {"folded_m": fold_points_m(pts).astype(np.float32), "length_m": float(x.max() - x.min()), "x_min_m": float(x.min())}
    if faces is not None:
        n = vertex_normals(pts, faces)
        n[:, 1] *= np.where(pts[:, 1] < 0, -1.0, 1.0)
        table["normals_folded"] = n.astype(np.float32)
    return table


def load_folded_table(path: Path) -> dict:
    from scipy.spatial import cKDTree

    with np.load(path) as data:
        table = {k: (data[k].item() if data[k].ndim == 0 else data[k]) for k in data.files}
    table["tree"] = cKDTree(table["folded_m"])
    return table


def unfold_norm(xyz_hat: np.ndarray, table: dict) -> np.ndarray:
    """(..., 3) normalised folded prediction -> folded metres (x, |y|, z)."""
    L, x0 = table["length_m"], table["x_min_m"]
    out = np.asarray(xyz_hat, float) * L
    out[..., 0] += x0
    return out


def snap_to_surface(folded_m: np.ndarray, table: dict) -> tuple[np.ndarray, np.ndarray]:
    """Nearest mesh vertex (folded) for each folded point: (N, 3) snapped points and distances (m)."""
    d, idx = table["tree"].query(np.asarray(folded_m, float).reshape(-1, 3), k=1)
    return table["folded_m"][idx].reshape(np.shape(folded_m)), d.reshape(np.shape(folded_m)[:-1])


def signed_points(folded_m: np.ndarray, side: np.ndarray) -> np.ndarray:
    """Folded (x, |y|, z) + side (+1 / -1) -> mesh-frame points (x, side * |y|, z)."""
    out = np.asarray(folded_m, float).copy()
    out[..., 1] = out[..., 1] * np.where(np.asarray(side) < 0, -1.0, 1.0)
    return out


def fit_pose_pnp_points(points_m: np.ndarray, pixels: np.ndarray, K: np.ndarray, reproj_px: float = 6.0, iterations: int = 300,
                        max_points: int = 4000, seed: int = 0) -> dict | None:
    """RANSAC PnP from mesh-frame points (N, 3, metres) to pixels (N, 2). Same conventions and
    output as fit_pose_pnp."""
    import cv2

    pts = np.asarray(points_m, np.float64).reshape(-1, 3)
    img = np.asarray(pixels, np.float64).reshape(-1, 2)
    if len(pts) < 12:
        return None
    if len(pts) > max_points:
        pick = np.random.default_rng(seed).choice(len(pts), max_points, replace=False)
        pts, img = pts[pick], img[pick]
    obj = pts @ RH_WHALE
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(obj, img, K.astype(np.float64), None, iterationsCount=iterations,
                                                 reprojectionError=reproj_px, confidence=0.999, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers) < 6:
        return None
    inl = inliers[:, 0]
    ok, rvec, tvec = cv2.solvePnP(obj[inl], img[inl], K.astype(np.float64), None, rvec=rvec, tvec=tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    R_cv, _ = cv2.Rodrigues(rvec)
    t_cv = tvec[:, 0]
    R = UE_TO_CV.T @ R_cv @ RH_WHALE
    t = UE_TO_CV.T @ t_cv
    if t[0] <= 0:
        return None
    return {"R": R, "t_m": t, "ppm": float(K[0, 0] / t[0]), "inliers": int(len(inl)), "inlier_fraction": float(len(inl) / len(pts)),
            "n_points": int(len(pts))}


def global_side_cue(xyz_hat: np.ndarray, valid: np.ndarray) -> tuple[int, float]:
    """Which flank faces the camera, read off the dense folded map alone: the image direction
    towards the head (least-squares gradient of x_hat) crossed with the direction towards the
    back (gradient of z_hat). For a bilaterally symmetric body seen in profile the sign of that
    cross product fixes the visible flank (constant calibrated on synthetic ground truth).
    Returns (+1 right / -1 left / 0 unknown, confidence = |sin| of the angle between the two
    directions; near 0 when the view is along the body or straight down, where both flanks may
    be visible and the cue says nothing)."""
    x = np.asarray(xyz_hat)
    if x.shape[0] != 3:
        x = np.moveaxis(x, -1, 0)
    ys, xs = np.nonzero(valid)
    if len(xs) < 50:
        return 0, 0.0
    A = np.stack([xs, ys, np.ones_like(xs)], 1).astype(float)
    cx = np.linalg.lstsq(A, x[0][ys, xs], rcond=None)[0]
    cz = np.linalg.lstsq(A, x[2][ys, xs], rcond=None)[0]
    h, d = cx[:2], cz[:2]
    nh, nd = np.linalg.norm(h), np.linalg.norm(d)
    if nh < 1e-9 or nd < 1e-9:
        return 0, 0.0
    cross = (h[0] * d[1] - h[1] * d[0]) / (nh * nd)
    return (-1 if cross > 0 else 1), float(abs(cross))


def choose_prior_side(R_direct: np.ndarray | None, xyz_hat: np.ndarray, valid: np.ndarray, cue_threshold: float = 0.85) -> tuple[int, float]:
    """Global facing-flank prior for resolve_side: the dense-map cue when it is confident,
    otherwise the direct rotation head. Returns (side, confidence)."""
    cue, conf = global_side_cue(xyz_hat, valid)
    if cue != 0 and conf >= cue_threshold:
        return cue, conf
    if R_direct is not None:
        return initial_side_from_rotation(R_direct), 0.5
    return (cue if cue != 0 else 1), conf


def initial_side_from_rotation(R: np.ndarray) -> int:
    """Which flank faces the camera according to a camera-from-whale rotation: the camera's
    viewing axis (UE: +x) expressed in the whale frame has a y component; looking toward -y
    means the camera sits on the whale's right (+y) side."""
    view_in_whale = R.T @ np.array([1.0, 0.0, 0.0])
    return -1 if view_in_whale[1] > 0 else 1


def _side_hypothesis(fm, pix, K, init_sign: int, normals, iterations, reproj_px, max_points, seed, facing_penalty_px, err_cap_px=60.0):
    """One run of the alternating side / pose fit from a global initial side. Returns the side
    per point, the fit, and the final mean cost (capped reprojection residual + facing penalty)."""
    n = len(pix)
    side = np.full(n, init_sign, np.int8)
    fit, cost = None, np.inf
    for _ in range(iterations):
        fit = fit_pose_pnp_points(signed_points(fm, side), pix, K, reproj_px=reproj_px, max_points=max_points, seed=seed)
        if fit is None:
            return side, None, np.inf
        err = []
        for sgn in (1, -1):
            pts_s = signed_points(fm, np.full(n, sgn))
            uv, depth = project_ue(pts_s, fit["R"], fit["t_m"], K)
            e = np.linalg.norm(uv - pix, axis=1)
            e[depth <= 0] = np.inf
            if normals is not None:
                n_s = normals * np.array([1.0, sgn, 1.0])
                cam_pts = pts_s @ fit["R"].T + fit["t_m"]
                ray = cam_pts / np.maximum(np.linalg.norm(cam_pts, axis=1, keepdims=True), 1e-9)
                facing_away = np.sum((n_s @ fit["R"].T) * ray, axis=1)       # > 0: the surface would face away from the camera
                e = e + facing_penalty_px * np.clip(facing_away, 0, 1)
            err.append(e)
        new_side = np.where(err[0] <= err[1], 1, -1).astype(np.int8)
        chosen = np.where(new_side == 1, err[0], err[1])
        cost = float(np.mean(np.minimum(chosen, err_cap_px)))
        changed = float(np.mean(new_side != side))
        side = new_side
        if changed < 0.005:
            break
    return side, fit, cost


def resolve_side(folded_m: np.ndarray, valid: np.ndarray, K: np.ndarray, R_init: np.ndarray | None = None,
                 iterations: int = 3, reproj_px: float = 8.0, max_points: int = 3000, seed: int = 0,
                 table: dict | None = None, facing_penalty_px: float = 200.0,
                 prior_side: int | None = None, prior_confidence: float = 0.0, prior_cost_px: float = 10.0) -> dict:
    """Decide, once per whale, which side of the midline each pixel's folded point lies on.

    The dense head cannot tell left from right (the body is symmetric), so the side is a latent
    of the rigid pose. Both global hypotheses (the flank the direct rotation head says faces the
    camera, and the opposite one) are run through an alternating fit: RANSAC pose from the
    signed points, then each pixel takes the sign whose reprojection lands nearer its position
    and whose surface faces the camera (table normals), then refit. The hypothesis with the
    lower final cost wins; the relative margin between the two is returned as a confidence.
    Returns the (H, W) side map (+1 / -1, 0 where not valid), the pose, purity and margin."""
    H, W = valid.shape
    ys, xs = np.nonzero(valid)
    if len(xs) < 12:
        return {"side": np.zeros((H, W), np.int8), "fit": None, "purity": None, "margin": None, "fraction_right": None, "agrees_with_prior": None}
    pix = np.stack([xs, ys], 1).astype(np.float64) + 0.5
    fm = np.asarray(folded_m)[ys, xs]
    normals = None
    if table is not None and "normals_folded" in table:
        _, idx = table["tree"].query(fm, k=1)
        normals = table["normals_folded"][idx]
    if prior_side:
        first = int(prior_side)
    else:
        first = initial_side_from_rotation(R_init) if R_init is not None else 1
    runs = [_side_hypothesis(fm, pix, K, sgn, normals, iterations, reproj_px, max_points, seed, facing_penalty_px) for sgn in (first, -first)]
    # the second hypothesis has to beat the prior by a margin that grows with the prior's confidence
    costs = [runs[0][2], runs[1][2] + prior_cost_px * float(prior_confidence)]
    best = int(np.argmin(costs))
    side, fit, cost = runs[best]
    other = costs[1 - best]
    margin = None if not np.isfinite(cost) else (1.0 if not np.isfinite(other) else float((other - cost) / max(other, 1e-6)))
    out = np.zeros((H, W), np.int8)
    out[ys, xs] = side
    frac_right = float(np.mean(side == 1))
    return {"side": out, "fit": fit, "purity": max(frac_right, 1 - frac_right), "fraction_right": frac_right, "margin": margin,
            "cost_px": cost, "cost_other_px": other, "agrees_with_prior": bool(best == 0), "prior_side": first}


def landmarks_from_map(points_m: np.ndarray, valid: np.ndarray, landmarks: dict[str, np.ndarray], max_dist_m: float = 0.35) -> dict:
    """Points of interest read off the dense map: for each landmark (mesh-frame metres, signed),
    the valid pixel whose predicted signed point is nearest, if within max_dist_m. Returns
    name -> {"uv": (x, y), "dist_m": d} (uv None when no pixel is close enough)."""
    ys, xs = np.nonzero(valid)
    out = {}
    if len(xs) == 0:
        return {n: {"uv": None, "dist_m": None} for n in landmarks}
    pts = np.asarray(points_m)[ys, xs]
    for name, lm in landmarks.items():
        d = np.linalg.norm(pts - np.asarray(lm, float)[None, :], axis=1)
        i = int(np.argmin(d))
        out[name] = {"uv": (float(xs[i]) + 0.5, float(ys[i]) + 0.5), "dist_m": float(d[i])} if d[i] <= max_dist_m else {"uv": None, "dist_m": float(d[i])}
    return out


def mesh_region(folded_m: np.ndarray, table: dict) -> np.ndarray:
    """Coarse region id for error breakdowns: 0 torso, 1 flukes, 2 pectoral fins, from the
    canonical position (body fraction and lateral offset)."""
    f = np.asarray(folded_m)
    u = (f[..., 0] - table["x_min_m"]) / table["length_m"]
    lateral = f[..., 1]
    region = np.zeros(f.shape[:-1], np.int8)
    region[(u < 0.12) & (lateral > 0.6)] = 1
    region[(u > 0.5) & (u < 0.75) & (lateral > 1.2) & (f[..., 2] < 0)] = 2
    return region

