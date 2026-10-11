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
