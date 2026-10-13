"""Reader for the run03 factory labels (Whale-Image-Data-Factory, config run03_bodycoords):
keypoints with visibility states, deformation parameters, body scale, the canonical and the
deformed per-pixel maps, the water row. Everything a body-coordinate / pose model needs beyond
the run01/run02 labels, in one place.

    from model_fine_tuning.common.run03_labels import KEYPOINTS, STATES, load_shot
    shot = load_shot(run_dir, "000018")      # dict of numpy arrays and scalars

Keypoint order is fixed (KEYPOINTS) so heads can index by position; a keypoint missing from a
shot's meta (e.g. eyes when the mesh has no eye group) gets state "missing".
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

KEYPOINTS = ["snout", "chin", "blowhole", "eye_left", "eye_right", "dorsal_fin_tip", "dorsal_fin_front", "dorsal_fin_back",
             "tail_base_top", "tail_base_bottom", "fluke_notch", "fluke_tip_left", "fluke_tip_right",
             "pectoral_tip_left", "pectoral_tip_right", "pectoral_base_left", "pectoral_base_right", "belly"]
STATES = ["visible", "hidden", "under_water", "occluded", "out_of_frame", "missing"]
STATE_ID = {s: i for i, s in enumerate(STATES)}
# left/right pairs for horizontal-flip augmentation (swap labels when the image is mirrored)
FLIP_PAIRS = [("eye_left", "eye_right"), ("fluke_tip_left", "fluke_tip_right"), ("pectoral_tip_left", "pectoral_tip_right"), ("pectoral_base_left", "pectoral_base_right")]
DEFORM_PARAMS = ["arch_deg", "tail_curl_deg", "head_pitch_deg", "lateral_deg", "fluke_twist_deg", "fin_raise_deg", "fin_sweep_deg"]
CANON_OFFSET = 32768


def read_canon(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(H, W, 3) metres (whale / canonical frame) and the defined mask, from a factory canon npz."""
    z = np.load(path)
    values = z["canon_mm"] if "canon_mm" in z.files else z[z.files[0]]
    defined = z["defined"] if "defined" in z.files else (values != 0).any(-1)
    cm = (values.astype(np.int32) - CANON_OFFSET) / 1000.0
    return cm.astype(np.float32), defined.astype(bool)


def keypoint_arrays(meta: dict) -> dict:
    """Fixed-order arrays: uv (K, 2) px, state ids (K,), whale-frame xyz (K, 3) m, canonical xyz (K, 3) m, height above water (K,) m."""
    kps = meta.get("keypoints", {})
    K = len(KEYPOINTS)
    uv = np.full((K, 2), np.nan, np.float32); st = np.full(K, STATE_ID["missing"], np.int64)
    xyz = np.full((K, 3), np.nan, np.float32); can = np.full((K, 3), np.nan, np.float32); h = np.full(K, np.nan, np.float32)
    for i, name in enumerate(KEYPOINTS):
        k = kps.get(name)
        if not k:
            continue
        uv[i] = k["uv"]; st[i] = STATE_ID.get(k.get("state", "missing"), STATE_ID["missing"])
        if "xyz_whale_cm" in k:
            xyz[i] = np.asarray(k["xyz_whale_cm"], np.float32) / 100.0
        if "xyz_canonical_cm" in k:
            can[i] = np.asarray(k["xyz_canonical_cm"], np.float32) / 100.0
        if "height_above_water_m" in k:
            h[i] = k["height_above_water_m"]
    return {"uv": uv, "state": st, "xyz_whale_m": xyz, "xyz_canonical_m": can, "height_m": h}


def load_shot(run_dir: Path, shot_id: str, maps: bool = True) -> dict:
    run_dir = Path(run_dir); ds = run_dir / "dataset"
    meta = json.loads((ds / "meta" / f"{shot_id}.json").read_text())
    out = {"id": shot_id, "meta": meta, "image": ds / meta["paths"]["image"], "mask": ds / meta["paths"]["mask"],
           "deform": np.array([float(meta.get("deform", {}).get(k, 0.0)) for k in DEFORM_PARAMS], np.float32),
           "body_scale": float(meta.get("body_scale", 1.0)), "water_row_px": meta.get("water_row_px"),
           "rotation": np.asarray(meta["pose_in_camera"]["rotation"], np.float32), "translation_m": np.asarray(meta["pose_in_camera"]["translation_cm"], np.float32) / 100.0,
           "intrinsics": meta["camera_intrinsics"], "category": meta["category"], "negative": bool(meta["negative"])}
    out.update(keypoint_arrays(meta))
    if maps:
        out["canon_m"], out["canon_defined"] = read_canon(ds / meta["paths"]["canon"])
        if "canon_deformed" in meta["paths"]:
            out["canon_deformed_m"], _ = read_canon(ds / meta["paths"]["canon_deformed"])
    return out


def flip_keypoints(uv: np.ndarray, state: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Labels for the horizontally mirrored image: x -> width - x, left/right names swapped."""
    uv2 = uv.copy(); uv2[:, 0] = width - uv2[:, 0]; st2 = state.copy()
    for a, b in FLIP_PAIRS:
        i, j = KEYPOINTS.index(a), KEYPOINTS.index(b)
        uv2[[i, j]] = uv2[[j, i]]; st2[[i, j]] = st2[[j, i]]
    return uv2, st2
