"""The researcher's clicks and paired regions on the two click days, in 2560-jpg pixel coordinates.

clicks(day)  -> {key: {name: (x, y)}}          (fluke_tip_left / right, fluke_notch, dorsal_fin_front / back / tip, ...)
pairs(day)   -> [dict(key_a, key_b, regions=[(ca, cb, size_a, size_b)], same_burst, cue)]  centres in jpg pixels
cutout_M(day, key, part) -> 2x3 matrix from jpg pixels to the cut-out v3 square (part 'finB' or 'fluke'), or None
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
from PIL import Image

from . import CUT, DAYS, LAB, guard


@lru_cache(maxsize=None)
def img_size(day: str, stem: str):
    return Image.open(DAYS / day / "images" / f"{stem}.jpg").size


def _labels(day):
    guard([day])
    z = json.loads((LAB / day / "pair_marks.json").read_text())
    return z["labels"] if isinstance(z, dict) else z


def clicks(day):
    out = {}
    for l in _labels(day):
        if l.get("verdict") == "skip":
            continue
        for sd in ("a", "b"):
            kp = (l.get("keypoints") or {}).get(sd) or {}
            key = l.get(f"key_{sd}")
            if kp and key:
                W, H = img_size(day, key.rsplit("~", 1)[0])
                out.setdefault(key, {}).update({k: (v["x"] * W, v["y"] * H) for k, v in kp.items()})
    return out


def pairs(day):
    out = []
    for l in _labels(day):
        if not l.get("regions") or l.get("verdict") == "skip":
            continue
        ka, kb = l["key_a"], l["key_b"]
        Wa, Ha = img_size(day, ka.rsplit("~", 1)[0]); Wb, Hb = img_size(day, kb.rsplit("~", 1)[0])
        regs = []
        for r in l["regions"]:
            a, b = r["a"], r["b"]
            ca = np.array([(a["x"] + a["w"] / 2) * Wa, (a["y"] + a["h"] / 2) * Ha])
            cb = np.array([(b["x"] + b["w"] / 2) * Wb, (b["y"] + b["h"] / 2) * Hb])
            regs.append((ca, cb, (a["w"] * Wa, a["h"] * Ha), (b["w"] * Wb, b["h"] * Hb)))
        out.append(dict(key_a=ka, key_b=kb, regions=regs, same_burst=bool(l.get("same_burst")), cue=l.get("cue") or [],
                        verdict=l.get("verdict"), catalog=l.get("catalog")))
    return out


@lru_cache(maxsize=None)
def _scale_full(day, key):
    for part in ("body", "fin", "fluke"):
        f = DAYS / day / "parts_up" / f"{key}__{part}.npz"
        if f.exists():
            return float(np.load(f)["scale_full"])
    return None


def cutout_M(day, key, part):
    """jpg pixel -> cut-out pixel (the cut-out code works in a frame scaled by parts_up's scale_full)."""
    f = CUT / day / part / f"{key}.npz"
    sf = _scale_full(day, key)
    if not f.exists() or sf is None:
        return None
    M = np.load(f)["M"].astype(float)
    return np.hstack([M[:, :2] * sf, M[:, 2:]])


def apply(M, p):
    p = np.asarray(p, float)
    if M.shape[0] == 3:
        q = M @ np.array([p[0], p[1], 1.0]); return q[:2] / q[2]
    return M[:, :2] @ p + M[:, 2]


def inv(M):
    if M.shape[0] == 3:
        return np.linalg.inv(M)
    A = np.linalg.inv(M[:, :2]); return np.hstack([A, -(A @ M[:, 2:])])
