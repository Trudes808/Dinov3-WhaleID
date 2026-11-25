"""Shared helpers of the fused-instance study (two whales in one "<stem>~k" instance). Read-only on the shadow root
(moonshot_cache/switch_root/days); candidates are built in moonshot_cache/fused_instances/. Evaluation days are refused
unless FUSED_EVAL_OK=1 (set only for the one frozen read, step 4 of README.md).

Frames: "crop" = the 512-px CSE crop of DAY/cse/<stem>_canon.npz (x0y0s maps crop px -> 2560-px photo frame:
X = cx / s + x0); SAM 3 masks are 288 x 288 over the whole photo (squashed).
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
NW = REPO.parent
SR = NW / "moonshot_cache" / "switch_root" / "days"          # shadow root (S6h + rule F), read-only
FI = NW / "moonshot_cache" / "fused_instances"              # this study's untracked folder
S3RAW_OLD = NW / "moonshot_cache" / "sam3_compare" / "raw"   # cached SAM 3 runs (val: all photos, click: clicked photos)
S3RAW = FI / "sam3" / "raw"                                  # this study's SAM 3 runs ("humpback whale" only)
LABELS = REPO / "model_fine_tuning" / "splits" / "labels_all_days.json"
SPLIT = json.loads((REPO / "model_fine_tuning" / "splits" / "days.json").read_text())
EVAL = set(SPLIT["eval"])
VAL = list(SPLIT["val"])
TRAIN = list(SPLIT["train"])
CLICK = ["2025-01-02", "2025-03-18"]
TRAIN3 = ["2025-02-03", "2025-01-23", "2025-03-26"]
CAND_DAYS = VAL + CLICK + TRAIN3
SAM_THR = 0.5                                                # settings_partB.json class_thr whale


def check_day(day: str) -> None:
    if day in EVAL and os.environ.get("FUSED_EVAL_OK") != "1":
        raise SystemExit(f"evaluation day {day} refused")


@lru_cache(maxsize=1)
def table():
    from model_fine_tuning.common import geometry, paths
    return geometry.load_folded_table(paths.FOLDED_TABLE)


def stems(day_dir: Path) -> list[str]:
    return [Path(json.loads(l)["image"]).stem for l in open(day_dir / "images" / "index.jsonl")]


def cse(day_dir: Path, stem: str):
    f = day_dir / "cse" / f"{stem}_canon.npz"
    if not f.exists():
        return None
    z = np.load(f)
    use = z["use"]; signed = z["signed_mm"].astype(np.float32) / 1000.0
    T = table(); u = (signed[..., 0] - T["x_min_m"]) / T["length_m"]
    return {"use": use, "signed": signed, "u": u, "side": z["side"], "x0y0s": tuple(float(v) for v in z["x0y0s"])}


def components(use: np.ndarray, min_frac: float = 0.1, min_px: int = 400):
    """The components split_instances considers (largest first), as boolean masks."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(use.astype(np.uint8), connectivity=8)
    if n <= 1:
        return []
    areas = st[1:, cv2.CC_STAT_AREA]; order = np.argsort(-areas)
    return [lab == (k + 1) for k in order if areas[k] >= max(min_px, min_frac * areas[order[0]])]


def fin_region(use, u, signed):
    return use & (u > 0.36) & (u < 0.54) & (signed[..., 2] > 0.9)


def fluke_region(use, u, signed):
    return use & (u < 0.12) & (np.abs(signed[..., 1]) > 0.5)


def to_frame(x0y0s, cx, cy):
    x0, y0, s = x0y0s
    return cx / s + x0, cy / s + y0


# ------------------------------------------------------------------------------------------------ SAM 3
def sam_record(day: str, stem: str):
    """(scores, masks 288 x 288 uint8 prob, hw) of the "humpback whale" instances, or None when SAM 3 did not run."""
    for root, setname in ((S3RAW, day), (S3RAW_OLD, "val"), (S3RAW_OLD, "click")):
        f = root / setname / f"{stem}.npz"
        if f.exists():
            z = np.load(f)
            pr = json.loads((root / setname / "_prompts.json").read_text())
            j = pr.index("humpback whale")
            return z[f"s{j}"], z[f"m{j}"], tuple(int(v) for v in z["hw"])
    return None


def sam_in_crop(day: str, stem: str, x0y0s, shape, thr: float = SAM_THR):
    """SAM 3 whale instances (score >= thr), de-overlapped by score, sampled on the crop grid: list of (score, bool mask)."""
    rec = sam_record(day, stem)
    if rec is None:
        return None
    sc, ms, (H, W) = rec
    keep = np.nonzero(sc >= thr)[0]
    if not len(keep):
        return []
    h, w = shape
    cy, cx = np.mgrid[0:h, 0:w].astype(np.float32)
    X, Y = to_frame(x0y0s, cx + 0.5, cy + 0.5)
    gx = np.clip((X * ms.shape[2] / W).astype(int), 0, ms.shape[2] - 1); gy = np.clip((Y * ms.shape[1] / H).astype(int), 0, ms.shape[1] - 1)
    order = keep[np.argsort(-sc[keep])]
    taken = np.zeros(shape, bool); out = []
    for j in order:
        m = (ms[j][gy, gx] >= 128) & ~taken
        taken |= ms[j][gy, gx] >= 128
        if m.any():
            out.append((float(sc[j]), m))
    return out


def sam_full(day: str, stem: str, hw, thr: float = SAM_THR):
    """SAM 3 whale instances at the photo frame hw (H, W), de-overlapped: list of (score, bool mask)."""
    rec = sam_record(day, stem)
    if rec is None:
        return None
    sc, ms, _ = rec
    keep = np.nonzero(sc >= thr)[0]
    order = keep[np.argsort(-sc[keep])]
    taken = np.zeros(hw, bool); out = []
    for j in order:
        m0 = cv2.resize(ms[j], (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR) >= 128
        m = m0 & ~taken; taken |= m0
        if m.any():
            out.append((float(sc[j]), m))
    return out


# ------------------------------------------------------------------------------------------------ the signals
GEOM = {"min_side_px": 400}


def instance_geometry(inst: np.ndarray, u: np.ndarray, signed: np.ndarray, x0y0s) -> dict:
    """Fin and fluke regions of one instance (as parts.py defines them) and where they lie relative to each other.
    along / perp: the fluke centroid relative to the fin-region centroid in the frame of the back's u gradient (a linear
    fit of u over the instance's back pixels, u > 0.2): along > 0 = towards the head; both in units of the distance one
    whale's fluke should lie from its fin by that gradient, (u_fin - u_fluke) / |grad u|."""
    s = x0y0s[2]
    fin = fin_region(inst, u, signed); flk = fluke_region(inst, u, signed)
    out = {"fin_px": int(fin.sum()), "fluke_px": int(flk.sum()), "area": int(inst.sum())}
    comps = components(inst, 0.0, 50)
    out["n_comp"] = len(comps)
    if out["fin_px"] < 50 or out["fluke_px"] < 50:
        return out
    fy, fx = np.nonzero(fin); ky, kx = np.nonzero(flk)
    cf = np.array([fx.mean(), fy.mean()]); ck = np.array([kx.mean(), ky.mean()])
    out["dist_px2560"] = round(float(np.hypot(*(ck - cf)) / s), 1)
    lab = np.zeros(inst.shape, np.int32)
    for i, c in enumerate(comps):
        lab[c] = i + 1
    out["fin_comp"] = int(np.bincount(lab[fin], minlength=len(comps) + 1)[1:].argmax()) if comps else -1
    out["fluke_comp"] = int(np.bincount(lab[flk], minlength=len(comps) + 1)[1:].argmax()) if comps else -1
    out["same_comp"] = out["fin_comp"] == out["fluke_comp"]
    back = inst & (u > 0.2)
    by, bx = np.nonzero(back)
    if len(by) >= 200:
        A = np.stack([bx, by, np.ones_like(bx)], 1).astype(np.float64)
        uu = u[by, bx].astype(np.float64)
        coef = np.linalg.lstsq(A, uu, rcond=None)[0]; g = coef[:2]
        gn = float(np.hypot(*g))
        if gn > 1e-6:
            e = g / gn
            reach = (float(u[fy, fx].mean()) - float(u[ky, kx].mean())) / gn            # px (crop) one whale's fluke lies behind its fin
            d = ck - cf
            out["reach_px2560"] = round(reach / s, 1)
            out["along"] = round(float(d @ e) / max(reach, 1e-6), 3)                     # one whale: about -1
            out["perp"] = round(float(abs(d[0] * e[1] - d[1] * e[0])) / max(reach, 1e-6), 3)
            out["fit_r2"] = round(float(1 - np.var(uu - A @ coef) / max(np.var(uu), 1e-9)), 3)
    return out


def sam_split(inst: np.ndarray, sam: list, min_share: float = 0.15, min_px: int = 400):
    """The SAM 3 whale instances that each hold >= min_share of our instance's pixels (and >= min_px crop px)."""
    if not sam:
        return []
    a = inst.sum()
    out = []
    for sc, m in sam:
        o = int((inst & m).sum())
        if o >= min_px and o >= min_share * a:
            out.append((sc, m, o / a))
    return out


def catalogue(day: str) -> dict:
    check_day(day)
    return json.loads(LABELS.read_text())["days"].get(day, {}).get("photos", {})
