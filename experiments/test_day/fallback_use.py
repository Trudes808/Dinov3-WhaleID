"""Switch prep, loss kind (a): the candidate mask paints a whale but the canon model's visibility map leaves too little of
it (use = vis AND mask < one 400-px component), so parts.split_instances makes no instance and the photo ends empty.
Fallback rule F (switch_prep/rules_switch.md, addendum): where the photo's use map holds no component of >= 400 px but
the mask inside the crop holds one, use = the mask inside the crop (so F changes only photos where it makes an instance). Every other photo is left byte-identical.

This builds the arm offline from an S6-src day folder (no GPU): ARM/DAY/canon = symlinks to the S6-src canon files,
with a rewritten <stem>_canon.npz only for the photos rule F changes; then switch_fallback.sh runs cse onward.
    .venv/bin/python experiments/test_day/fallback_use.py --src S6DAY --dst ARMDAY
-> ARMDAY/canon/, ARMDAY/fallback.json (the photos changed: use px before, mask px in the crop)
The pipeline form of the same rule is a patch to canon/predict.py estimate() (scratch folder, not applied).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

MIN_PX = 400        # parts.split_instances min_px and parts.extract_photo's floor (crop px)
EVAL = {"2024-12-18", "2024-12-26", "2025-02-19", "2025-03-21"}


def needs_fallback(use: np.ndarray) -> bool:
    if use.sum() < MIN_PX:
        return True
    n, _, st, _ = cv2.connectedComponentsWithStats(use.astype(np.uint8), connectivity=8)
    return n <= 1 or st[1:, cv2.CC_STAT_AREA].max() < MIN_PX


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--src", type=Path, required=True); p.add_argument("--dst", type=Path, required=True); a = p.parse_args()
    if a.src.resolve().name in EVAL:
        raise SystemExit("evaluation days are not read")
    src, dst = a.src / "canon", a.dst / "canon"; dst.mkdir(parents=True, exist_ok=True)
    for x in ("images",):
        if not (a.dst / x).exists():
            os.symlink(os.path.realpath(a.src / x), a.dst / x)
    if not (a.dst / "timestamps.json").exists():
        (a.dst / "timestamps.json").write_text((a.src / "timestamps.json").read_text())
    changed = {}
    for f in sorted(src.iterdir()):
        t = dst / f.name
        if t.exists() or t.is_symlink():
            t.unlink()
        if not f.name.endswith("_canon.npz"):
            os.symlink(os.path.realpath(f), t); continue
        stem = f.name[: -len("_canon.npz")]
        z = dict(np.load(f)); use = z["use"].astype(bool)
        if not needs_fallback(use):
            os.symlink(os.path.realpath(f), t); continue
        m = cv2.imread(str(src / f"{stem}_mask.png"), cv2.IMREAD_GRAYSCALE)
        x0, y0, s = (float(v) for v in z["x0y0s"]); M = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], np.float64)
        seg = cv2.warpAffine((m > 0).astype(np.uint8), M, use.shape[::-1], flags=cv2.INTER_NEAREST) > 0
        if needs_fallback(seg):                                   # the mask in the crop has no 400-px piece either: no instance either way
            os.symlink(os.path.realpath(f), t); continue
        z["use"] = seg
        np.savez_compressed(t, **z)
        changed[stem] = {"use_px": int(use.sum()), "mask_in_crop_px": int(seg.sum()), "mask_px_2560": int((m > 0).sum()), "scale": round(s, 4)}
    (a.dst / "fallback.json").write_text(json.dumps({"rule": f"use = mask in crop where no use component >= {MIN_PX} px and the mask in the crop has one", "changed": changed}, indent=1))
    print(a.src.name, "photos changed by rule F:", len(changed))


if __name__ == "__main__":
    main()
