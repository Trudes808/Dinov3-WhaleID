"""Round 2: the candidate whale mask (NOT live) swapped into the live chain, two ways (rules: round2_rules.md, section 1).

    flock LOCK env PYTHONPATH=. .venv/bin/python experiments/test_day/s6_swap.py canon --images DAY/images --out DAY/canon
        canon.predict unchanged (seg_r04_ft4 replaced; keep_main_components, crop, canon model, overlays as live), with
        segment() returning the candidate mask: seg_splash_s6 P(whale) at long side 1280, hysteresis 0.6 / 0.1, nearest
        to the full frame (as seg_real.followup e2e does it).
    flock LOCK env PYTHONPATH=. .venv/bin/python experiments/test_day/s6_swap.py v2 --day-dir DAY
        instances_v2 candidates with the same mask + assemble with its defaults and join_gap 0.35 -> DAY/instances_v2
Settings are the candidate's own (experiments/LOG.md 2026-11-17 18:20); nothing chosen on the test day.
"""
from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
CKPT = REPO / "experiments" / "seg_splash_s6" / "checkpoints" / "last.pt"
CANON = REPO / "experiments" / "canon_r0304_ft4" / "checkpoints" / "best.pt"


def patch_segment() -> None:
    from model_fine_tuning.segmentation import predict as SP
    from model_fine_tuning.seg_real.followup import mask_of
    orig = SP.segment

    def seg(model, image, device, max_side=1280, threshold=0.5, flip_tta=False):
        small, _ = SP.resize_long_side(image, max_side)
        prob, _ = orig(model, small, device, max_side, threshold, flip_tta)
        m = mask_of(prob, "h0.6_0.1").astype(np.uint8)
        m = cv2.resize(m, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        return None, m
    SP.segment = seg


def main() -> int:
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("canon"); p.add_argument("--images", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("v2"); p.add_argument("--day-dir", required=True); p.add_argument("--join-gap", type=float, default=0.35)
    a = ap.parse_args()
    patch_segment()
    if a.cmd == "canon":
        from model_fine_tuning.canon import predict as CP          # imported after the patch: binds the candidate segment()
        return CP.main(["--seg-checkpoint", str(CKPT), "--checkpoint", str(CANON), "--images", a.images, "--out", a.out, "--max-side", "1280"])
    from model_fine_tuning.cluster import instances_v2 as V
    d = Path(a.day_dir)
    if not (d / "instances_v2" / "feat.npz").exists():
        V.cmd_candidates(Namespace(day_dir=str(d), collect=100, workers=4, seg_checkpoint=str(CKPT)))
    V.cmd_assemble(Namespace(day_dir=str(d), **{**V.DEFAULTS, "join_gap": a.join_gap}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
