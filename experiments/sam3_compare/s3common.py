"""Shared readers for the SAM 3 comparison: raw SAM 3 outputs (run_sam3.py), our masks on the val / click / test days."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
LIVE = REPO / "experiments" / "days_all"
NW = Path("/home/sat3737/NOAA_whales")
S3 = NW / "moonshot_cache" / "sam3_compare"
RAW = S3 / "raw"
TD = NW / "moonshot_cache" / "test_days" / "2026-04-07"
SP = NW / "moonshot_cache" / "test_days" / "switch_prep"
VAL = ["2025-01-15", "2025-02-27", "2025-03-14"]
CLICK = ["2025-01-02", "2025-03-18"]
EVAL = {"2024-12-18", "2024-12-26", "2025-02-19", "2025-03-21"}
PROMPTS = json.loads((Path(__file__).parent / "prompts_all.json").read_text())


@lru_cache(maxsize=None)
def prompt_index(set_name: str) -> dict:
    return {p: i for i, p in enumerate(json.loads((RAW / set_name / "_prompts.json").read_text()))}


def load(set_name: str, pid: str) -> dict | None:
    f = RAW / set_name / f"{pid}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    return {k: z[k] for k in z.files}


def inst(rec: dict, j: int, thr: float, hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, list]:
    """scores, boxes (scaled to hw), boolean masks at hw for the instances of prompt j with score >= thr."""
    s = rec[f"s{j}"]; keep = s >= thr
    H0, W0 = rec["hw"]; H, W = hw
    b = rec[f"b{j}"][keep] * np.array([W / W0, H / H0, W / W0, H / H0], np.float32)
    ms = [cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR) >= 128 for m in rec[f"m{j}"][keep]]
    return s[keep], b, ms


def union(rec: dict, j: int | list, thr: float, hw) -> np.ndarray:
    js = [j] if isinstance(j, int) else j
    u = np.zeros(hw, bool)
    for jj in js:
        for m in inst(rec, jj, thr, hw)[2]:
            u |= m
    return u


def deoverlap(scores, masks) -> list:
    """Each pixel to the highest-scoring instance covering it; returns masks in score order (empty ones dropped)."""
    order = np.argsort(-np.asarray(scores)); taken = None; out = []
    for i in order:
        m = masks[i] if taken is None else masks[i] & ~taken
        taken = masks[i].copy() if taken is None else taken | masks[i]
        if m.any():
            out.append(m)
    return out


def read_bin(path: Path, hw) -> np.ndarray | None:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    return cv2.resize(m, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST) > 0


def day_stems(day_dir: Path) -> list[str]:
    return [Path(json.loads(l)["image"]).stem for l in open(day_dir / "images" / "index.jsonl")]


def live_mask(day: str, stem: str, hw):
    return read_bin(LIVE / day / "canon" / f"{stem}_mask.png", hw)


def s6_mask_val(day: str, stem: str, hw):
    return read_bin(SP / "days_s6" / day / "canon" / f"{stem}_mask.png", hw)


@lru_cache(maxsize=1)
def test_stems() -> dict:
    D = TD / "days" / "2026-04-07"
    return {s.split("_")[-2]: s for s in day_stems(D)}


def test_live(seq: str, hw):
    return read_bin(TD / "days" / "2026-04-07" / "canon" / f"{test_stems()[seq]}_mask.png", hw)


def test_s6(seq: str, hw):
    return read_bin(TD / "cand_s6" / f"{test_stems()[seq]}_s6.png", hw)


def catalogue() -> dict:
    return json.loads((REPO / "model_fine_tuning" / "splits" / "labels_all_days.json").read_text())["days"]


def truth_tsv() -> dict:
    rows = [l.rstrip("\n").split("\t") for l in open(TD / "truth.tsv")]
    h = rows[0]
    return {r[0]: dict(zip(h, r)) for r in rows[1:]}
