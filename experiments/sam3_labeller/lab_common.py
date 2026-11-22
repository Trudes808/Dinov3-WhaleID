"""Shared paths and helpers for the SAM 3 labeller experiment (rules_labeller.md). Text only in the repo; every
per-photo file goes to the untracked moonshot_cache/sam3_labeller/. Evaluation days are refused."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
NW = Path("/home/sat3737/NOAA_whales")
OUT = NW / "moonshot_cache" / "sam3_labeller"
S3RAW = NW / "moonshot_cache" / "sam3_compare" / "raw"          # the comparison's raw SAM 3 outputs (read-only)
DAYS = REPO / "experiments" / "days_all"
SPLIT = json.loads((REPO / "model_fine_tuning" / "splits" / "days.json").read_text())
TRAIN, VAL, EVAL = SPLIT["train"], SPLIT["val"], set(SPLIT["eval"])
TEST = "2026-04-07"
TD = NW / "moonshot_cache" / "test_days" / TEST
PROMPTS = json.loads((REPO / "experiments" / "sam3_compare" / "prompts_all.json").read_text())
PRM = PROMPTS["prompts"]
S6 = REPO / "experiments" / "seg_splash_s6" / "checkpoints" / "last.pt"
SCRATCH = Path("/tmp/claude-1072903897/-home-sat3737-NOAA-whales/4fbba7ff-3ba5-4b50-9cb7-970c81250409/scratchpad/sam3lab")
LOCK = "/var/tmp/sat3737-parts-cache/gpu_train.lock"
IGNORE = 255
# our classes
CLS = ["water_sky", "whale", "person", "rail", "tape", "deck", "land"]
TO_TRUTH = np.array([0, 1, 2, 3, 4, 5, 8] + [255] * 249, np.uint8)
TO_TRUTH[255] = 255


def check_day(day: str) -> None:
    if day in EVAL:
        raise RuntimeError(f"{day} is an evaluation day and is never read")


def day_stems(day: str) -> list[str]:
    check_day(day)
    return [Path(json.loads(l)["image"]).stem for l in open(DAYS / day / "images" / "index.jsonl")]


def photo_path(day: str, stem: str) -> Path:
    check_day(day)
    if day == TEST:
        return TD / "days" / TEST / "images" / f"{stem}.jpg"
    return DAYS / day / "images" / f"{stem}.jpg"


def resize_1280(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]; s = 1280 / max(h, w)
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)


def hyst(p: np.ndarray, hi: float = 0.6, lo: float = 0.1) -> np.ndarray:
    n, lab = cv2.connectedComponents((p >= lo).astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool); keep[np.unique(lab[p >= hi])] = True; keep[0] = False
    return keep[lab]


def s6_prob(day: str, stem: str) -> np.ndarray:
    f = OUT / "s6prob" / day / f"{stem}.png"
    return cv2.imread(str(f), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0


def load_raw(path: Path) -> dict | None:
    if not path.exists():
        return None
    z = np.load(path)
    return {k: z[k] for k in z.files}


def class_mask(rec: dict, idx: list[int], lo: float, hw, hi: float = 1.01) -> np.ndarray:
    """union of the instances of prompt indices idx with lo <= score < hi, mask >= 0.5, at hw (H, W)."""
    H, W = hw; u = np.zeros(hw, bool)
    for j in idx:
        s = rec.get(f"s{j}")
        if s is None:
            continue
        for k in np.nonzero((s >= lo) & (s < hi))[0]:
            u |= cv2.resize(rec[f"m{j}"][k], (W, H), interpolation=cv2.INTER_LINEAR) >= 128
    return u


def disk(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def day_stems_any(day: str) -> list[str]:
    """day_stems, plus the test day (read only at its single scored run, rules part C)."""
    if day == TEST:
        return [Path(json.loads(l)["image"]).stem for l in open(TD / "days" / TEST / "images" / "index.jsonl")]
    return day_stems(day)


def seq_of(stem: str) -> str:
    return stem.split("_")[-2]
