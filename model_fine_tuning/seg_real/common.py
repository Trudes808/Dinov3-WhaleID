"""Shared paths, split guards and blob access for seg_real (read-only on day folders; never reads evaluation days).

Units are the blobs of DAY/instances_v2/blobs/<stem>.png (seg_r04_ft4 raw mask at long side 1280, threshold 0.5,
8-connected, >= 100 px of the 2560-px frame; id 1 = largest) made by cluster/instances_v2.py candidates.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DAYS = REPO / "experiments" / "days_all"
EXP = REPO / "experiments" / "seg_splash"
SPLIT = json.loads((REPO / "model_fine_tuning" / "splits" / "days.json").read_text())
TRAIN, VAL, EVAL = SPLIT["train"], SPLIT["val"], SPLIT["eval"]
CACHE = Path("/var/tmp/sat3737-parts-cache/seg_splash")
SHEETS = Path("/tmp/claude-1072903897/-home-sat3737-NOAA-whales/4fbba7ff-3ba5-4b50-9cb7-970c81250409/scratchpad/seg_splash")
LOCK = Path("/var/tmp/sat3737-parts-cache/gpu_train.lock")
RUN05 = Path("/home/sat3737/NOAA_whales/Whale-Image-Data-Factory/data/renders/run05")
SEG_R04 = REPO / "experiments" / "seg_r04_ft4" / "checkpoints" / "best.pt"
SEG_R05 = REPO / "experiments" / "seg_r05_ft4" / "checkpoints" / "best.pt"
assert not set(TRAIN + VAL) & set(EVAL)

# label categories (blob_labels.json)
CATS = ["whale", "whale_splash", "splash", "wake", "water", "shore", "boat", "person", "pole", "drone", "bird", "other", "unclear"]
CAT_NAMES = {"whale": "whale (whole or clear part)", "whale_splash": "whale under / behind splash (whale + white water mixed)",
             "splash": "splash or spray (incl. blow)", "wake": "wake, white water, foam line", "water": "plain water or glitter",
             "shore": "shore or land", "boat": "boat or boat part", "person": "person", "pole": "pole, rod or line",
             "drone": "drone", "bird": "bird", "other": "other", "unclear": "unclear"}
NEG = {"splash", "wake", "water", "shore", "boat", "person", "pole", "drone", "bird", "other"}
POS = {"whale"}
IGN = {"whale_splash", "unclear"}


def check_day(day: str) -> None:
    if day in EVAL:
        raise RuntimeError(f"{day} is an evaluation day and is never read")
    if day not in TRAIN + VAL:
        raise RuntimeError(f"{day} is not a train / val day")


def split_of(day: str) -> str:
    check_day(day)
    return "train" if day in TRAIN else "val"


def blob_rows(day: str) -> dict:
    check_day(day)
    return {json.loads(l)["stem"]: json.loads(l) for l in open(DAYS / day / "instances_v2" / "blobs.jsonl")}


def blob_image(day: str, stem: str) -> np.ndarray:
    check_day(day)
    return cv2.imread(str(DAYS / day / "instances_v2" / "blobs" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)


def photo_path(day: str, stem: str) -> Path:
    check_day(day)
    return DAYS / day / "images" / f"{stem}.jpg"


def photo(day: str, stem: str) -> np.ndarray:
    """BGR, 2560-px frame."""
    return cv2.imread(str(photo_path(day, stem)), cv2.IMREAD_COLOR)


def load_labels(path: Path = EXP / "blob_labels.json") -> list:
    return json.loads(path.read_text())["blobs"]


def key_of(day: str, stem: str, b: int) -> str:
    return f"{day}|{stem}|{b}"
