"""Part-model gap instrument: shared loading (real instances, model arms, decision rules) and the taxonomy.

Never reads the evaluation days (refused in code). Sheets with real photos go to the scratchpad only (SCRATCH).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..common import paths
from ..parts_seg import evaluate_real as ER

ROOT = paths.REPO_ROOT
DAYS_ALL = paths.EXPERIMENTS_DIR / "days_all"
SPLIT = json.loads((paths.SPLITS_DIR / "days.json").read_text())
EVAL_DAYS = set(SPLIT["eval"])
TRAIN_DAYS, VAL_DAYS = list(SPLIT["train"]), list(SPLIT["val"])
assert not EVAL_DAYS & set(TRAIN_DAYS + VAL_DAYS)
OUT = paths.EXPERIMENTS_DIR / "part_gaps"                 # small json only (no photos: the repo is public)
SCRATCH = Path("/tmp/claude-1072903897/-home-sat3737-NOAA-whales/4fbba7ff-3ba5-4b50-9cb7-970c81250409/scratchpad/part_gaps")
CACHE = Path("/var/tmp/sat3737-parts-cache/part_gaps")
SHADOW_ROOTS = [DAYS_ALL, Path("/var/tmp/sat3737-parts-cache/realwater/days_shadow"),
                Path("/var/tmp/sat3737-parts-cache/parts_weak/days_shadow")]
FACTORY = ROOT.parent / "Whale-Image-Data-Factory" / "data" / "renders"
CLASSES = ["background", "fluke", "dorsal_fin", "pectoral_fin", "head", "body"]


def check_day(day: str) -> None:
    if day in EVAL_DAYS:
        raise RuntimeError(f"evaluation day {day} must not be read")


def arm_root(name: str) -> Path | None:
    """the day-folder root that holds DAY/<name>/instances.json for this model (baseline in days_all, arms in shadows)."""
    roots = list(SHADOW_ROOTS) + sorted(Path("/var/tmp/sat3737-parts-cache").glob("*/days_shadow"))
    for r in roots:
        if (r / VAL_DAYS[0] / name / "instances.json").exists():
            return r
    return None


def arm_rows(name: str, days: list[str]) -> list[dict]:
    """evaluate_real rows (one per real instance) of model <name>, with t_eye / t_vc truths."""
    for d in days:
        check_day(d)
    root = arm_root(name)
    if root is None:
        raise FileNotFoundError(f"no real-photo outputs for {name} (DAY/{name}/instances.json) under {SHADOW_ROOTS}")
    old = ER.DAYS
    ER.DAYS = root
    try:
        rows = ER.load_all(name, days)
    finally:
        ER.DAYS = old
    for r in rows:
        r["t_eye"] = ER.eye_three(r["eye"])
        r["t_vc"] = ER.vc_three(r["vc"]) if not r["not_whale"] else None
    return rows


def choose_rule(rows: list[dict], variant: str):
    """the fluke / fin / other decision rule, chosen on the TRAINING days in evaluate_real's two-part family (as
    experiments/realwater/compare.py does for every arm)."""
    tf, td, _ = ER.choose_rule(rows, variant)
    return (tuple(tf) if isinstance(tf, (list, tuple)) else (tf, 0.0)), td


def pred_shares(p: dict) -> dict:
    px = p.get("pixels_near", p["pixels"])
    tot = max(p["covered"], 1)
    return {c: px[c] / tot for c in CLASSES}


def size_px(bbox) -> float | None:
    if not bbox:
        return None
    x1, y1, x2, y2 = bbox
    return float(max(x2 - x1, y2 - y1))


SIZE_BINS = [(0, 200, "<200"), (200, 400, "200-400"), (400, 800, "400-800"), (800, 1e9, ">=800")]


def size_bin(L) -> str:
    if L is None:
        return "?"
    for a, b, n in SIZE_BINS:
        if a <= L < b:
            return n
    return "?"


def vc_stratum(r: dict) -> str:
    if r["not_whale"]:
        return "not_whale"
    t = ER.vc_three(r["vc"])
    return t or "uncertain"


# ------------------------------------------------------------------------------------------------ taxonomy
# One primary category per instance (what part of the whale the instance shows), plus attribute flags. Codes are typed
# by eye on blind sheets (the photo only, never the model's output). AI labels, not the researcher's.
CATS = {
    "F_up":    ("fluke", "fluke raised, seen from behind or in front: both blades spread, clear of the water"),
    "F_low":   ("fluke", "fluke rising low or rolling out: water pouring / sheeting off, blades near the surface"),
    "F_edge":  ("fluke", "fluke edge-on or side-on: blades seen in profile, single blade, peduncle + fluke in side view"),
    "F_tips":  ("fluke", "fluke tips only, sinking or mostly submerged fluke"),
    "D_tall":  ("dorsal", "dorsal fin clearly visible, tall / falcate blade on the back"),
    "D_small": ("dorsal", "dorsal fin small, stubby or knob-like, usually on a hump (arched back)"),
    "D_tip":   ("dorsal", "dorsal fin only partly out of the water, tip / far or tiny fin, or fin seen end-on"),
    "B_back":  ("other", "back or peduncle with no dorsal fin in view (fin behind the back, under water or cut off)"),
    "P_pec":   ("other", "pectoral fin (raised, slapping, or white flipper on the surface)"),
    "H_head":  ("other", "head / rostrum / blowholes / chin (head rise, lunge, blow with head only)"),
    "R_roll":  ("other", "belly up, roll, breach, body slap, tail-stock throw seen from the side"),
    "X_none":  (None, "not a whale or unreadable: splash only, wake, blow only, boat, water"),
}
ATTRS = {
    "s1": "some splash / white water touching the whale", "s2": "heavy splash, much of the part hidden",
    "m": "another whale in the crop or touching this one", "e": "cut by the photo edge",
    "b": "blurred / out of focus / backlit silhouette", "h": "hump visible (dorsal views)",
    "k": "hooked / falcate fin shape notable (dorsal views)", "w": "water sheeting / pouring (fluke)",
    "a": "ambiguous on a second look: low fin on a hump / back seen side-on, or a fluke blade seen edge-on",
    "a2": "ambiguous on a second look: pectoral fin, or a single fluke blade",
}
THREE_OF = {k: v[0] for k, v in CATS.items()}


def parse_code(code: str) -> dict:
    """'D_small h s1 m' -> {'cat': 'D_small', 'attrs': ['h', 's1', 'm'], 'three': 'dorsal'}"""
    parts = code.split()
    cat = parts[0]
    if cat not in CATS:
        raise ValueError(f"unknown category {cat!r} in {code!r}")
    attrs = parts[1:]
    for a in attrs:
        if a not in ATTRS:
            raise ValueError(f"unknown attribute {a!r} in {code!r}")
    return {"cat": cat, "attrs": attrs, "three": THREE_OF[cat]}


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))
