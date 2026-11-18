"""Side experiment (2026-11-17): pose-estimation ideas from people applied to humpback flukes and dorsal fins.

Keypoints with a per-point state on frozen DINOv3 features, trained on renders, then the fin / fluke rectified to a
canonical frame from the points; tested on the researcher's clicks and on the researcher's paired regions ("the same
place on two photos") of the two click days 2025-01-02 and 2025-03-18 (training days). Offline only: nothing in a day
folder, the web app or the pipeline is written. Sheets go to the scratchpad, never into the repo.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DAYS = ROOT / "experiments" / "days_all"
LAB = ROOT.parent / "Whale-Review-Web" / "data"
OUT = ROOT / "experiments" / "pose_side"
CACHE = Path("/var/tmp/sat3737-parts-cache/pose_side")
SHEETS = Path("/tmp/claude-1072903897/-home-sat3737-NOAA-whales/4fbba7ff-3ba5-4b50-9cb7-970c81250409/scratchpad/pose_side")
CUT = Path("/var/tmp/sat3737-parts-cache/cutouts_v3")
FACTORY = ROOT.parent / "Whale-Image-Data-Factory" / "data" / "renders"
EVAL_DAYS = {"2024-12-18", "2024-12-26", "2025-02-19", "2025-03-21"}
CLICK_DAYS = ("2025-01-02", "2025-03-18")


def guard(days):
    bad = [d for d in days if d in EVAL_DAYS]
    if bad:
        raise SystemExit(f"evaluation days must not be read: {bad}")
    return list(days)
