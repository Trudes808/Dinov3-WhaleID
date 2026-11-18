"""Weak real-photo supervision for the part segmentation (parts_seg, unchanged): render batches and their dense labels
stay as they are, real crops of TRAINING-day photos with constraints that are certainly true are added to each batch.

  build.py   per photo a weak label map (certain background, clicked points) and an instance map, plus the per-instance
             constraints (forbidden / required part classes) from the by-eye view codes and the view classifier
  data.py    the real crops (instance-centred or at the render crop scale), preview sheet
  losses.py  the weak losses
  train.py   wraps parts_seg.train (same recipe), adds the real crops to every render batch

Never reads the evaluation days; the validation days and the click test day 2025-03-18 are never used for training.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DAYS = ROOT / "experiments" / "days_all"
SPLIT = json.loads((ROOT / "model_fine_tuning" / "splits" / "days.json").read_text())
EVAL_DAYS = set(SPLIT["eval"])
VAL_DAYS = set(SPLIT["val"])
CLICK_TEST_DAY = "2025-03-18"                    # the researcher's clicks of this day are the click test: never trained on
CLICK_TRAIN_DAY = "2025-01-02"
TRAIN_DAYS = [d for d in SPLIT["train"] if d != CLICK_TEST_DAY]
CACHE = Path("/var/tmp/sat3737-parts-cache/parts_weak")
MAPS = CACHE / "maps"
INDEX = CACHE / "index.json"
SHEETS = Path("/tmp/claude-1072903897/-home-sat3737-NOAA-whales/4fbba7ff-3ba5-4b50-9cb7-970c81250409/scratchpad/parts_weak")
LAB = ROOT.parent / "Whale-Review-Web" / "data"            # read only
assert not EVAL_DAYS & set(TRAIN_DAYS) and not VAL_DAYS & set(TRAIN_DAYS)

# class ids of parts_seg.labels.CLASSES
BG, FLUKE, FIN, PEC, HEAD, BODY = range(6)
IGNORE = 255
NOTFIN = 6                                          # weak-label value: whale, certainly not dorsal (W3c, build_tri2.py)
MAX_INST = 8                                        # instances per photo carried into a crop (ids 1..8 of the instance map)
OVERLAP = 254                                       # instance-map value where two instances' regions overlap


def check_train_day(day: str) -> None:
    if day not in TRAIN_DAYS:
        raise RuntimeError(f"{day} is not a weak-label training day (eval / val / click-test days are never used)")
