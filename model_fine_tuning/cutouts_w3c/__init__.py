"""Cut-outs W3c (2026-11-18): dorsal fin cut-outs centred and sized on the W3c part model's dorsal region (fin + hump,
the researcher's clicked triangle down to the back line), turn and mirror of parts_up kept. A measurement next to the
cut-outs v3 machinery (model_fine_tuning/cutouts_v3, unchanged); offline, nothing of a day folder is written.
See regions.py (W3c dorsal region per instance), choose.py (the two sizes, from training-day clicks by geometry only),
make.py, embed.py, score.py, analyze.py."""

from pathlib import Path

from ..cutouts_v3 import DAYS, EVAL_DAYS, ROOT, day_list  # noqa: F401  (day_list refuses the evaluation days)

CACHE = Path("/var/tmp/sat3737-parts-cache/cutouts_w3c")
V3_CACHE = Path("/var/tmp/sat3737-parts-cache/cutouts_v3")
SHADOW = Path("/var/tmp/sat3737-parts-cache/parts_weak/days_shadow")       # eval_w.py's class maps of the W3c runs
MODELS = {"s0": "parts_seg_r05_w3c", "s1": "parts_seg_r05_w3cs1"}
RESULTS = ROOT / "experiments" / "cutouts_w3c"
