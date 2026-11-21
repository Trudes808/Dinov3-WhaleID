"""Load a test day the way the Labeling Assistant does (backend.labeling.load_day) in a SCRATCH COPY of Whale-Review-Web,
with the day folders, RAW root and labelling state pointed at test folders; the live app and its data are not touched.

    WHALE_DAYS_ROOT=DAYS_ROOT WHALE_RAW_ROOT=RAW_ROOT WHALE_LABELING_STATE=STATE \
        Whale-Review-Web/.venv/bin/python experiments/test_day/labeling_load.py --webcopy WEBCOPY --day 2026-04-07 --out OUT.json
WEBCOPY = a copy of Whale-Review-Web holding backend/, frontend/ and data/trip_log/ (no cached day data).
Writes {"stacks", "sightings", "boundaries", "photos"} as round 1's labeling_day.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--webcopy", type=Path, required=True); p.add_argument("--day", required=True); p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if "Whale-Review-Web" in str(a.webcopy.resolve()).split("/scratchpad")[0] and "scratchpad" not in str(a.webcopy.resolve()):
        raise SystemExit("refusing to load into the live Whale-Review-Web folder")
    sys.path.insert(0, str(a.webcopy))
    from backend import labeling
    D = labeling.load_day(a.day)
    keep = {k: D[k] for k in ("stacks", "sightings", "boundaries", "photos") if k in D}
    a.out.write_text(json.dumps(keep, default=str)); print({k: len(v) for k, v in keep.items()}, "->", a.out)


if __name__ == "__main__":
    main()
