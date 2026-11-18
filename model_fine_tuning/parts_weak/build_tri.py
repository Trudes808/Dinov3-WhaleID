"""W3's weak maps: the clicked fins of 2025-01-02 labelled as the WHOLE clicked triangle (tip / front base / back base),
since the dorsal class now means fin + hump (user's decision 2026-11-17). Everything else as build.py.

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.parts_weak.build_tri
        -> CACHE/maps_tri/<day>/<stem>.png: the clicked photos of 2025-01-02 rebuilt by build.work with the triangle
           (shrunk 5 % towards its centre; still only on whale evidence, as every clicked shape), every other photo a
           symlink to CACHE/maps (identical). index.json is unchanged (same instances and constraints).
2025-03-18 stays the click test and is never read here.
"""
from __future__ import annotations

import json
import os

from . import CACHE, CLICK_TRAIN_DAY, INDEX, MAPS
from . import build as B

OUT = CACHE / "maps_tri"


def main() -> int:
    B.TIP_PART = 1.0                  # the whole triangle
    B.SHRINK_FIN = 0.05
    B.MAPS = OUT
    photos = json.loads(INDEX.read_text())["photos"]
    clicked = {(r["day"], r["stem"]) for r in photos if r["day"] == CLICK_TRAIN_DAY and r["points"]}
    jobs = [j for j in B.jobs() if (j[0], j[1]) in clicked]
    n_pts, errs = 0, 0
    for j in jobs:
        r, err = B.work(j)
        if err:
            errs += 1
            print("error", j[1], err)
        else:
            n_pts += r["points"]
    for r in photos:
        dst = OUT / r["day"] / f"{r['stem']}.png"
        if (r["day"], r["stem"]) in clicked:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            os.symlink(MAPS / r["day"] / f"{r['stem']}.png", dst)
    s = {"rebuilt_photos": len(jobs), "click_shapes": n_pts, "errors": errs, "tip_part": 1.0, "shrink": 0.05}
    (CACHE / "maps_tri_summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
