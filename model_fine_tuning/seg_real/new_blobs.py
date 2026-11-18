"""A candidate's OWN false whales: blobs it calls whale where the live seg_r04_ft4 had nothing (3a is blind to them,
because its units are r04's blobs). Validation days, whole days, from the e2e shadow runs (seg_real.e2e run).

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.seg_real.new_blobs find --names r05,s1,s2,s3,s0
        -> experiments/seg_splash/new_blobs.json: per candidate, per val day: photos, new blobs >= 300 px (2560-px frame)
           that overlap no r04 blob (DAY/instances_v2/blobs, every blob >= 100 px), per photo; and a blind random sample
           of up to 120 per candidate, drawn as sheets SHEETS/newblobs/nb_<name>_NNN.jpg (28 tiles, no score shown)
    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.seg_real.new_blobs merge --names ...
        reads SHEETS/newblobs_raw/nb_<name>_NNN.json (by-eye labels, LABEL_GUIDE.md) -> new_blobs.json "labels"
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter

import cv2
import numpy as np

from . import common as C
from .e2e import E2E
from .sample import blob_tile, write_sheet

OUT = C.EXP / "new_blobs.json"


def cmd_find(a) -> None:
    R = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name in a.names.split(","):
        found, per_day = [], {}
        for day in C.VAL:
            rows = [json.loads(l) for l in open(E2E / name / day / "instances_v2" / "blobs.jsonl")]
            n_new = 0
            for r in rows:
                cand = [b for b in r["blobs"] if b["area"] >= 300]
                if not cand:
                    continue
                old = C.blob_image(day, r["stem"]) > 0
                lab = cv2.imread(str(E2E / name / day / "instances_v2" / "blobs" / f"{r['stem']}.png"), cv2.IMREAD_UNCHANGED)
                for b in cand:
                    x0, y0, x1, y1 = b["bbox"]
                    m = lab[y0:y1, x0:x1] == b["b"]
                    if not (m & old[y0:y1, x0:x1]).any():
                        found.append({"day": day, "stem": r["stem"], "b": b["b"], "area": b["area"], "bbox": b["bbox"]}); n_new += 1
            per_day[day] = {"photos": len(rows), "new_blobs": n_new}
        photos = sum(v["photos"] for v in per_day.values())
        rng = random.Random(0); smp = list(found); rng.shuffle(smp); smp = smp[:a.n]
        for i, t in enumerate(smp):
            t["tile"] = i + 1
        R[name] = {"per_day": per_day, "photos": photos, "new_blobs": len(found), "per_photo": round(len(found) / photos, 4),
                   "area_px": int(sum(f["area"] for f in found)), "sizes": dict(Counter("300-1k" if f["area"] < 1000 else "1k-10k" if f["area"] < 10000 else "10k+" for f in found)),
                   "sample": smp}
        out = C.SHEETS / "newblobs"
        for s in range(0, len(smp), 28):
            ims = []
            for t in smp[s:s + 28]:
                lab = cv2.imread(str(E2E / name / t["day"] / "instances_v2" / "blobs" / f"{t['stem']}.png"), cv2.IMREAD_UNCHANGED)
                ims.append(blob_tile(t, lab=lab))
            write_sheet(ims, out / f"nb_{name}_{s // 28 + 1:03d}.jpg", 4)
        print(name, R[name]["new_blobs"], "new blobs on", photos, "val photos", R[name]["sizes"], "sheets", (len(smp) + 27) // 28, flush=True)
    OUT.write_text(json.dumps(R, indent=0))


def cmd_merge(a) -> None:
    R = json.loads(OUT.read_text())
    for name in a.names.split(","):
        raw = {}
        for f in sorted((C.SHEETS / "newblobs_raw").glob(f"nb_{name}_*.json")):
            raw.update(json.loads(f.read_text()))
        for t in R[name]["sample"]:
            v = raw.get(str(t["tile"]))
            if v:
                t["label"], t["sure"] = v[0], bool(v[1])
        lab = [t for t in R[name]["sample"] if "label" in t]
        c = Counter(t["label"] for t in lab)
        R[name]["labels"] = {"labelled": len(lab), "counts": dict(c.most_common()),
                             "whale_share": round(sum(t["label"] in ("whale", "whale_splash") for t in lab) / max(len(lab), 1), 3),
                             "note": "AI by-eye labels (sub-agents, LABEL_GUIDE.md), blind to the model and to scores"}
        # estimated non-whale new blobs per val photo = new blobs per photo x non-whale share of the sample
        R[name]["labels"]["nonwhale_new_per_photo"] = round(R[name]["per_photo"] * (1 - R[name]["labels"]["whale_share"]), 4)
        print(name, R[name]["labels"])
    OUT.write_text(json.dumps(R, indent=0))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("find"); p.add_argument("--names", required=True); p.add_argument("--n", type=int, default=120); p.set_defaults(fn=cmd_find)
    p = sub.add_parser("merge"); p.add_argument("--names", required=True); p.set_defaults(fn=cmd_merge)
    a = ap.parse_args(argv); a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
