"""Per-photo readings for removal rule v2 ("does anything see a trace of a whale?"). Text only in this repo; outputs go to
the untracked moonshot_cache/removal_v2/. Nothing here is fitted.

    .venv/bin/python experiments/removal_v2/readings.py --day DAY --day-dir DAYDIR --s6prob DIR [--scene DIR]
        [--sam3 DIR --sam3-key stem|seq] [--sam3-extra DIR] [--kept-day DIR] [--reasons s6_DAY.json] [--stems LIST] --out OUT.jsonl

DAYDIR    a day folder with images/index.jsonl (2560-px JPEGs) and timestamps.json
--s6prob  S6 P(whale) maps, <stem>.png uint8 (x255) at long side 1280 (seg_multi/markers.py cache layout)
--scene   scene-model maps <stem>.png (0 water/sky, 1 whale, 2 person, 3 rail, 4 tape, 5 deck, 6 land)
--sam3    raw SAM 3 outputs (experiments/sam3_compare/run_sam3.py; prompts_all.json order) <key>.npz
--sam3-extra  raw SAM 3 outputs of the trace prompts (experiments/removal_v2/prompts_trace.json) <stem>.npz
--old-kept-day  the old live chain's day folder (seg_r04_ft4): live_kept per photo
--kept-day    a live-chain day folder (parts_hr/index.jsonl + safe/not_whale.json): kept instances per photo
--reasons     experiments/test_day/reasons.py output: live_kept (old live mask, whale-scored blobs) and ADE shares
Per photo: S6h gate (hysteresis 0.6 / 0.1, >= 50 px, as markers.py), S6 max / pixels >= 0.1 / 0.25, scene shares and
whale pixels, SAM 3 best score per phrase family, glow (turquoise light under the surface), white water (top-hat foam outside person / rail / tape / deck / land).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

EVAL = {"2024-12-18", "2024-12-26", "2025-02-19", "2025-03-21"}
SEALED = "raw_test_days_unsorted/2026-3-30"
SCENE = ["water_sky", "whale", "person", "rail", "tape", "deck", "land"]
FAMILIES = {"whale": ["whale", "humpback whale", "marine mammal"], "fluke_fin": ["whale tail", "whale fluke", "dorsal fin"],
            "person": ["person"], "land": ["land", "mountain", "island"], "rail": ["metal pole", "railing"], "deck": ["boat deck", "cleat", "bollard"]}
SEQ = re.compile(r"_(\d{3})_[A-Za-z0-9]+$")


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model_fine_tuning.seg_multi.removal import kept_by_photo, s6_read, sam3_read as _sam3_read, water_read  # noqa: E402  (one copy of the readings)


def sam3_read(npz: Path, order: list[str], fams: dict) -> dict:
    return _sam3_read(npz, order, fams)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day", required=True); ap.add_argument("--day-dir", type=Path, required=True); ap.add_argument("--s6prob", type=Path, required=True)
    ap.add_argument("--scene", type=Path); ap.add_argument("--sam3", type=Path); ap.add_argument("--sam3-key", default="stem", choices=["stem", "seq"])
    ap.add_argument("--sam3-extra", type=Path); ap.add_argument("--kept-day", type=Path); ap.add_argument("--reasons", type=Path)
    ap.add_argument("--old-kept-day", type=Path, help="the old live chain's day folder (live mask seg_r04_ft4): its kept instances = live_kept")
    ap.add_argument("--stems", type=Path, help="only these stems (one per line)"); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.day in EVAL:
        raise SystemExit("evaluation days are not read")
    for p in (a.day_dir, a.s6prob, a.scene, a.sam3, a.sam3_extra, a.kept_day):
        if p is not None and SEALED in str(Path(p).resolve()):
            raise SystemExit("sealed folder: refused")
    stems = [Path(json.loads(l)["image"]).stem for l in open(a.day_dir / "images" / "index.jsonl")]
    if a.stems:
        want = set(a.stems.read_text().split()); stems = [s for s in stems if s in want]
    order = json.loads((a.sam3 / "_prompts.json").read_text()) if a.sam3 else None
    xorder = json.loads((a.sam3_extra / "_prompts.json").read_text()) if a.sam3_extra else None
    kept = kept_by_photo(a.kept_day) if a.kept_day else None
    old = kept_by_photo(a.old_kept_day) if a.old_kept_day else None
    R = json.loads(a.reasons.read_text())["photos"] if a.reasons else None
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for s in stems:
            m = SEQ.search(s); r = {"photo": s, "day": a.day, "seq": m.group(1) if m else None}
            r.update(s6_read(cv2.imread(str(a.s6prob / f"{s}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0))
            sc = None
            if a.scene and (a.scene / f"{s}.png").exists():
                sc = cv2.imread(str(a.scene / f"{s}.png"), cv2.IMREAD_UNCHANGED); sc = np.where(sc == 255, 0, sc)
                r["scene"] = {c: round(float((sc == k).mean()), 5) for k, c in enumerate(SCENE)}; r["scene_whale_px"] = int((sc == 1).sum())
            r.update(water_read(cv2.imread(str(a.day_dir / "images" / f"{s}.jpg")), sc))
            if a.sam3:
                key = r["seq"] if a.sam3_key == "seq" else s
                if (a.sam3 / f"{key}.npz").exists():
                    r.update(sam3_read(a.sam3 / f"{key}.npz", order, FAMILIES))
            if a.sam3_extra and (a.sam3_extra / f"{s}.npz").exists():
                r.update(sam3_read(a.sam3_extra / f"{s}.npz", xorder, {q.replace(" ", "_"): [q] for q in xorder}))
            if kept is not None:
                r["kept"] = kept.get(s, 0)
            if old is not None:
                r["live_kept"] = old.get(s, 0)
            if R is not None and s in R:
                r["live_kept"] = R[s]["live_kept"]; r["ade"] = R[s]["shares"]
            f.write(json.dumps(r) + "\n")
    print(len(stems), "->", a.out)


if __name__ == "__main__":
    main()
