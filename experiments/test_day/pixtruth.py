"""Pixel truth for the extra classes on the test day 2026-04-07, and scoring of any model on it.
Rules: pixtruth_rules.md (registered before any truth was drawn or any model output opened).

    R=/home/sat3737/NOAA_whales/moonshot_cache/test_days/2026-04-07; P=$R/pixel_truth
    .venv/bin/python experiments/test_day/pixtruth.py sample --root $R            # P/manifest.json (sample list)
    # drawing: pixtruth_draw.py (prompts -> SAM 2.1 -> label PNGs), checked by eye on overlays
    .venv/bin/python experiments/test_day/pixtruth.py freeze --root $R            # hashes + time stamp into the manifest
    .venv/bin/python experiments/test_day/pixtruth_score.py --root $R ...        # scores (see that file)

Photos are named by their 3-digit sequence number everywhere.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

NOWHALE = {"marker_rail", "bow", "crew_nowhale", "marker_land", "water"}
SEED = 20261120
CLASSES = {0: "water_sky", 1: "whale", 2: "person", 3: "rail", 4: "tape", 5: "deck", 6: "pontoon", 7: "other_vessel",
           8: "land", 9: "gear", 255: "unsure"}


def draw_sample(truth: Path) -> dict:
    rows = list(csv.DictReader(open(truth), delimiter="\t"))
    by = {r["seq"]: r for r in rows}
    parts = {
        "nowhale": sorted(s for s, r in by.items() if r["class"] in NOWHALE),
        "whale_crew": sorted(s for s, r in by.items() if r["class"] == "whale_crew"),
        "vessel": sorted(s for s, r in by.items() if r["vessel"] == "1"),
    }
    taken = set().union(*parts.values())
    pool = sorted(s for s, r in by.items() if r["class"] == "whale" and s not in taken
                  and (r["land_bg"] == "1" or r["rail"] == "1" or r["gear"] == "1"))
    parts["random_whale"] = sorted(random.Random(SEED).sample(pool, 30))
    allseq = sorted(set().union(*parts.values()))
    return {"parts": parts, "pool_size": len(pool), "seed": SEED, "sample": allseq,
            "truth_class": {s: by[s]["class"] for s in allseq}}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["sample", "freeze"]); ap.add_argument("--root", type=Path, required=True)
    a = ap.parse_args(); P = a.root / "pixel_truth"; man = P / "manifest.json"
    if a.cmd == "sample":
        if man.exists():
            raise SystemExit("manifest exists; the sample is fixed")
        s = draw_sample(a.root / "truth.tsv")
        m = {"created": time.strftime("%Y-%m-%d %H:%M:%S %z"), "rules": {"file": "rules_pixtruth.md", "sha256": sha(P / "rules_pixtruth.md")},
             "classes": CLASSES, "grid": "half of full resolution, 2748 x 1835", **s}
        man.write_text(json.dumps(m, indent=1)); print(len(s["sample"]), "photos;", {k: len(v) for k, v in s["parts"].items()}, "pool", s["pool_size"])
    else:
        m = json.loads(man.read_text())
        if "frozen" in m:
            raise SystemExit("already frozen at " + m["frozen"]["time"])
        files = sorted((P / "labels").glob("*.png")) + [P / "tape_colours.tsv", P / "notes.tsv"]
        missing = [s for s in m["sample"] if not (P / "labels" / f"{s}.png").exists()]
        if missing:
            raise SystemExit(f"missing labels: {missing}")
        m["frozen"] = {"time": time.strftime("%Y-%m-%d %H:%M:%S %z"), "sha256": {str(f.relative_to(P)): sha(f) for f in files}}
        man.write_text(json.dumps(m, indent=1)); print("frozen", len(files), "files at", m["frozen"]["time"])


if __name__ == "__main__":
    main()
