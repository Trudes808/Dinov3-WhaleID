"""Turn the by-eye notes of a test day into truth.tsv (all inputs and outputs in the untracked day folder).

    .venv/bin/python experiments/test_day/build_truth.py --root /home/sat3737/NOAA_whales/moonshot_cache/test_days/2026-04-07

Reads ROOT/truth_raw.txt (seq|whales|tags|marker|sharp|cull|note, the blind pass), ROOT/truth_overrides.txt
(seq|field=value;... adjudications from larger views) and ROOT/exif.tsv. Writes ROOT/truth.tsv with one row per photo:
has_whale yes / under (whale only below the surface or awash) / no; a '?' marks an unsure value; and one primary class
per photo for the sheets:
  marker_land, marker_rail, bow (the bow-cleat close-ups), crew_nowhale, water (nothing but sea / splash / footprint),
  under, whale_crew (a whale with crew in frame), whale.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

TAGS = ["far", "blow", "splash", "person", "rail", "deck", "gear", "vessel", "land_subj", "land_bg", "water", "underwater"]


def build(root: Path) -> list[dict]:
    ex = {r["seq"]: r for r in csv.DictReader(open(root / "exif.tsv"), delimiter="\t")}
    over = {}
    for l in open(root / "truth_overrides.txt"):
        if l.startswith("#") or not l.strip():
            continue
        s, kv = l.strip().split("|", 1); over[s] = dict(x.split("=", 1) for x in kv.split(";"))
    rows = []
    for l in open(root / "truth_raw.txt"):
        if l.startswith("#") or not l.strip():
            continue
        seq, wh, tags, marker, sharp, cull, note = (l.rstrip("\n").split("|") + [""] * 7)[:7]
        tg = {t.rstrip("?"): ("?" if t.endswith("?") else "1") for t in tags.split(",") if t}
        n = wh.rstrip("?"); unsure_n = wh.endswith("?")
        has = "yes" if n not in ("0",) else "no"
        if "underwater" in tg and n != "0" and unsure_n:
            has = "under"
        o = over.get(seq, {})
        if "has" in o:
            has = o["has"]
        if o.get("person") == "1":
            tg["person"] = "1"
        h = has.rstrip("?")
        m = marker.rstrip("?")
        if m == "land":
            cls = "marker_land"
        elif m.startswith("rail_"):
            cls = "marker_rail"
        elif m == "bow":
            cls = "bow"
        elif h == "no" and tg.get("person"):
            cls = "crew_nowhale"
        elif h == "no":
            cls = "water"
        elif h == "under":
            cls = "under"
        elif tg.get("person"):
            cls = "whale_crew"
        else:
            cls = "whale"
        e = ex[seq]
        row = {"seq": seq, "frame": e["frame"], "time": e["datetime"][11:] + "." + e["subsec"][:2], "t": e["time"], "focal": e["focal"],
               "has_whale": has, "n_whales": ("0" if h != "yes" else n) + ("?" if unsure_n else ""), "class": cls,
               "marker": marker if marker != "none" else "", **{t: tg.get(t, "") for t in TAGS},
               "sharp": sharp, "cull_guess": cull.split(":")[0], "cull_reason": cull.split(":", 1)[1] if ":" in cull else "", "note": note}
        rows.append(row)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True); a = p.parse_args()
    rows = build(a.root)
    with open(a.root / "truth.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t"); w.writeheader(); w.writerows(rows)
    from collections import Counter
    print(len(rows), "photos;", dict(Counter(r["class"] for r in rows)), dict(Counter(r["has_whale"] for r in rows)))


if __name__ == "__main__":
    main()
