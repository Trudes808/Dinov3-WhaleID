"""Per-photo marker table for the test day 2026-04-07 (all photos of the day), from truth.tsv and, for the photos with
pixel truth, from the label PNGs. Describes the day; nothing is fitted or tuned on it.

    R=/home/sat3737/NOAA_whales/moonshot_cache/test_days/2026-04-07
    .venv/bin/python experiments/test_day/pixtruth_markers.py --root $R [--labels DIR]     # -> $R/pixel_truth/markers.tsv

Marker definition (the researcher, 2026-11-21): a rail + tape marker is a deliberate close-up where the rail takes up the
whole photo, the tape is prominent and there is no whale; the tape colour describes the animal of the series just before
(red = female in a group or the mother with a calf, green = male in a competitive group or the escort, blue = calf; yellow
not explained yet). A rail with tape seen small in a busy photo is an accidental picture of the railing, not a marker.
Bollard (bow cleat) close-ups mark that the fluke just photographed is an opportunistic one of another animal, not the
series' animal. Land shots mark that the boat switches to another group.
refers_to (rail and bollard markers): the whale photos before the marker, walking back past the marker's own run of
consecutive markers, until the previous marker or a gap of more than 60 s between consecutive photos.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

W, H = 2748, 1835
MEANING = {"red": "female in a group, or the mother with a calf", "green": "male in a competitive group, or the escort",
           "blue": "calf", "yellow": "unknown (not explained yet)"}
KIND = {"marker_rail": "rail", "marker_land": "land", "bow": "bollard"}


def pixel_facts(lab: np.ndarray) -> dict:
    n = lab.size
    rt = np.isin(lab, [3, 4]).astype(np.uint8)
    tube = 0.0
    if rt.any():   # widest tube = 2 x the largest distance from a rail / tape pixel to the nearest other pixel
        tube = 2 * float(cv2.distanceTransform(np.pad(rt, 1), cv2.DIST_L2, 5).max())
    return {"tape_share": (lab == 4).sum() / n, "rail_share": (lab == 3).sum() / n, "rail_tape_share": rt.sum() / n,
            "tube_px": tube, "tube_over_width": tube / W, "whale_px": int((lab == 1).sum()), "person_px": int((lab == 2).sum())}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--root", type=Path, required=True); ap.add_argument("--labels", type=Path)
    ap.add_argument("--out", type=Path); a = ap.parse_args()
    P = a.root / "pixel_truth"; labdir = a.labels or P / "labels"; out = a.out or P / "markers.tsv"
    rows = list(csv.DictReader(open(a.root / "truth.tsv"), delimiter="\t"))
    colour = {r["seq"]: r["tape_colour_by_eye"] for r in csv.DictReader(open(P / "tape_colours.tsv"), delimiter="\t")}
    sample = set(json.loads((P / "manifest.json").read_text())["sample"])
    kind = [KIND.get(r["class"], "none") for r in rows]
    t = [float(r["t"]) for r in rows]
    res = []
    for i, r in enumerate(rows):
        s = r["seq"]; k = kind[i]
        f = pixel_facts(np.asarray(Image.open(labdir / f"{s}.png"))) if s in sample else None
        col = colour.get(s, "").split(" ")[0] if k == "rail" else ""
        meaning = (MEANING.get(col, "unknown") if k == "rail" else "the boat switches to another group" if k == "land"
                   else "the fluke just photographed is an opportunistic one of another animal, not the series' animal" if k == "bollard" else "")
        refers = []; far = ""
        if k in ("rail", "bollard"):
            j = i - 1
            while j >= 0 and kind[j] != "none" and t[j + 1] - t[j] <= 60:   # the marker's own run of consecutive markers
                j -= 1
            while j >= 0 and kind[j] == "none" and t[j + 1] - t[j] <= 60:
                if rows[j]["has_whale"] == "yes":
                    refers.append(rows[j]["seq"])
                j -= 1
            refers = sorted(refers)
            if not refers:   # nothing within 60 s: name the nearest earlier run of whale photos and how far back it is
                j0 = j
                while j0 >= 0 and rows[j0]["has_whale"] != "yes":
                    j0 -= 1
                run = []
                while j0 >= 0 and rows[j0]["has_whale"] == "yes" and (not run or t[j0 + 1] - t[j0] <= 60):
                    run.append(rows[j0]["seq"]); j0 -= 1
                if run:
                    far = f"none within 60 s; nearest earlier whale photos {min(run)}-{max(run)}, {t[i] - t[int(max(run))]:.0f} s before"
        rail_seen = r["rail"] == "1" or (f is not None and f["rail_tape_share"] > 0)
        tape_seen = f is not None and f["tape_share"] > 0
        whale = r["has_whale"] == "yes" or (f is not None and f["whale_px"] > 0)
        person = r["person"] == "1" or (f is not None and f["person_px"] > 0)
        res.append({
            "seq": s, "time": r["time"], "photo_class": r["class"], "marker": k, "tape_colour": col, "meaning": meaning,
            "refers_to": " ".join(refers), "n_refers_to": str(len(refers)) if k in ("rail", "bollard") else "", "refers_note": far,
            "incidental_rail": "yes" if k == "none" and rail_seen else "no",
            "incidental_tape": ("yes" if tape_seen else "no") if k == "none" and f is not None else ("unknown" if k == "none" and r["rail"] == "1" else "no"),
            "pixel_truth": "yes" if f is not None else "no",
            "tape_share_pct": f"{100 * f['tape_share']:.4f}" if f else "",
            "rail_share_pct": f"{100 * f['rail_share']:.3f}" if f else "",
            "tube_width_px_half": f"{f['tube_px']:.0f}" if f else "",
            "tube_width_over_frame_width": f"{f['tube_over_width']:.4f}" if f else "",
            "whale_in_photo": "yes" if whale else "no", "person_in_photo": "yes" if person else "no"})
    with open(out, "w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res[0]), delimiter="\t", lineterminator="\n"); w.writeheader(); w.writerows(res)
    # summary for the tables
    def rng(v): return f"{min(v):.4g} - {max(v):.4g}" if v else "-"
    groups = {"rail markers": [x for x in res if x["marker"] == "rail"],
              "bollard markers": [x for x in res if x["marker"] == "bollard"],
              "incidental rail, pixel truth": [x for x in res if x["incidental_rail"] == "yes" and x["pixel_truth"] == "yes"],
              "incidental rail + tape, pixel truth": [x for x in res if x["incidental_tape"] == "yes"]}
    for g, xs in groups.items():
        print(f"{g}: n={len(xs)} tape% {rng([float(x['tape_share_pct']) for x in xs if x['tape_share_pct']])} "
              f"rail% {rng([float(x['rail_share_pct']) for x in xs if x['rail_share_pct']])} "
              f"tube/W {rng([float(x['tube_width_over_frame_width']) for x in xs if x['tube_width_over_frame_width']])} "
              f"whale {sum(x['whale_in_photo'] == 'yes' for x in xs)} person {sum(x['person_in_photo'] == 'yes' for x in xs)}")
    print("incidental rail without pixel truth:", [x["seq"] for x in res if x["incidental_rail"] == "yes" and x["pixel_truth"] == "no"])
    for x in res:
        if x["marker"] in ("rail", "bollard"):
            print(x["seq"], x["marker"], x["tape_colour"], x["n_refers_to"], x["refers_to"][:80])
    print("wrote", len(res), "rows to", out)


if __name__ == "__main__":
    main()
