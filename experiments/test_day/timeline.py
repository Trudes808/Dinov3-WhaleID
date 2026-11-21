"""Timeline strip of the test day over EXIF time: truth class per photo, the marker photos, the gaps, the trip log's groups,
and what the Labeling Assistant would cut (sightings = runs <= 120 s; encounter-cut proposals P >= 0.35) and the safe clusters.

    .venv/bin/python experiments/test_day/timeline.py --root ROOT --triplog TRIP_LOG_DAY.json --out timeline.jpg

Reads ROOT/truth.tsv, ROOT/photos.tsv, ROOT/labeling_day.json (written by loading the day in a scratch copy of the
review backend), ROOT/days/<day>/safe/clusters.json. Prints the marker / gap table used in tables.md.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COL = {"whale": "#3cb44b", "whale_crew": "#f58230", "under": "#46c8dc", "water": "#285aff", "crew_nowhale": "#f032e6",
       "marker_rail": "#e6194b", "marker_land": "#aa6e28", "bow": "#e1c800"}
NAME = {"whale": "whale", "whale_crew": "whale + crew", "under": "under water", "water": "water only", "crew_nowhale": "crew, no whale",
        "marker_rail": "rail marker", "marker_land": "land marker", "bow": "bow cleat"}


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True); p.add_argument("--triplog", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True); a = p.parse_args()
    R = a.root; T = list(csv.DictReader(open(R / "truth.tsv"), delimiter="\t")); L = json.loads((R / "labeling_day.json").read_text())
    tl = json.loads(a.triplog.read_text()); CL = json.loads((R / "days" / R.name / "safe" / "clusters.json").read_text())["clusters"]
    t = [float(r["t"]) for r in T]; day0 = dt.datetime.fromtimestamp(t[0]).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    h = lambda x: (x - day0) / 3600.0
    fig, ax = plt.subplots(figsize=(22, 6.5), dpi=110)
    # trip log groups
    for g in tl["groups"]:
        ax.axvspan(g["start_s"] / 3600, g["end_s"] / 3600, ymin=0.80, ymax=0.97, color="#d9d9d9")
        ax.text((g["start_s"] + g["end_s"]) / 7200, 4.55, f"log group {g['n']} {g['type']}\n{g['start'][:5]}-{g['end'][:5]}", ha="center", va="center", fontsize=9)
    # photos by truth class
    for r, x in zip(T, t):
        big = r["class"] in ("marker_rail", "marker_land", "bow")
        ax.plot([h(x)] * 2, [3.0, 3.9] if big else [3.2, 3.7], color=COL[r["class"]], lw=2.6 if big else 1.0)
        if big:
            mk = {"marker_rail": r["marker"].replace("rail_", "").rstrip("?")[:1].upper(), "marker_land": "L", "bow": "B"}[r["class"]]
            ax.text(h(x), 4.0, mk, ha="center", va="bottom", fontsize=8, color=COL[r["class"]], fontweight="bold")
    # sightings and proposals
    ph = L["photos"]
    for k, (i, j) in enumerate(L["sightings"]):
        ax.plot([h(ph[i]["t"]), h(ph[j]["t"]) + 0.004], [2.4, 2.4], color="#555555" if k % 2 else "#999999", lw=7, solid_capstyle="butt")
    props = [b for b in L["boundaries"] if b["p"] >= 0.35]
    for b in props:
        x = h(ph[b["at"]]["t"]); ax.plot([x, x], [1.9, 2.9], color="black", lw=1.4, ls="--")
        ax.text(x, 1.75, f"P {b['p']:.2f}", ha="center", fontsize=8)
    # safe clusters (one row of short bars, alternating shade)
    for k, c in enumerate(sorted(CL, key=lambda c: c["start"])):
        ax.plot([h(c["start"]), h(c["end"]) + 0.002], [1.2 + 0.15 * (k % 3)] * 2, color="#4363d8", lw=3, solid_capstyle="butt")
    ax.set_yticks([1.35, 2.4, 3.45, 4.55]); ax.set_yticklabels(["safe clusters", "sightings (<=120 s)\n--- cut proposal", "photos (truth)", "trip log"])
    ticks = [x / 4 for x in range(int(h(t[0]) * 4), int(h(t[-1]) * 4) + 2)]
    ax.set_xticks(ticks); ax.set_xticklabels([f"{int(x)}:{int(round((x % 1) * 60)):02d}" for x in ticks]); ax.set_xlim(h(t[0]) - 0.05, h(t[-1]) + 0.05)
    ax.set_ylim(0.9, 5.0); ax.spines[["top", "right", "left"]].set_visible(False); ax.grid(axis="x", color="#eeeeee")
    hs = [plt.Line2D([0], [0], color=c, lw=4) for c in COL.values()]
    ax.legend(hs, [NAME[k] for k in COL], ncol=8, loc="upper center", bbox_to_anchor=(0.5, -0.07), frameon=False, fontsize=10)
    ax.set_title(f"test day 2026-04-07: {len(T)} photos over EXIF time; markers lettered (R/B/G/Y rail tape colour, L land, B bow cleat); "
                 f"{len(L['sightings'])} sightings, {len(props)} cut proposals, {len(CL)} safe clusters", fontsize=12)
    fig.tight_layout(); fig.savefig(a.out, dpi=110, pil_kwargs={"quality": 88}); print("->", a.out)
    # marker / gap table
    for i, r in enumerate(T):
        if r["class"] in ("marker_rail", "marker_land", "bow"):
            b = t[i] - t[i - 1] if i else 0; f = t[i + 1] - t[i] if i + 1 < len(t) else 0
            print(r["seq"], r["time"][:8], r["class"], r["marker"], f"before {b:.0f} s", f"after {f:.0f} s", "prev", T[i - 1]["class"] if i else "", "next", T[i + 1]["class"] if i + 1 < len(T) else "")


if __name__ == "__main__":
    main()
