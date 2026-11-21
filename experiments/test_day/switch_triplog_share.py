"""Switch prep (2026-11-20): the trip log's clock question (Whale-Review-Web backend/trip_log.py CLOCK_WARN = 0.5) on
the live and the shadow root. Per day with a trip log: the share the page tests (main + "no whale" stacks whose median
photo time lies inside a noted humpback window), and candidate replacements: the share of PHOTOS (in those stacks)
inside a window, the stack share with windows widened by the page's own near rule (NEAR_BEFORE / NEAR_AFTER), and
whether place() raises the question / proposes an offset. Run in a scratch copy of the app (WHALE_DAYS_ROOT set).

    WHALE_DAYS_ROOT=ROOT .venv/bin/python experiments/test_day/switch_triplog_share.py --webcopy WEBCOPY --days D1,D2 --out OUT.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--webcopy", type=Path, required=True); ap.add_argument("--days", required=True); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if "scratchpad" not in str(a.webcopy.resolve()):
        raise SystemExit("refusing: the web copy must live in a scratch folder")
    sys.path.insert(0, str(a.webcopy))
    from backend import labeling, trip_log
    out = {}
    for day in a.days.split(","):
        L = trip_log.read_day(day)
        if not L or not any(g.get("humpback") for g in L.get("groups", [])):
            out[day] = None; continue
        try:
            D = labeling.load_day(day)
        except Exception as e:                    # not processed in this root
            out[day] = {"error": str(e)[:80]}; continue
        photos = trip_log.page_photos(D["photos"])
        W = trip_log._windows([g for g in L.get("groups", []) if g.get("n") is not None], None); hump = [w for w in W if w["humpback"]]
        main = [s for s in D["stacks"] if s["kind"] != "extra"]
        inside = lambda t, b=0.0, f=0.0: any(w["a"] - b <= t <= w["b"] + f for w in hump)
        tmid = [sorted(photos[p]["sod"] for p, _ in s["items"])[len(s["items"]) // 2] for s in main]
        ph = [photos[p]["sod"] for s in main for p, _ in s["items"]]
        P = trip_log.place(L, photos, D["stacks"])
        out[day] = {"split": labeling.pairs.day_split(day), "stacks": len(main), "photos": len(ph),
                    "share_stacks": round(sum(inside(t) for t in tmid) / max(1, len(tmid)), 4),
                    "share_photos": round(sum(inside(t) for t in ph) / max(1, len(ph)), 4),
                    "share_stacks_near": round(sum(inside(t, trip_log.NEAR_BEFORE, trip_log.NEAR_AFTER) for t in tmid) / max(1, len(tmid)), 4),
                    "main_stacks": sum(1 for s in main if s["kind"] == "main"), "nowhale_stacks": sum(1 for s in main if s["kind"] == "nowhale"),
                    "clock": P["clock"],
                    # the log's times moved on purpose (synthetic clock faults, minutes): the stack share the rule would see
                    "share_shifted": {m: round(sum(any(w["a"] + 60 * m <= t <= w["b"] + 60 * m for w in hump) for t in tmid) / max(1, len(tmid)), 4)
                                      for m in (-90, -60, -30, -15, 15, 30, 60, 90)},
                    "share_near_shifted": {m: round(sum(any(w["a"] + 60 * m - trip_log.NEAR_BEFORE <= t <= w["b"] + 60 * m + trip_log.NEAR_AFTER for w in hump) for t in tmid) / max(1, len(tmid)), 4)
                                           for m in (-90, -60, -30, -15, 15, 30, 60, 90)}}
        print(day, out[day], flush=True)
    a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
