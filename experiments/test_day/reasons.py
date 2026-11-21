"""Why is this photo empty? One record per photo of a day: does it hold a kept whale instance; if not, the reason the
ready-made ADE20K model gives (person / our boat or rail / land / water only) with its pixel shares, marker candidates,
and a removal (cull) candidate flag with its reason. Rules: moonshot_cache/test_days/switch_prep/rules_switch.md,
section 2 (written before the module ran on any day; thresholds from round2_rules.md: 0.5 % of the frame, land as
subject 25 %). Nothing is fitted.

    .venv/bin/python experiments/test_day/reasons.py --day-dir DAY --live-dir LIVEDAY --ade ADE_DIR [--ade-key row|seq] --out OUT.json

DAY      the arm's day folder (parts_hr/index.jsonl + safe/not_whale.json: the kept whale instances)
LIVEDAY  the live chain's day folder of the same day (its kept instances make "unsure"; may be DAY itself)
ADE_DIR  ade/<key>.png arg-max ADE20K class maps (readymade.py / readymade_day.py); --ade-key row = 4-digit row of
         DAY/images/index.jsonl (readymade_day.py), seq = the 3-digit sequence number of the 2026 file naming (readymade.py)

state     whale  = >= 1 kept instance in DAY
          unsure = none in DAY, but the live chain keeps >= 1 instance (a live-mask blob the non-whale filter scores as
                   whale: thin awash backs, whales just under the surface)
          empty  = neither
reason    (empty / unsure) first of: person >= 0.5 %; boat + vessel >= 0.5 % ("boat": rail, deck, our bow or another
          vessel; the model cannot tell them apart); land >= 25 %; else water
marker    (empty only) reason boat -> "role" (rail or deck close-up: role marker candidate; rail and bow cleat not told
          apart); land >= 25 % -> "group_change" (land filling the frame)
removal   empty -> True with the reason ("marker: role", "marker: group change", "person", "boat", "land", "water");
          unsure -> False; whale -> never
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from readymade_analyze import GROUPS, lut_for  # noqa: E402

X = 0.005            # a thing is present at >= 0.5 % of the frame (round2_rules.md section 2)
LAND_SUBJECT = 0.25  # land as subject
EVAL = {"2024-12-18", "2024-12-26", "2025-02-19", "2025-03-21"}


def kept_by_photo(d: Path) -> dict:
    nwf = d / "safe" / "not_whale.json"
    drop = set(json.loads(nwf.read_text()).get("drop", [])) if nwf.exists() else set()
    out = {}
    for l in open(d / "parts_hr" / "index.jsonl"):
        r = json.loads(l); out[r["stem"]] = sum(1 for i in r.get("instances", []) if i["key"] not in drop)
    return out


def record(kept: int, live_kept: int, shares: dict) -> dict:
    r = {"kept": kept, "live_kept": live_kept, "shares": {k: round(v, 4) for k, v in shares.items()}}
    if kept:
        r.update(state="whale", reason=None, marker=None, removal=False, removal_reason=None); return r
    r["state"] = "unsure" if live_kept else "empty"
    boat = shares["boat"] + shares["vessel"]
    r["reason"] = "person" if shares["person"] >= X else "boat" if boat >= X else "land" if shares["land"] >= LAND_SUBJECT else "water"
    r["marker"] = None
    if r["state"] == "empty":
        if r["reason"] == "boat":
            r["marker"] = "role"
        if shares["land"] >= LAND_SUBJECT:
            r["marker"] = "group_change"
    r["removal"] = r["state"] == "empty"
    r["removal_reason"] = None if not r["removal"] else ("marker: role" if r["marker"] == "role" else "marker: group change" if r["marker"] == "group_change" else r["reason"])
    if r["state"] == "unsure":
        r["note"] = "live mask holds a whale-scored blob"
    return r


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--day-dir", type=Path, required=True); p.add_argument("--live-dir", type=Path, required=True)
    p.add_argument("--ade", type=Path, required=True); p.add_argument("--ade-key", default="row", choices=["row", "seq"]); p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.day_dir.resolve().name in EVAL or a.live_dir.resolve().name in EVAL:
        raise SystemExit("evaluation days are not read")
    _, lut = lut_for("ade")
    stems = [Path(json.loads(l)["image"]).stem for l in open(a.day_dir / "images" / "index.jsonl")]
    kept, live = kept_by_photo(a.day_dir), kept_by_photo(a.live_dir)
    photos = {}
    for i, s in enumerate(stems):
        key = f"{i:04d}" if a.ade_key == "row" else s.split("_")[-2]
        lab = np.array(Image.open(a.ade / "ade" / f"{key}.png")); g = lut[lab]; n = g.size
        shares = {grp: float((g == j).sum()) / n for j, grp in enumerate(GROUPS)}
        photos[s] = record(kept.get(s, 0), live.get(s, 0), shares)
    summ = {"photos": len(photos), "state": dict(Counter(r["state"] for r in photos.values())),
            "reason": dict(Counter(f"{r['state']}:{r['reason']}" for r in photos.values() if r["state"] != "whale")),
            "marker": dict(Counter(r["marker"] for r in photos.values() if r["marker"])),
            "removal": dict(Counter(r["removal_reason"] for r in photos.values() if r["removal"]))}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"rules": {"x": X, "land_subject": LAND_SUBJECT, "order": ["person", "boat", "land", "water"], "rules_file": "switch_prep/rules_switch.md s.2"},
                                 "summary": summ, "photos": photos}, indent=1))
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
