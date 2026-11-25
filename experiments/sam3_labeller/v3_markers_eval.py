"""Marker reader checks off the test day (rules_scene_v3.md S6.1): false marker calls on the culled days (3 val + 5 train
days; every call there is false) and held-out render markers, for each whale gate; the gate is chosen by S6.1.

    .venv/bin/python experiments/sam3_labeller/v3_markers_eval.py --name NAME [--out V3/markers_eval_NAME.json]
Needs V3/pred/NAME/<day>/<stem>.png on the val + train5 days and the held-out renders, s6prob/<day>, V3/s6prob_renders.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import lab_common as L  # noqa: E402
from v3_tools import TRAIN5, V3  # noqa: E402

sys.path.insert(0, str(L.REPO))
from model_fine_tuning.seg_multi import markers as M  # noqa: E402

GATES = ["s6h", "own", "both"]


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--out", type=Path); a = ap.parse_args()
    out = {"name": a.name, "culled": {g: {"n": 0, "calls": {"rail": 0, "land": 0, "bollard": 0}, "list": []} for g in GATES}, "renders": {}}
    for day in L.VAL + TRAIN5:
        ts = json.loads((L.DAYS / day / "timestamps.json").read_text())
        for stem in L.day_stems(day):
            f = V3 / "pred" / a.name / day / f"{stem}.png"
            scene = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
            if scene is None:
                continue
            s6h = L.hyst(L.s6_prob(day, stem))
            photo = None
            for g in GATES:
                r = M.read_photo(scene, s6h, None, g); out["culled"][g]["n"] += 1
                if r["marker"] != "none":
                    if photo is None:
                        photo = cv2.imread(str(L.photo_path(day, stem)))
                    r = M.read_photo(scene, s6h, photo, g)
                    out["culled"][g]["calls"][r["marker"]] += 1
                    out["culled"][g]["list"].append({"day": day, "stem": stem, "time": ts.get(stem, {}).get("time"), **{k: r[k] for k in
                                                     ("marker", "tape_colour", "whale_px_own", "whale_px_s6h", "shares", "rail_tape", "tube", "tape_big")}})
    hold = json.loads((V3 / "renders" / "hold" / "index.json").read_text())
    for g in GATES:
        rows = []
        for x in hold:
            if not (x["shot_type"] or "").startswith("marker"):
                continue
            scene = cv2.imread(str(V3 / "pred" / a.name / x["day"] / f"{x['stem']}.png"), cv2.IMREAD_UNCHANGED)
            p6 = cv2.imread(str(V3 / "s6prob_renders" / x["day"] / f"{x['stem']}.png"), cv2.IMREAD_UNCHANGED)
            if scene is None or p6 is None:
                continue
            photo = cv2.imread(str(V3 / "renders" / "hold" / x["day"] / f"{x['stem']}.jpg"))
            r = M.read_photo(scene, L.hyst(p6.astype(np.float32) / 255.0), photo, g)
            want = {"marker_rail": "rail", "marker_deck": "bollard", "marker_land": "land"}[x["shot_type"]]
            rows.append({"run": x["day"], "id": x["stem"], "want": want, "call": r["marker"], "colour_want": (x.get("marker") or {}).get("colour"),
                         "colour": r["tape_colour"], "whale_px_own": r["whale_px_own"], "whale_px_s6h": r["whale_px_s6h"]})
        k = {w: {"n": sum(r["want"] == w for r in rows), "right": sum(r["want"] == w and r["call"] == w for r in rows)} for w in ("rail", "bollard", "land")}
        col = [r for r in rows if r["want"] == "rail"]
        out["renders"][g] = {"n": len(rows), "missed_or_wrong": sum(r["call"] != r["want"] for r in rows), "per_kind": k,
                             "colour_right_on_rail": sum(r["colour"] == r["colour_want"] for r in col), "rows": rows}
    score = {g: out["culled"][g]["n"] and (sum(out["culled"][g]["calls"].values()) + out["renders"][g]["missed_or_wrong"]) for g in GATES}
    best = min(GATES, key=lambda g: (score[g], g != "s6h"))
    out["gate_score"] = score; out["gate_chosen"] = best
    print(json.dumps({"culled": {g: {k: v for k, v in out["culled"][g].items() if k != "list"} for g in GATES},
                      "renders": {g: {k: v for k, v in out["renders"][g].items() if k != "rows"} for g in GATES}, "score": score, "chosen": best}, indent=1))
    (a.out or V3 / f"markers_eval_{a.name}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
