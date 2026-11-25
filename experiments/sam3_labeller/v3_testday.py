"""Part S7 of rules_scene_v3.md: the single scored reading of the test day 2026-04-07 for scene v3 + the marker reader,
from the one-command runner's outputs (run_new_day.sh). Nothing here is fitted or chosen.

    .venv/bin/python experiments/sam3_labeller/v3_testday.py --run OUT_ROOT --name v3NAME [--extra NAME=SCENE_DIR ...] [--records-extra NAME=JSONL ...]
-> V3/testday/{pixtruth_in/<name>/<seq>.png, scores (pixtruth_score.py), markers_<name>.json, summary.json}
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import lab_common as L  # noqa: E402
from v3_tools import V3  # noqa: E402

PT = L.TD / "pixel_truth"
OUTD = V3 / "testday"


def to_truth_dir(scene_dir: Path, name: str) -> Path:
    o = OUTD / "pixtruth_in" / name; o.mkdir(parents=True, exist_ok=True)
    for f in sorted(scene_dir.glob("*.png")):
        lab = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        cv2.imwrite(str(o / f"{L.seq_of(f.stem)}.png"), L.TO_TRUTH[lab])
    return o


def score_markers(records: list[dict], markers_tsv: Path = PT / "markers.tsv", colours_tsv: Path = PT / "tape_colours.tsv") -> dict:
    """truth: markers_tsv (columns seq, marker = rail | bollard | land | none, refers_to = space-separated seqs) and
    colours_tsv (seq, tape_colour_by_eye) for the rail markers; seq = the photo stem's second-last '_' token."""
    truth = {r["seq"]: r for r in csv.DictReader(open(markers_tsv), delimiter="\t")}
    tc = {r["seq"]: r["tape_colour_by_eye"].split()[0] for r in csv.DictReader(open(colours_tsv), delimiter="\t")}
    rec = {L.seq_of(r["photo"]): r for r in records}
    kinds = ["rail", "bollard", "land"]
    out = {k: {"n": sum(t["marker"] == k for t in truth.values()),
               "found": sum(t["marker"] == k and rec[s]["marker"] != "none" for s, t in truth.items()),
               "right_kind": sum(t["marker"] == k and rec[s]["marker"] == k for s, t in truth.items()),
               "missed": sorted(s for s, t in truth.items() if t["marker"] == k and rec[s]["marker"] == "none"),
               "wrong_kind": sorted(s for s, t in truth.items() if t["marker"] == k and rec[s]["marker"] not in ("none", k))} for k in kinds}
    others = [s for s, t in truth.items() if t["marker"] == "none"]
    out["false_markers"] = {"n_other": len(others), "n": sum(rec[s]["marker"] != "none" for s in others),
                            "seqs": {s: rec[s]["marker"] for s in others if rec[s]["marker"] != "none"}}
    out["tape_colour"] = {"n": len(tc), "right": sum(rec[s]["tape_colour_any"] == c for s, c in tc.items()),
                          "right_on_rail_calls": sum(rec[s]["marker"] == "rail" and rec[s]["tape_colour"] == c for s, c in tc.items()),
                          "per_marker": {s: [c, rec[s]["tape_colour_any"], rec[s]["marker"]] for s, c in sorted(tc.items())}}
    ref = []
    for s, t in truth.items():
        if t["marker"] in ("rail", "bollard") and rec[s]["marker"] != "none":
            want = set(t["refers_to"].split()); got = {L.seq_of(p) for p in (rec[s].get("refers_to") or [])}
            ref.append({"seq": s, "truth_n": len(want), "ours_n": len(got), "exact": want == got,
                        "jaccard": len(want & got) / len(want | got) if (want | got) else 1.0})
    out["refers_to"] = {"n_markers_called": len(ref), "exact": sum(r["exact"] for r in ref),
                        "mean_jaccard": float(np.mean([r["jaccard"] for r in ref])) if ref else None, "rows": ref}
    return out


def nowhale_painted(per_photo: Path, names: list[str]) -> dict:
    rows = list(csv.DictReader(open(per_photo), delimiter="\t"))
    res = {}
    for n in names:
        w = [r for r in rows if r["model"] == n and r["pair"] == "whale"]
        nw = [r for r in w if int(r["truth_px"]) == 0]
        res[n] = {"no_whale_photos": len(nw), "painted_any": sum(int(r["pred_px"]) > 0 for r in nw),
                  "painted_ge50_at_truth_grid": sum(int(r["pred_px"]) >= 50 for r in nw)}
    return res


def markers_only() -> None:
    """a new day: python v3_testday.py markers-only RECORDS.jsonl MARKERS.tsv COLOURS.tsv OUT.json"""
    recs = [json.loads(l) for l in open(sys.argv[2])]
    res = score_markers(recs, Path(sys.argv[3]), Path(sys.argv[4])); Path(sys.argv[5]).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: (v if k not in ("refers_to", "tape_colour") else {x: y for x, y in v.items() if x not in ("rows", "per_marker")}) for k, v in res.items()}, indent=1))


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "markers-only":
        return markers_only()
    ap = argparse.ArgumentParser(); ap.add_argument("--run", type=Path, required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--extra", action="append", default=[], help="NAME=SCENE_DIR (another variant's scene maps on the same photos)")
    ap.add_argument("--records-extra", action="append", default=[], help="NAME=markers JSONL")
    a = ap.parse_args()
    day = a.run / "days" / L.TEST
    maps = {a.name: day / "scene3" / "scene", **{k: Path(v) for k, v in (e.split("=", 1) for e in a.extra)}}
    ours = [f"{n}={to_truth_dir(d, n)}" for n, d in maps.items()]
    ours.append(f"v2s0={L.OUT / 'pixtruth_in' / 'seg_scene_sam3v2_s0'}")
    sc = OUTD / "scores"; sc.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(L.REPO / "experiments" / "test_day" / "pixtruth_score.py"), "--root", str(L.TD), "--out", str(sc)]
    for o in ours:
        cmd += ["--ours", o]
    subprocess.run(cmd, check=True, cwd=str(L.REPO))
    summ = {"pixel_scores": str(sc / "scores.json"), "nowhale": nowhale_painted(sc / "per_photo.tsv", [*maps, "v2s0"])}
    recs = {a.name: [json.loads(l) for l in open(a.run / f"{L.TEST}_markers.jsonl")]}
    for e in a.records_extra:
        k, v = e.split("=", 1); recs[k] = [json.loads(l) for l in open(v)]
    summ["markers"] = {k: score_markers(v) for k, v in recs.items()}
    (OUTD / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({"nowhale": summ["nowhale"], "markers": {k: {kk: (vv if kk not in ("refers_to", "tape_colour") else {x: y for x, y in vv.items() if x not in ("rows", "per_marker")})
                                                                  for kk, vv in v.items()} for k, v in summ["markers"].items()}}, indent=1))


if __name__ == "__main__":
    main()
