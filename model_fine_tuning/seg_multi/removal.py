"""Removal rule v2 (2026-11-22): which photos of an unculled day to PROPOSE for removal (the researcher's cull), and
which to show apart as "faint whale?". Rules in words: experiments/removal_v2/README.md. Nothing is removed by this:
the page proposes, the researcher confirms.

    # after model_fine_tuning/seg_multi/run_new_day.sh (its OUT_ROOT / day folder / markers JSONL):
    .venv/bin/python -m model_fine_tuning.seg_multi.removal --day-dir OUT_ROOT/days/DAY --markers OUT_ROOT/DAY_markers.jsonl \
        --cache OUT_ROOT/days/DAY/scene3 --sam3-out OUT_ROOT/days/DAY/removal_sam3 --out OUT_ROOT/DAY_removal.jsonl \
        [--kept-day LIVE_DAY_FOLDER] [--old-kept-day OLD_LIVE_DAY_FOLDER] [--no-sam3-run | --sam3-cpu]

INPUTS (the interface; all per photo, keyed by the photo stem of DAY/images/index.jsonl):
  --markers   seg_multi/markers.py records: photo, time, whale (S6h gate), whale_px_s6h, shares (scene model: person,
              land, ...), marker (rail | land | bollard | none). Without it (stand-alone): the gate and shares are read
              from the cache maps below and no photo is a marker.
  --cache     CACHE/s6prob/<stem>.png  S6 P(whale) x 255 at long side 1280; CACHE/scene/<stem>.png scene map (0 water/sky,
              1 whale, 2 person, 3 rail, 4 tape, 5 deck, 6 land) at the same size. Both written by markers.py.
  --sam3-out  SAM 3 raw outputs of experiments/removal_v2/prompts_trace.json, <stem>.npz, on the photos without a
              whale (experiments/sam3_compare/run_sam3.py). Missing ones are computed here (GPU, under the shared
              lock, with the Underwater-WhaleID venv used read-only) unless --no-sam3-run; a photo still without
              them is never proposed ("faint whale?": SAM 3 not read).
  --kept-day      optional: the live chain's day folder (parts_hr + safe/not_whale.json): a kept instance = a whale.
  --old-kept-day  optional: the old live chain (live mask seg_r04_ft4): a kept instance there = a trace reading.
OUTPUT: the markers records (or a bare record per photo) with "removal_v2": {proposed, reason, faint, seen, why,
readings}; reason marker | crew | land | water; faint = no whale found but >= 1 reading sees a trace (never proposed).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
EVAL = {"2024-12-18", "2024-12-26", "2025-02-19", "2025-03-21"}
SAM3_PY = Path("/home/sat3737/NOAA_whales/Underwater-WhaleID/.venv/bin/python")
SAM3_RUN = REPO / "experiments" / "sam3_compare" / "run_sam3.py"
SAM3_RUN_CPU = REPO / "experiments" / "removal_v2" / "run_sam3_cpu.py"      # the same on the CPU (float32, ~15 s a photo, no GPU lock)
PROMPTS = REPO / "experiments" / "removal_v2" / "prompts_trace.json"
LOCK = "/var/tmp/sat3737-parts-cache/gpu_train.lock"
SCENE = ["water_sky", "whale", "person", "rail", "tape", "deck", "land"]

# ---- the rule (frozen with removal_FREEZE.json) ------------------------------------------------------------------------
WHALE_PX = 50                 # S6h hysteresis 0.6 / 0.1 paints >= 50 px at 1280: a whale (as markers.py)
PERSON, LAND = 0.005, 0.25    # reason: person >= 0.5 % of the frame -> crew; land >= 25 % -> land; else water (as reasons.py)
TRACE = {                     # a reading "sees a trace of a whale" when ... (set on the development day 2026-04-07: the lowest
                              # level silent on all 30 of its photos with nothing of a whale, markers included; foam: on its 7
                              # non-marker ones, a land marker's coastal surf reaches 2.4 %)
    "S6h low": ("s6_px25", 20),                    # S6 P(whale) >= 0.25 on >= 20 px (none: <= 5 px)
    "old live mask": ("live_kept", 1),             # the old live chain kept a whale-scored instance (only where it was run)
    "SAM 3 whale": ("sam3_whale", 0.10),           # "whale" / "humpback whale" / "marine mammal" instance score (none: 0)
    "SAM 3 fluke / fin": ("sam3_part", 0.20),      # "whale tail" / "whale fluke" / "dorsal fin" (none: <= 0.15)
    "SAM 3 blow / splash": ("sam3_trace", 0.20),   # "whale blow" / "whale spout" / "whale splash" / "splash" (none: <= 0.16)
    "SAM 3 under water": ("sam3_under", 0.40),     # "whale under water" / "submerged whale" (none: <= 0.36)
    "white water": ("foam", 0.010),                # foam outside crew / boat / land >= 1 % of the frame (non-marker none: <= 0.7 %)
    "glow under water": ("glow_big", 0.0002),      # a turquoise patch under the surface (a white flipper / belly) >= 0.02 % (none: <= 0.001 %)
}
FAMILIES = {"whale": ["whale", "humpback whale", "marine mammal"], "part": ["whale tail", "whale fluke", "dorsal fin"],
            "trace": ["whale blow", "whale spout", "whale splash", "splash"], "under": ["whale under water", "submerged whale"],
            "person": ["person"], "land": ["land", "mountain"], "white": ["white water", "foam"]}


def hyst(p, hi=0.6, lo=0.1):
    n, lab = cv2.connectedComponents((p >= lo).astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool); keep[np.unique(lab[p >= hi])] = True; keep[0] = False
    return keep[lab]


def s6_read(p: np.ndarray) -> dict:
    return {"s6_gate_px": int(hyst(p).sum()), "s6_max": round(float(p.max()), 4), "s6_px10": int((p >= 0.1).sum()),
            "s6_px25": int((p >= 0.25).sum()), "s6_px50": int((p >= 0.5).sum())}


def water_read(img_bgr: np.ndarray, scene: np.ndarray | None) -> dict:
    """foam = pixels much brighter than their 41-px neighbourhood (top-hat >= 45) and pale (S <= 90 / 255), outside
    person / rail / tape / deck / land (dilated 15 px), closed 7 px; glow = cyan hue (OpenCV H 78-100), S >= 90, V >= 110,
    brighter than the neighbourhood (V top-hat >= 20), same mask, opened 3 px. Fractions of the frame at long side 1280."""
    h, w = img_bgr.shape[:2]; s = 1280 / max(h, w)
    im = cv2.resize(img_bgr, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY); hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    free = np.ones(g.shape, bool)
    if scene is not None:
        sc = cv2.resize(scene, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
        free = cv2.dilate(np.isin(sc, [2, 3, 4, 5, 6]).astype(np.uint8), np.ones((15, 15), np.uint8)) == 0
    foam = (th >= 45) & (hsv[..., 1] <= 90) & free
    foam = cv2.morphologyEx(foam.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(foam, connectivity=8)
    A = foam.size; big = int(st[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0
    hh, ss, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    vth = cv2.morphologyEx(vv, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    glow = (hh >= 78) & (hh <= 100) & (ss >= 90) & (vv >= 110) & (vth >= 20) & free
    glow = cv2.morphologyEx(glow.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n2, _, st2, _ = cv2.connectedComponentsWithStats(glow, connectivity=8)
    big2 = int(st2[1:, cv2.CC_STAT_AREA].max()) if n2 > 1 else 0
    return {"foam": round(float(foam.mean()), 5), "foam_big": round(big / A, 5), "glow": round(float(glow.mean()), 6), "glow_big": round(big2 / A, 6)}


def sam3_read(npz: Path, order: list[str], fams: dict = FAMILIES) -> dict:
    z = np.load(npz); idx = {q: i for i, q in enumerate(order)}; out = {}
    for fam, qs in fams.items():
        best = 0.0
        for q in qs:
            if q in idx:
                s = z[f"s{idx[q]}"]; best = max(best, float(s.max()) if s.size else 0.0)
        out[f"sam3_{fam}"] = round(best, 4)
    return out


def kept_by_photo(d: Path) -> dict:
    nwf = d / "safe" / "not_whale.json"
    drop = set(json.loads(nwf.read_text()).get("drop", [])) if nwf.exists() else set()
    out = {}
    for l in open(d / "parts_hr" / "index.jsonl"):
        r = json.loads(l); out[r["stem"]] = sum(1 for i in r.get("instances", []) if i["key"] not in drop)
    return out


def decide(whale: bool, marker: str | None, shares: dict, rd: dict, sam3_read_ok: bool) -> dict:
    """the rule: a marker -> proposed (marker); a whale -> nothing; else every reading must see no trace for a proposal
    (crew / land / water by the shares); one or more see a trace (or SAM 3 could not be read) -> "faint whale?"."""
    if marker in ("rail", "land", "bollard"):
        seen = [k for k, (f, t) in TRACE.items() if rd.get(f) is not None and rd[f] >= t]
        return {"proposed": True, "reason": "marker", "faint": False, "seen": seen, "why": f"a {marker} marker photo (no whale)"}
    if whale:
        return {"proposed": False, "reason": None, "faint": False, "seen": [], "why": "whale"}
    seen = [k for k, (f, t) in TRACE.items() if rd.get(f) is not None and rd[f] >= t]
    if not sam3_read_ok:
        seen.append("SAM 3 not read")
    if seen:
        return {"proposed": False, "reason": None, "faint": True, "seen": seen,
                "why": "no whale found, but a reading sees a trace of one: " + ", ".join(seen)}
    reason = "crew" if shares.get("person", 0.0) >= PERSON else "land" if shares.get("land", 0.0) >= LAND else "water"
    words = {"crew": "crew only, no whale", "land": "land only, no whale", "water": "water only, no whale"}[reason]
    return {"proposed": True, "reason": reason, "faint": False, "seen": [], "why": words + " (rule v2: every reading sees no trace)"}


def run_sam3(todo: list[tuple[str, Path]], out_dir: Path, cpu: bool = False) -> None:
    lst = out_dir.parent / f"_todo_{out_dir.name}.tsv"; out_dir.mkdir(parents=True, exist_ok=True)
    lst.write_text("".join(f"{s}\t{p}\n" for s, p in todo))
    args = ["--list", str(lst), "--prompts", str(PROMPTS), "--set", out_dir.name, "--out", str(out_dir.parent)]
    cmd = [str(SAM3_PY), str(SAM3_RUN_CPU), *args, "--budget", "36000"] if cpu else ["flock", LOCK, str(SAM3_PY), str(SAM3_RUN), *args]
    env = {**__import__("os").environ, "CUDA_VISIBLE_DEVICES": ""} if cpu else None
    for _ in range(20):                                    # resumable chunks (the script stops after its time budget)
        subprocess.run(cmd, check=True, cwd=str(REPO), env=env)
        if all((out_dir / f"{s}.npz").exists() for s, _ in todo):
            break


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day-dir", type=Path, required=True); ap.add_argument("--markers", type=Path); ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--sam3-out", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--kept-day", type=Path); ap.add_argument("--old-kept-day", type=Path); ap.add_argument("--no-sam3-run", action="store_true")
    ap.add_argument("--sam3-cpu", action="store_true", help="run SAM 3 on the CPU (no GPU lock; ~15 s a photo)")
    ap.add_argument("--day", help="ISO day (default: the day folder's name)")
    a = ap.parse_args()
    day = a.day or a.day_dir.resolve().name
    if day in EVAL:
        sys.exit("evaluation day: refused")
    stems = [Path(json.loads(l)["image"]).stem for l in open(a.day_dir / "images" / "index.jsonl")]
    recs = {}
    if a.markers:
        for l in open(a.markers):
            v = json.loads(l); recs[v["photo"]] = v
    kept = kept_by_photo(a.kept_day) if a.kept_day else None
    old = kept_by_photo(a.old_kept_day) if a.old_kept_day else None
    s6dir, scdir = a.cache / "s6prob", a.cache / "scene"
    rows = []
    for s in stems:
        v = recs.get(s) or {"photo": s}
        p = cv2.imread(str(s6dir / f"{s}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
        rd = s6_read(p)
        sc = cv2.imread(str(scdir / f"{s}.png"), cv2.IMREAD_UNCHANGED) if (scdir / f"{s}.png").exists() else None
        if sc is not None:
            sc = np.where(sc == 255, 0, sc)
        shares = v.get("shares") or ({c: float((sc == k).mean()) for k, c in enumerate(SCENE)} if sc is not None else {})
        whale = rd["s6_gate_px"] >= WHALE_PX or (kept is not None and kept.get(s, 0) >= 1)
        if old is not None:
            rd["live_kept"] = old.get(s, 0)
        if kept is not None:
            rd["kept"] = kept.get(s, 0)
        rows.append((s, v, rd, sc, shares, whale))
    cand = [(s, a.day_dir / "images" / f"{s}.jpg") for s, v, rd, sc, sh, wh in rows if not wh]
    todo = [(s, p) for s, p in cand if not (a.sam3_out / f"{s}.npz").exists()]
    if todo and not a.no_sam3_run:
        print(f"SAM 3 on {len(todo)} photos without a whale ...", flush=True); run_sam3(todo, a.sam3_out, a.sam3_cpu)
    order = json.loads((a.sam3_out / "_prompts.json").read_text()) if (a.sam3_out / "_prompts.json").exists() else None
    a.out.parent.mkdir(parents=True, exist_ok=True); n = {"proposed": 0, "faint": 0}
    with open(a.out, "w") as f:
        for s, v, rd, sc, shares, whale in rows:
            ok = True
            if not whale and v.get("marker") not in ("rail", "land", "bollard"):
                rd.update(water_read(cv2.imread(str(a.day_dir / "images" / f"{s}.jpg")), sc))
                npz = a.sam3_out / f"{s}.npz"
                ok = order is not None and npz.exists()
                if ok:
                    rd.update(sam3_read(npz, order))
            d = decide(whale, v.get("marker"), shares, rd, ok); d["readings"] = rd
            n["proposed"] += d["proposed"]; n["faint"] += d["faint"]
            f.write(json.dumps({**v, "removal_v2": d}) + "\n")
    print(f"{day}: {len(rows)} photos, {n['proposed']} proposed, {n['faint']} faint whale? -> {a.out}")


if __name__ == "__main__":
    main()
