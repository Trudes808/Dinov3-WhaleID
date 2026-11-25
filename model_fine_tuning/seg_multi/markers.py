"""Marker reader (experiments/sam3_labeller/rules_scene_v3.md S6): one record per photo of a day folder.

    flock /var/tmp/sat3737-parts-cache/gpu_train.lock env PYTHONPATH=. .venv/bin/python -m model_fine_tuning.seg_multi.markers \
        --day-dir DAY --ckpt SCENE_CKPT [--gate s6h|own|both] --cache CACHE --out DAY_markers.jsonl
        [--scene-maps DIR --s6-probs DIR]      # read cached maps (<stem>.png) instead of predicting

Per photo (time order from DAY/timestamps.json): whale present by the gate (S6h = S6 P(whale) with hysteresis 0.6 / 0.1;
own = the scene model's whale, same rule; both = whale only when both paint >= 50 px), class shares of the frame, the
marker kind (rules_labeller.md C3: rail + tape >= 4 %, widest tube >= 5.5 % of the width, largest tape component
>= 0.12 % -> rail; else land >= 25 % -> land; else deck >= 10 % and person <= 10 % -> bollard; any marker needs no whale),
the tape colour (--colour-rule: lab_centre by default, the band nearest the frame centre read by its CIE a*b* hue
angle; old = C3's HSV rule on the largest band, as frozen), its meaning, and refers_to (whale photos walking back past the marker's own run of
markers, to the previous marker or a gap > 60 s).
Scene maps: our classes at long side 1280 (0 water / sky, 1 whale, 2 person, 3 rail, 4 tape, 5 deck, 6 land).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["water_sky", "whale", "person", "rail", "tape", "deck", "land"]
WHALE_PX = 50
GAP_S = 60.0
MEANING = {"red": "female in a group, or the mother with a calf", "green": "male in a competitive group, or the escort", "blue": "calf",
           "yellow": "challenger or joiner to the group"}
MEANING_KIND = {"bollard": "the animal of the series just before is a passer-by / opportunistic animal, not part of that group "
                           "(catalogue <day>_Onn)", "land": "the boat moves to a new group"}
GATE_DEFAULT = "own"        # rules_scene_v3.md S6.1, chosen 2026-11-23 on non-test data (culled days + held-out renders)
# Tape colour rule (experiments/sam3_labeller/COLOUR2.md, 2026-11-23). "lab_centre": the tape band nearest the frame centre
# (of the components with >= 25 % of the largest's area), class = nearest prototype of the hue angle of the band's mean
# CIE a*, b*. "old": the frozen C3 rule (largest band, HSV ranges); with it the records are byte-identical to the frozen
# reader's. Prototypes = circular means of the labelled bands of both unculled days (2026-04-07, 2026-03-30), rounded.
COLOUR_RULE_DEFAULT = "lab_centre"
COLOUR_RULES = ("lab_centre", "old")
LAB_PROTO = {"red": 13.0, "yellow": 83.0, "green": 198.0, "blue": 284.0}     # degrees; boundaries at the midpoints
BAND_MIN_REL = 0.25


def hyst(p: np.ndarray, hi: float = 0.6, lo: float = 0.1) -> np.ndarray:
    n, lab = cv2.connectedComponents((p >= lo).astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool); keep[np.unique(lab[p >= hi])] = True; keep[0] = False
    return keep[lab]


def hue_class(h: float) -> str:
    if h < 20 or h >= 320:
        return "red"
    return "yellow" if h < 70 else "green" if h < 170 else "blue" if h < 260 else "other"


def facts(lab: np.ndarray) -> dict:
    H, W = lab.shape; A = H * W
    rt = ((lab == 3) | (lab == 4)).astype(np.uint8)
    tube = 2 * float(cv2.distanceTransform(rt, cv2.DIST_L2, 5).max()) / W if rt.any() else 0.0
    tape = (lab == 4).astype(np.uint8); n, cc, st, _ = cv2.connectedComponentsWithStats(tape, connectivity=8)
    big = int(st[1:, cv2.CC_STAT_AREA].argmax()) + 1 if n > 1 else 0
    return {"shares": {c: round(float((lab == k).mean()), 6) for k, c in enumerate(CLASSES)}, "rail_tape": float(rt.mean()), "tube": tube,
            "tape_big": float(st[big, cv2.CC_STAT_AREA] / A) if big else 0.0, "_big": (cc == big) if big else None}


def kind(f: dict, whale: bool) -> tuple[str, bool]:
    s = f["shares"]
    rail = f["rail_tape"] >= 0.04 and f["tube"] >= 0.055 and f["tape_big"] >= 0.0012
    if whale:
        return "none", rail
    if rail:
        return "rail", rail
    if s["land"] >= 0.25:
        return "land", rail
    if s["deck"] >= 0.10 and s["person"] <= 0.10:
        return "bollard", rail
    return "none", rail


def tape_colour(big: np.ndarray | None, photo_bgr: np.ndarray) -> str:
    if big is None:
        return "none"
    h, w = photo_bgr.shape[:2]
    m = cv2.resize(big.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    k = max(1, int(round(0.002 * w))); m = cv2.erode(m, np.ones((k, k), np.uint8)).astype(bool)
    hsv = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2HSV)
    Hh = hsv[..., 0][m].astype(np.float64) * 2; S = hsv[..., 1][m] / 255.0; V = hsv[..., 2][m] / 255.0
    col = (S >= 0.25) & (V >= 0.20)
    if col.sum() < 200:
        return "none"
    ang = np.deg2rad(Hh[col]); return hue_class((np.rad2deg(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) + 360) % 360)


def centre_band(lab: np.ndarray) -> tuple[np.ndarray | None, float | None]:
    """the tape component (8-connected, scene grid) nearest the frame centre among those with >= BAND_MIN_REL of the
    largest's area; distance = centroid to centre in frame-normalised coordinates / half the normalised diagonal."""
    tape = (lab == 4).astype(np.uint8); n, cc, st, cen = cv2.connectedComponentsWithStats(tape, connectivity=8)
    if n <= 1:
        return None, None
    H, W = lab.shape; amax = st[1:, cv2.CC_STAT_AREA].max()
    cand = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= BAND_MIN_REL * amax]
    d = {i: float(np.hypot(cen[i][0] / W - 0.5, cen[i][1] / H - 0.5) / np.hypot(0.5, 0.5)) for i in cand}
    best = min(cand, key=lambda i: (d[i], i))
    return cc == best, d[best]


def tape_colour_lab(band: np.ndarray | None, photo_bgr: np.ndarray) -> tuple[str, float | None]:
    """colour of one band: hue angle of the mean CIE a*, b* (OpenCV 8-bit Lab of the photo) over the band upsampled to the
    photo and eroded as the old rule; < 200 px -> none; class = nearest LAB_PROTO (circular)."""
    if band is None:
        return "none", None
    h, w = photo_bgr.shape[:2]
    m = cv2.resize(band.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    k = max(1, int(round(0.002 * w))); m = cv2.erode(m, np.ones((k, k), np.uint8)).astype(bool)
    if m.sum() < 200:
        return "none", None
    lab = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2LAB)[m].astype(np.float64)
    a, b = lab[:, 1].mean() - 128.0, lab[:, 2].mean() - 128.0
    hab = float(np.rad2deg(np.arctan2(b, a)) % 360)
    dist = {c: min(abs(hab - p) % 360, 360 - abs(hab - p) % 360) for c, p in LAB_PROTO.items()}
    return min(dist, key=dist.get), hab


def whale_gate(own_px: int, s6h_px: int | None, gate: str) -> bool:
    if gate == "own":
        return own_px >= WHALE_PX
    if gate == "s6h":
        return s6h_px >= WHALE_PX
    return own_px >= WHALE_PX and s6h_px >= WHALE_PX          # both must paint the whale


def read_photo(scene: np.ndarray, s6h: np.ndarray | None, photo_bgr: np.ndarray | None, gate: str,
               colour_rule: str = COLOUR_RULE_DEFAULT) -> dict:
    """one photo: scene map (ours, 1280), S6h mask at the same grid (or None for gate 'own'), the photo for the colour.
    colour_rule 'old' gives the frozen reader's record exactly; 'lab_centre' adds colour_rule, tape_hue_ab, tape_band_dist."""
    if colour_rule not in COLOUR_RULES:
        raise ValueError(colour_rule)
    lab = np.where(scene == 255, 0, scene).astype(np.uint8)
    f = facts(lab); own = int((lab == 1).sum()); s6 = int(s6h.sum()) if s6h is not None else None
    whale = whale_gate(own, s6, gate); k, railpass = kind(f, whale)
    extra = {}
    if colour_rule == "old":
        col = tape_colour(f["_big"], photo_bgr) if (photo_bgr is not None and f["tape_big"] > 0) else "none"
    else:
        col, hab, bd = "none", None, None
        if photo_bgr is not None and f["tape_big"] > 0:
            band, bd = centre_band(lab); col, hab = tape_colour_lab(band, photo_bgr)
        extra = {"colour_rule": colour_rule, "tape_hue_ab": None if hab is None else round(hab, 2),
                 "tape_band_dist": None if bd is None else round(bd, 4)}
    return {"whale": whale, "whale_px_own": own, "whale_px_s6h": s6, "gate": gate, "shares": f["shares"],
            "present": [c for c in CLASSES[1:] if f["shares"][c] >= 0.0005], "marker": k, "rail_rule_passed": bool(railpass),
            "rail_tape": round(f["rail_tape"], 5), "tube": round(f["tube"], 5), "tape_big": round(f["tape_big"], 6),
            "tape_colour": col if k == "rail" else "none", "tape_colour_any": col, "meaning": meaning(k, col), **extra}


def meaning(k: str, col: str) -> str:
    if k == "rail":
        return MEANING.get(col, f"tape colour not read ({col})")
    return MEANING_KIND.get(k, "")


def refers_to(recs: list[dict]) -> None:
    """recs in time order (dicts with 'photo', 'time', 'whale', 'marker'): fills 'refers_to' on each marker record: walking
    back past the marker's own run of consecutive markers (each step <= 60 s), then the whale photos until the previous
    marker or a gap > 60 s."""
    def gap(j):   # seconds between recs[j - 1] and recs[j]
        a_, b_ = recs[j - 1]["time"], recs[j]["time"]
        return float("inf") if a_ is None or b_ is None else b_ - a_
    for i, r in enumerate(recs):
        if r["marker"] == "none":
            continue
        out = []; j = i
        while j > 0 and recs[j - 1]["marker"] != "none" and gap(j) <= GAP_S:
            j -= 1
        while j > 0:
            prev, cur = recs[j - 1], recs[j]
            if prev["time"] is None or cur["time"] is None or cur["time"] - prev["time"] > GAP_S or prev["marker"] != "none":
                break
            if prev["whale"]:
                out.append(prev["photo"])
            j -= 1
        r["refers_to"] = out[::-1]


def write_timestamps(day_dir: Path) -> dict:
    """timestamps.json as test_scripts/bash/process_day_v2.sh writes it (EXIF DateTimeOriginal + SubSec of the RAW)."""
    import datetime
    import io
    import struct
    import exifread
    rows = {}
    for l in open(day_dir / "images" / "index.jsonl"):
        r = json.loads(l); src = r["source"]; t = {}
        try:
            if src.lower().endswith(".cr3"):
                data = open(src, "rb").read(4_000_000)
                for tag in (b"CMT1", b"CMT2"):
                    i = data.find(tag)
                    if i >= 0:
                        size = struct.unpack(">I", data[i - 4:i])[0]
                        t.update({k: str(v) for k, v in exifread.process_file(io.BytesIO(data[i + 4:i + size]), details=False).items()})
            else:
                with open(src, "rb") as fh:
                    t = {k: str(v) for k, v in exifread.process_file(fh, details=False).items()}
        except Exception:
            t = {}
        dto = t.get("EXIF DateTimeOriginal") or t.get("Image DateTimeOriginal")
        sub = (t.get("EXIF SubSecTimeOriginal") or t.get("Image SubSecTimeOriginal") or "0").strip()
        try:
            ts = datetime.datetime.strptime(dto, "%Y:%m:%d %H:%M:%S").timestamp() + float("0." + sub) if dto else None
        except Exception:
            ts = None
        rows[Path(r["image"]).stem] = {"time": ts, "datetime": dto, "camera": t.get("Image Model", "?"),
                                       "focal_mm": t.get("EXIF FocalLength") or t.get("Image FocalLength"), "folder": r["folder"]}
    (day_dir / "timestamps.json").write_text(json.dumps(rows, indent=1))
    return rows


PAL = np.array([(0, 0, 0), (255, 255, 255), (0, 0, 255), (0, 140, 255), (255, 0, 255), (255, 140, 60), (40, 90, 140)], np.uint8)   # BGR


def sheet(recs: list[dict], day_dir: Path, scene_dir: Path, out: Path, per_page: int = 24) -> list[str]:
    """every marker call: photo | class overlay, caption = time-order number, call, colour, refers_to count, meaning."""
    calls = [(k, r) for k, r in enumerate(recs) if r["marker"] != "none"]
    pages = []
    for p0 in range(0, max(1, len(calls)), per_page):
        tiles = []
        for k, r in calls[p0:p0 + per_page]:
            im = cv2.resize(cv2.imread(str(day_dir / "images" / f"{r['photo']}.jpg")), (360, 240))
            lab = cv2.imread(str(scene_dir / f"{r['photo']}.png"), cv2.IMREAD_UNCHANGED)
            ov = cv2.resize(PAL[np.clip(lab, 0, 6)], (360, 240), interpolation=cv2.INTER_NEAREST)
            t = np.hstack([im, cv2.addWeighted(im, 0.35, ov, 0.65, 0)]); cap = np.zeros((44, 720, 3), np.uint8)
            txt1 = f"#{k:03d} {r['photo'][-8:]}  {r['marker'].upper()}  tape {r['tape_colour']}  refers_to {len(r.get('refers_to') or [])}"
            cv2.putText(cap, txt1, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(cap, (r.get("meaning") or "")[:95], (6, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
            tiles.append(np.vstack([t, cap]))
        if not tiles:
            tiles = [np.zeros((284, 720, 3), np.uint8)]
            cv2.putText(tiles[0], "no marker calls", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        tiles += [np.zeros_like(tiles[0])] * ((-len(tiles)) % 3)
        img = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
        f = out.parent / f"{out.stem}_{p0 // per_page}.jpg"; cv2.imwrite(str(f), img, [cv2.IMWRITE_JPEG_QUALITY, 85]); pages.append(str(f))
    return pages


def resize_1280(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]; s = 1280 / max(h, w)
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day-dir", type=Path, required=True); ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--s6-ckpt", type=Path, default=Path(__file__).resolve().parents[2] / "experiments" / "seg_splash_s6" / "checkpoints" / "last.pt")
    ap.add_argument("--gate", default=GATE_DEFAULT, choices=["s6h", "own", "both"]); ap.add_argument("--cache", type=Path)
    ap.add_argument("--scene-maps", type=Path); ap.add_argument("--s6-probs", type=Path); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sheet", type=Path, help="write a sheet of every marker call (PATH_<page>.jpg)")
    ap.add_argument("--colour-rule", default=COLOUR_RULE_DEFAULT, choices=COLOUR_RULES,
                    help="tape colour: lab_centre (default, COLOUR2.md) or old (the frozen C3 rule; byte-identical records)")
    a = ap.parse_args()
    stems = [Path(json.loads(l)["image"]).stem for l in open(a.day_dir / "images" / "index.jsonl")]
    ts = json.loads((a.day_dir / "timestamps.json").read_text()) if (a.day_dir / "timestamps.json").exists() else write_timestamps(a.day_dir)
    scene_dir = a.scene_maps or (a.cache / "scene"); s6_dir = a.s6_probs or (a.cache / "s6prob")
    need_s6 = True        # S6 P is always cached: removal.py reads it, and records carry whale_px_s6h whatever the gate
    todo = [s for s in stems if not (scene_dir / f"{s}.png").exists() or (need_s6 and not (s6_dir / f"{s}.png").exists())]
    if todo:                                              # predict what is not cached (GPU)
        import torch
        from model_fine_tuning.common.data import pad_to_multiple, to_tensor
        from model_fine_tuning.seg_multi.model import SceneSegModel
        from model_fine_tuning.segmentation.model import WhaleSegModel
        dev = torch.device("cuda"); scene_m, _ = SceneSegModel.from_checkpoint(a.ckpt, dev)
        s6_m = WhaleSegModel.from_checkpoint(a.s6_ckpt, dev)[0] if need_s6 else None
        scene_dir.mkdir(parents=True, exist_ok=True); s6_dir.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            for s in todo:
                small = resize_1280(cv2.cvtColor(cv2.imread(str(a.day_dir / "images" / f"{s}.jpg")), cv2.COLOR_BGR2RGB))
                padded, (h, w) = pad_to_multiple(small); x = to_tensor(padded)[None].to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prob = torch.softmax(scene_m(x).float(), 1)[0, :, :h, :w].cpu().numpy()
                    if s6_m is not None:
                        p6 = torch.sigmoid(s6_m(x).float())[0, :h, :w].cpu().numpy()
                whale = hyst(prob[1]); q = prob.copy(); q[1] = -1; lab = q.argmax(0).astype(np.uint8); lab[whale] = 1
                cv2.imwrite(str(scene_dir / f"{s}.png"), lab)
                if s6_m is not None:
                    cv2.imwrite(str(s6_dir / f"{s}.png"), np.round(p6 * 255).astype(np.uint8))
    recs = []
    for s in stems:
        scene = cv2.imread(str(scene_dir / f"{s}.png"), cv2.IMREAD_UNCHANGED)
        s6h = hyst(cv2.imread(str(s6_dir / f"{s}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0) if need_s6 else None
        photo = cv2.imread(str(a.day_dir / "images" / f"{s}.jpg"))
        r = {"photo": s, "time": ts.get(s, {}).get("time"), **read_photo(scene, s6h, photo, a.gate, a.colour_rule), "refers_to": None}
        recs.append(r)
    order = sorted(range(len(recs)), key=lambda i: (recs[i]["time"] is None, recs[i]["time"] or 0, i))
    srt = [recs[i] for i in order]; refers_to(srt)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in srt:
            f.write(json.dumps(r) + "\n")
    n = {k: sum(r["marker"] == k for r in srt) for k in ("rail", "land", "bollard")}
    print(f"{len(srt)} photos, markers {n}, gate {a.gate}" + ("" if a.colour_rule == "old" else f", colour {a.colour_rule}") + f" -> {a.out}")
    if a.sheet:
        print("sheet:", sheet(srt, a.day_dir, scene_dir, a.sheet))


if __name__ == "__main__":
    main()
