"""Before / after sheets of the part model on real photos: baseline parts_seg_r05_a against a weak-label run.

    PYTHONPATH=. .venv/bin/python experiments/parts_weak/sheets.py select      (baseline only; run BEFORE the new model exists)
        -> experiments/parts_weak/sheet_selection.json
    PYTHONPATH=. .venv/bin/python experiments/parts_weak/sheets.py render --name parts_seg_r05_w1
        -> SHEETS/<name>/<category>.jpg (+ .json legend); pictures of real photos, never in the repository

Instances are drawn from days never trained on: the validation days and 2025-03-18 (the click-test day), from the
baseline's per-instance numbers (hr variant), the view classifier and the by-eye codes only; 24 per category, shuffled
with seed 0:
  01_stubby_fin_on_hump     dorsal-fin views (by eye D, else classifier P(dorsal) >= 0.9 and P(fluke) <= 0.1) where the
                            baseline paints no dorsal fin or under 0.2 % of the instance, and paints head on the instance
  02_fluke_in_transition    fluke views (by eye F, else P(fluke) >= 0.9) where the baseline's fluke covers under 25 %
  03_white_pectoral         no-fluke views (P(fluke) <= 0.1 / eye D or O) with pectoral >= 0.5 % and fluke >= 0.3 %
  04_wake_at_body_end       no-fluke views with fluke >= 0.3 % and pectoral under half the fluke
  05_photo_border           instances whose hr window runs over the photo border, the baseline painting >= 20 fluke /
                            pectoral pixels on the black padding (filled up with other border windows if fewer)
Each tile: the hr window of the photo (as predict.py feeds it), the baseline's classes, the new model's classes.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from model_fine_tuning.parts_seg.labels import PALETTE  # noqa: E402
from model_fine_tuning.parts_weak import DAYS, EVAL_DAYS, SHEETS, SPLIT  # noqa: E402

SHADOW = Path("/var/tmp/sat3737-parts-cache/parts_weak/days_shadow")
SEL = Path(__file__).resolve().parent / "sheet_selection.json"
BASE = "parts_seg_r05_a"
SHEET_DAYS = SPLIT["val"] + ["2025-03-18"]
CATS = ("01_stubby_fin_on_hump", "02_fluke_in_transition", "03_white_pectoral", "04_wake_at_body_end", "05_photo_border")


def views(day):
    eye = json.loads((REPO / "experiments" / "view_class" / "view_labels.json").read_text())
    vc = json.loads((DAYS / day / "safe" / "view_class.json").read_text())["views"]
    return {k: v["code"] for k, v in eye.items() if v["day"] == day}, vc


def padding_hits(day, key, rec, W, H) -> int:
    hx, hy, side, osz = rec["hr_window"]
    m = np.asarray(Image.open(DAYS / day / BASE / "hr" / f"{key}.png"))
    a = osz / side
    yy, xx = np.mgrid[0:m.shape[0], 0:m.shape[1]]
    X, Y = xx / a + hx, yy / a + hy
    pad = (X < 0) | (Y < 0) | (X >= W) | (Y >= H)
    return int((pad & ((m == 1) | (m == 3))).sum())


def cmd_select(a) -> None:
    assert not set(SHEET_DAYS) & EVAL_DAYS
    cands = {c: [] for c in CATS}
    border_any = []
    for day in SHEET_DAYS:
        code, vc = views(day)
        inst = json.loads((DAYS / day / BASE / "instances.json").read_text())["instances"]
        drop = set(json.loads((DAYS / day / "safe" / "not_whale.json").read_text()).get("drop", []))
        sizes = {}
        for k, r in inst.items():
            if "hr" not in r or k in drop:
                continue
            c = code.get(k)
            pf, pd = vc.get(k, (None, None))
            if c in ("X",) or (c is None and pf is None):
                continue
            fluke_view = ("F" in c) if c else pf >= 0.9
            fin_view = (c is not None and "D" in c and "F" not in c and "f" not in c) or (c is None and pd >= 0.9 and pf <= 0.1)
            no_fluke = (c is not None and "F" not in c and "f" not in c) or (c is None and pf <= 0.1)
            h = r["hr"]; px = h["pixels_near"]; A = max(h["area"], 1)
            sh = {n: px[n] / A for n in px}
            row = {"day": day, "key": k}
            if fin_view and sh["dorsal_fin"] < 0.002 and px["head"] > 0:
                cands["01_stubby_fin_on_hump"].append(row)
            if fluke_view and sh["fluke"] < 0.25:
                cands["02_fluke_in_transition"].append(row)
            if no_fluke and sh["pectoral_fin"] >= 0.005 and sh["fluke"] >= 0.003:
                cands["03_white_pectoral"].append(row)
            if no_fluke and sh["fluke"] >= 0.003 and sh["pectoral_fin"] < 0.5 * sh["fluke"]:
                cands["04_wake_at_body_end"].append(row)
            hx, hy, side, osz = r["hr_window"]
            stem = k.rsplit("~", 1)[0]
            if stem not in sizes:
                sizes[stem] = Image.open(DAYS / day / "images" / f"{stem}.jpg").size
            W, H = sizes[stem]
            if hx < 0 or hy < 0 or hx + side > W or hy + side > H:
                n = padding_hits(day, k, r, W, H)
                (cands["05_photo_border"] if n >= 20 else border_any).append(dict(row, padding_fluke_pec_px=n))
    out = {"note": "chosen from the baseline only (hr variant), before the new model was scored; seed 0", "days": SHEET_DAYS,
           "candidates": {c: len(v) for c, v in cands.items()}, "picked": {}}
    for c, lst in cands.items():
        random.Random(0).shuffle(lst)
        pick = lst[:24]
        if c == "05_photo_border" and len(pick) < 24:
            random.Random(0).shuffle(border_any)
            pick += border_any[: 24 - len(pick)]
        out["picked"][c] = pick
    SEL.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["candidates"]), "->", SEL)


def overlay(rgb, cls):
    pal = np.zeros((256, 3), np.uint8)
    pal[: len(PALETTE) // 3] = np.asarray(PALETTE, np.uint8).reshape(-1, 3)
    m = (cls > 0) & (cls < 255)
    o = rgb.copy()
    o[m] = (0.4 * rgb[m] + 0.6 * pal[cls][m]).astype(np.uint8)
    return o


def tile(day, key, name, size=240):
    stem = key.rsplit("~", 1)[0]
    rgb = cv2.cvtColor(cv2.imread(str(DAYS / day / "images" / f"{stem}.jpg")), cv2.COLOR_BGR2RGB)
    out = []
    maps = []
    for root, nm in ((DAYS, BASE), (SHADOW, name)):
        maps.append(np.asarray(Image.open(root / day / nm / "hr" / f"{key}.png")))
    hx, hy, side, osz = INST[(day, key)]["hr_window"]
    M = np.array([[osz / side, 0, -hx * osz / side], [0, osz / side, -hy * osz / side]])
    win = cv2.warpAffine(rgb, M, (int(osz), int(osz)), flags=cv2.INTER_AREA if side > osz else cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    out.append(cv2.resize(win, (size, size), interpolation=cv2.INTER_AREA))
    for m in maps:
        out.append(cv2.resize(overlay(win, m), (size, size), interpolation=cv2.INTER_NEAREST))
    t = np.hstack([np.pad(x, ((0, 0), (0, 3), (0, 0)), constant_values=255) for x in out])
    return cv2.cvtColor(t, cv2.COLOR_RGB2BGR)


INST: dict = {}


def cmd_render(a) -> None:
    sel = json.loads(SEL.read_text())
    dest = SHEETS / a.name
    dest.mkdir(parents=True, exist_ok=True)
    for day in sel["days"]:
        for k, r in json.loads((DAYS / day / BASE / "instances.json").read_text())["instances"].items():
            INST[(day, k)] = r
    made = []
    for c in CATS:
        rows = sel["picked"][c]
        tiles = []
        for n, r in enumerate(rows):
            t = tile(r["day"], r["key"], a.name)
            for txt, x in ((str(n + 1), 4), ("before", 248), ("after", 491)):
                cv2.putText(t, txt, (x, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
                cv2.putText(t, txt, (x, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            tiles.append(t)
        if not tiles:
            continue
        cols = 2
        while len(tiles) % cols:
            tiles.append(np.full_like(tiles[0], 255))
        h, w = tiles[0].shape[:2]
        body = np.vstack([np.hstack([np.pad(t, ((4, 4), (6, 6), (0, 0)), constant_values=255) for t in tiles[i:i + cols]]) for i in range(0, len(tiles), cols)])
        head = np.full((34, body.shape[1], 3), 255, np.uint8)
        cv2.putText(head, f"{c}: photo | baseline parts_seg_r05_a | {a.name}   (red fluke, green dorsal fin, yellow pectoral, blue head, grey body)",
                    (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        p = dest / f"{c}.jpg"
        cv2.imwrite(str(p), np.vstack([head, body]), [cv2.IMWRITE_JPEG_QUALITY, 88])
        p.with_suffix(".json").write_text(json.dumps([dict(tile=n + 1, **r) for n, r in enumerate(rows)], indent=1))
        made.append(str(p))
    print("\n".join(made))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select").set_defaults(fn=cmd_select)
    p = sub.add_parser("render"); p.add_argument("--name", default="parts_seg_r05_w1"); p.set_defaults(fn=cmd_render)
    a = ap.parse_args()
    a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
