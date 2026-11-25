"""Sheets for scene v3 (scratch only; they show crew).

    .venv/bin/python experiments/sam3_labeller/v3_sheets.py dev --names v2s0,NAME --out SHEET.jpg       # real dev + held-out render markers
    .venv/bin/python experiments/sam3_labeller/v3_sheets.py false --eval V3/markers_eval_NAME.json --gate G --name NAME --out SHEET.jpg
    .venv/bin/python experiments/sam3_labeller/v3_sheets.py testday --records R.jsonl --scene-dir DIR --day-dir D --truth markers.tsv --out SHEET.jpg
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import lab_common as L  # noqa: E402
from v3_tools import V3, dev_list  # noqa: E402

PAL = np.array([(0, 0, 0), (255, 255, 255), (0, 0, 255), (0, 140, 255), (255, 0, 255), (255, 140, 60), (40, 90, 140)], np.uint8)
TW, TH = 300, 200


def ov(im, lab):
    o = cv2.resize(PAL[np.clip(lab, 0, 6)], (TW, TH), interpolation=cv2.INTER_NEAREST)
    return cv2.addWeighted(im, 0.35, o, 0.65, 0)


def cap(t, w=None):
    c = np.zeros((22, w or TW, 3), np.uint8); cv2.putText(c, t[:60], (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1); return c


def grid(tiles, cols, out):
    tiles += [np.zeros_like(tiles[0])] * ((-len(tiles)) % cols)
    cv2.imwrite(str(out), np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]), [cv2.IMWRITE_JPEG_QUALITY, 82])
    print("wrote", out)


def dev(a):
    names = a.names.split(","); sc = json.loads((V3 / "dev_scores.json").read_text())
    items = []
    rows = {n: {(r["day"], r["stem"]): r for r in sc[n]["real_dev"]["rows"]} for n in names}
    for d, s in dev_list():
        worst = max(rows[n].get((d, s), {}).get("whale_on_boat", 0) for n in names)
        items.append((worst, "real", d, s, L.photo_path(d, s)))
    items.sort(key=lambda x: -x[0]); sel = items[:8]
    hold = json.loads((V3 / "renders" / "hold" / "index.json").read_text())
    rend = [x for x in hold if (x["shot_type"] or "").startswith("marker")]
    sel += [(0, "render", x["day"], x["stem"], V3 / "renders" / "hold" / x["day"] / f"{x['stem']}.jpg") for x in rend[:: max(1, len(rend) // 8)][:8]]
    tiles = []
    for worst, kind, d, s, p in sel:
        im = cv2.resize(cv2.imread(str(p)), (TW, TH)); row = [np.vstack([im, cap(f"{kind} {d[-5:]} {s[-6:]}")])]
        for n in names:
            lab = cv2.imread(str(V3 / "pred" / n / d / f"{s}.png"), cv2.IMREAD_UNCHANGED)
            row.append(np.vstack([ov(im, lab), cap(f"{n}: whale {int((lab == 1).sum())} px")]))
        tiles.append(np.hstack(row))
    grid(tiles, 2, a.out)


def false(a):
    ev = json.loads(Path(a.eval).read_text()); calls = ev["culled"][a.gate]["list"]; tiles = []
    for c in calls:
        im = cv2.resize(cv2.imread(str(L.photo_path(c["day"], c["stem"]))), (TW, TH))
        lab = cv2.imread(str(V3 / "pred" / a.name / c["day"] / f"{c['stem']}.png"), cv2.IMREAD_UNCHANGED)
        tiles.append(np.hstack([np.vstack([im, cap(f"{c['day']} {c['stem'][-6:]}")]),
                                np.vstack([ov(im, lab), cap(f"{c['marker']} tape {c['tape_colour']} wh {c['whale_px_own']}/{c['whale_px_s6h']}")])]))
    if not tiles:
        t = np.zeros((TH + 22, 2 * TW, 3), np.uint8); cv2.putText(t, "no marker calls on the culled days", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        tiles = [t]
    grid(tiles, 3, a.out)


def testday(a):
    recs = [json.loads(l) for l in open(a.records)]; by = {r["photo"]: r for r in recs}
    truth = {r["seq"]: r for r in csv.DictReader(open(a.truth), delimiter="\t")}
    stems = {L.seq_of(r["photo"]): r["photo"] for r in recs}
    seqs = sorted(set(s for s, t in truth.items() if t["marker"] != "none") | {L.seq_of(r["photo"]) for r in recs if r["marker"] != "none"})
    tiles = []
    for q in seqs:
        r = by[stems[q]]; t = truth[q]
        im = cv2.resize(cv2.imread(str(Path(a.day_dir) / "images" / f"{r['photo']}.jpg")), (TW, TH))
        lab = cv2.imread(str(Path(a.scene_dir) / f"{r['photo']}.png"), cv2.IMREAD_UNCHANGED)
        pres = ",".join(c[:4] for c in r["present"])
        tiles.append(np.hstack([np.vstack([im, cap(f"{q} truth {t['marker']} {t['tape_colour']}")]),
                                np.vstack([ov(im, lab), cap(f"call {r['marker']} {r['tape_colour']} | {pres}")])]))
    grid(tiles, 3, a.out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["dev", "false", "testday"]); ap.add_argument("--names"); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--eval"); ap.add_argument("--gate"); ap.add_argument("--name"); ap.add_argument("--records"); ap.add_argument("--scene-dir")
    ap.add_argument("--day-dir"); ap.add_argument("--truth")
    a = ap.parse_args(); {"dev": dev, "false": false, "testday": testday}[a.mode](a)


if __name__ == "__main__":
    main()
