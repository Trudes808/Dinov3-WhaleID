"""Where the W3c fin cut-outs win and lose, false fins, and the found / lost sheet.

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.cutouts_w3c.analyze split --score CACHE/score_val [--json OUT]
    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.cutouts_w3c.analyze falsefins --days val [--json OUT]
    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.cutouts_w3c.analyze sheet OUT.jpg --score CACHE/score_val --arm G.W [--n 40]

split: fin queries of score.py (queries_fin.json) by (a) the region's height across / length along the back (low <
median) and its length in 2560 px (small < median), (b) centre moved more than one fin base from parts_up's centre
(fin base estimated as L / 1.21, the median L / clicked base on the clicked fins), (c) no W3c region (fallback).
falsefins: instances the view classifier calls other or fluke: how many have a W3c dorsal region (= would get a W3c
fin cut-out) against how many have a parts_up fin cut-out today.
sheet: written outside the repository only.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from . import CACHE, DAYS, day_list
from .choose import load_rule
from ..cutouts_v3.sheets import T, grid, tile, up

BASE_PER_L = 1 / 1.21


def meta(day, seed="s0"):
    f = CACHE / day / f"meta_{seed}.jsonl"
    return {json.loads(l)["key"]: json.loads(l) for l in open(f)} if f.exists() else {}


def cmd_split(a):
    q = json.loads((a.score / "queries_fin.json").read_text()); M = {}
    for d in {x["day"] for x in q}:
        M.update({(d, k): v for k, v in meta(d).items()})
    rule = load_rule()
    hl = [M[(x["day"], x["key"])]["H"] / M[(x["day"], x["key"])]["L"] for x in q if M.get((x["day"], x["key"]), {}).get("src") == "w3c"]
    ll = [M[(x["day"], x["key"])]["L"] for x in q if M.get((x["day"], x["key"]), {}).get("src") == "w3c"]
    mh, ml = float(np.median(hl)), float(np.median(ll))

    def group(x):
        m = M.get((x["day"], x["key"]))
        g = {}
        if m is None or m["src"] == "A":
            g["c fallback (no region)"] = True
            return g
        g["c has region"] = True
        g["a1 low fin (H/L < median %.2f)" % mh] = m["H"] / m["L"] < mh
        g["a1 tall fin (H/L >= median)"] = m["H"] / m["L"] >= mh
        g["a2 small (L < median %.0f px)" % ml] = m["L"] < ml
        g["a2 large (L >= median)"] = m["L"] >= ml
        g["b centre moved > 1 fin base"] = m["shift_L"] / BASE_PER_L > 1
        g["b centre moved <= 1 fin base"] = m["shift_L"] / BASE_PER_L <= 1
        return g

    names = sorted({k for x in q for k, v in group(x).items() if v} | {k for x in q for k in group(x)})
    out = {}
    for base, arms in (("G", ["G.W", "G.T", "G.X1"]), ("R0", ["R0.W", "R0.T", "R0.X1"])):
        for arm in arms:
            if not q or arm not in q[0]["hit"]:
                continue
            for nm in names:
                sel = [x for x in q if group(x).get(nm)]
                if not sel:
                    continue
                b = np.array([x["hit"][base] for x in sel]); h = np.array([x["hit"][arm] for x in sel])
                out.setdefault(arm, {})[nm] = dict(n=len(sel), base=round(float(b.mean()), 3), arm=round(float(h.mean()), 3), found=int((h & ~b).sum()), lost=int((~h & b).sum()))
    shifts = [M[(x["day"], x["key"])]["shift_L"] / BASE_PER_L for x in q if M.get((x["day"], x["key"]), {}).get("src") == "w3c"]
    out["_info"] = dict(n_queries=len(q), fallback=sum(1 for x in q if M.get((x["day"], x["key"]), {}).get("src") != "w3c"), median_HL=mh, median_L=ml,
                        shift_bases_median=float(np.median(shifts)), shift_over_1=float(np.mean(np.array(shifts) > 1)), sideA_over_L_median=float(np.median([M[(x["day"], x["key"])]["sideA_L"] for x in q if M.get((x["day"], x["key"]), {}).get("src") == "w3c"])),
                        zoom_vs_A_wide=float(np.median([M[(x["day"], x["key"])]["sideA_L"] / rule["wide"] for x in q if M.get((x["day"], x["key"]), {}).get("src") == "w3c"])))
    print(json.dumps(out, indent=1))
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))


def cmd_falsefins(a):
    out = {}
    for seed in ("s0", "s1"):
        tot = {}
        for day in day_list(a.days):
            R = json.loads((CACHE / "regions" / seed / f"{day}.json").read_text())
            for key, r in R.items():
                v = r["view"]; t = tot.setdefault(v, dict(n=0, w3c_region=0, parts_up_fin=0, both=0, main_n=0, main_w3c=0, main_up=0))
                w = r["reg"] is not None; u = bool(r["up_fin"])
                t["n"] += 1; t["w3c_region"] += w; t["parts_up_fin"] += u; t["both"] += w and u
                if key.endswith("~0"):
                    t["main_n"] += 1; t["main_w3c"] += w; t["main_up"] += u
        out[seed] = tot
    print(json.dumps(out, indent=1))
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))


def cmd_sheet(a):
    if "Dinov3-WhaleID" in str(a.out.resolve()):
        raise SystemExit("sheets never go into the repository")
    q = json.loads((a.score / "queries_fin.json").read_text()); base = a.arm.split(".")[0]
    setname = {"W": "s0_wide", "T": "s0_tight"}.get(a.arm.split(".")[1], a.extra)
    ch = [x for x in q if x["hit"][base] != x["hit"][a.arm]]
    lost = [x for x in ch if not x["hit"][a.arm]]; found = [x for x in ch if x["hit"][a.arm]]
    print(f"{a.arm} against {base}: {len(found)} found, {len(lost)} lost")
    rng = random.Random(0)
    pick = rng.sample(lost, min(len(lost), a.n - min(len(found), a.n // 3))) + rng.sample(found, min(len(found), a.n // 3))
    rows = []
    for x in pick:
        d, k = x["day"], x["key"]; m = meta(d).get(k, {})
        new = CACHE / d / setname / f"{k}.jpg"; nn_b, nn_v = x["nn"][base], x["nn"][a.arm]
        nn_new = CACHE / d / setname / f"{nn_v}.jpg"
        lab = ("LOST " if not x["hit"][a.arm] else "found ") + (f"sh{m['shift_L'] / BASE_PER_L:.1f} hl{m['H'] / m['L']:.2f}" if m.get("src") == "w3c" else "fallback")
        rows.append([tile(up(d, k, "fin"), "A query"), tile(up(d, nn_b, "fin"), "A nn " + ("ok" if x["hit"][base] else "WRONG")),
                     tile(new if new.exists() else up(d, k, "fin"), lab, (120, 255, 120) if x["hit"][a.arm] else (120, 120, 255)),
                     tile(nn_new if nn_new.exists() else up(d, nn_v, "fin"), "new nn " + ("ok" if x["hit"][a.arm] else "WRONG"))])
    sheet = grid(rows, 2); cv2.imwrite(str(a.out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85]); print(a.out, sheet.shape)
    (a.out.with_suffix(".keys.json")).write_text(json.dumps([dict(day=x["day"], key=x["key"], hit_base=x["hit"][base], hit_arm=x["hit"][a.arm]) for x in pick], indent=0))


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["split", "falsefins", "sheet"]); ap.add_argument("out", type=Path, nargs="?")
    ap.add_argument("--score", type=Path, default=CACHE / "score_val"); ap.add_argument("--json", type=Path, default=None); ap.add_argument("--days", default="val")
    ap.add_argument("--arm", default="G.W"); ap.add_argument("--n", type=int, default=40); ap.add_argument("--extra", default="s1_wide")
    a = ap.parse_args(argv)
    {"split": cmd_split, "falsefins": cmd_falsefins, "sheet": cmd_sheet}[a.cmd](a)


if __name__ == "__main__":
    main()
