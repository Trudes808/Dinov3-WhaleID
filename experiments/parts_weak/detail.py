"""Breakdowns and test-set intervals, baseline against a weak-label run (validation days; clicks of 2025-03-18 only).

    PYTHONPATH=. .venv/bin/python experiments/parts_weak/detail.py --arm parts_seg_r05_w1 --out experiments/parts_weak/detail_w1.json

  three-way call (two-part rule re-chosen on the training days per model, as compare.py) by truth view, whale size
    (long side of the box, 2560-px photo) and one / several whale instances in the photo, by eye and by classifier;
  per-instance yes / no measures with a paired bootstrap over photos (2000 resamples, 95 %): dorsal views with no fin
    pixel, fluke views with no fluke pixel, cut-out needs (fin region; fluke >= 25 % of the instance);
  clicks of 2025-03-18: fin tip in the largest fin region, fluke notch / tips in the fluke (missing region = miss), and
    what the model paints on the tip part of each clicked fin (tip to 0.3 of the way to the two bases, as the training
    labels of 2025-01-02): mostly fin / head / body, and fins with no fin pixel near the instance at all.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import compare as CP  # noqa: E402
from model_fine_tuning.common import geometry, paths  # noqa: E402
from model_fine_tuning.parts_seg import evaluate_real as ER  # noqa: E402

CLICK_DAY = "2025-03-18"


def boot(base: dict, arm: dict) -> dict:
    return CP.paired(base, arm)


def three_way_breakdown(rows_b, rows_a, variant, rule_b, rule_a) -> dict:
    out = {}
    keyfns = {"view": lambda r, t: r[t], "size": lambda r, t: ER.size_bin(r), "whales_in_photo": lambda r, t: "1" if r["n_inst"] <= 1 else ">=2"}
    for truth in ("t_eye", "t_vc"):
        cb = CP.correct(rows_b, variant, *rule_b, truth)
        ca = CP.correct(rows_a, variant, *rule_a, truth)
        by_key = {r["key"]: r for r in rows_b}
        for kname, fn in keyfns.items():
            groups = {}
            for k in cb:
                if k in ca:
                    groups.setdefault(fn(by_key[k], truth), []).append(k)
            out.setdefault(truth, {})[kname] = {g: boot({k: cb[k] for k in ks}, {k: ca[k] for k in ks}) for g, ks in sorted(groups.items())}
    return out


def instance_measures(rows, variant, parts_up) -> dict:
    """key -> (stem, outcome) per measure, validation days."""
    m = {"dorsal_view_no_fin_px": {}, "fluke_view_no_fluke_px": {}, "cutout_fin_region": {}, "cutout_fluke_quarter": {}}
    for r in rows:
        if r["split"] != "val" or variant not in r["pred"] or r["not_whale"]:
            continue
        p = r["pred"][variant]
        v = CP.view_of(r)
        if v == "dorsal":
            m["dorsal_view_no_fin_px"][r["key"]] = (r["stem"], int(p["pixels_near"]["dorsal_fin"] == 0))
        if v == "fluke":
            m["fluke_view_no_fluke_px"][r["key"]] = (r["stem"], int(p["pixels_near"]["fluke"] == 0))
        if variant == "hr" and r["vc"]:
            if r["vc"][1] >= 0.5 and "fin" in parts_up.get(r["key"], []):
                m["cutout_fin_region"][r["key"]] = (r["stem"], int("fin" in p))
            if r["vc"][0] >= 0.5:
                m["cutout_fluke_quarter"][r["key"]] = (r["stem"], int(p["pixels_near"]["fluke"] >= 0.25 * max(p["area"], 1)))
    return m


def click_outcomes(name: str, table: dict) -> dict:
    """per clicked key on 2025-03-18 (hr variant): outcomes and the classes on the fin's tip part."""
    root = CP.root_of(name, CLICK_DAY)
    ER.DAYS = root
    ER.CLICK_DAYS = (CLICK_DAY,)
    _, recs = ER.click_metrics(name, "hr", table)
    inst = json.loads((root / CLICK_DAY / name / "instances.json").read_text())["instances"]
    labs = json.loads((ER.LAB / CLICK_DAY / "pair_marks.json").read_text())
    labs = labs["labels"] if isinstance(labs, dict) else labs
    kps = {}
    for l in labs:
        if l.get("verdict") == "skip":
            continue
        for sd in ("a", "b"):
            kp = (l.get("keypoints") or {}).get(sd) or {}
            if kp and l.get(f"key_{sd}"):
                kps.setdefault(l[f"key_{sd}"], {}).update(kp)
    out = {}
    for rec in recs:
        key = rec["key"]
        stem = key.rsplit("~", 1)[0]
        o = {}
        if "tip_to_fin_region_px" in rec:
            d = rec["tip_to_fin_region_px"]
            o["fin_tip_in_region"] = int(d is not None and d <= ER.NEAR_PX)
        for n in ("fluke_notch", "fluke_tip_left", "fluke_tip_right"):
            if f"{n}_to_fluke_px" in rec:
                d = rec[f"{n}_to_fluke_px"]
                o[f"{n}_in_fluke"] = int(d is not None and d <= ER.NEAR_PX)
        kp = kps.get(key, {})
        r = inst.get(key, {})
        if all(n in kp for n in ("dorsal_fin_tip", "dorsal_fin_front", "dorsal_fin_back")) and "hr_window" in r:
            W, H = Image.open(root / CLICK_DAY / "images" / f"{stem}.jpg").size
            hx, hy, side, osz = r["hr_window"]
            a = osz / side
            P = {n: np.array([(kp[n]["x"] * W - hx) * a, (kp[n]["y"] * H - hy) * a]) for n in ("dorsal_fin_tip", "dorsal_fin_front", "dorsal_fin_back")}
            t = P["dorsal_fin_tip"]
            tri = np.stack([t, t + 0.3 * (P["dorsal_fin_front"] - t), t + 0.3 * (P["dorsal_fin_back"] - t)])
            cm = np.asarray(Image.open(root / CLICK_DAY / name / "hr" / f"{key}.png"))
            sh = np.zeros(cm.shape, np.uint8)
            cv2.fillPoly(sh, [np.round(tri).astype(np.int32)], 1)
            cv2.circle(sh, (int(round(t[0])), int(round(t[1]))), 2, 1, -1)
            c = np.bincount(cm[sh > 0], minlength=256)[:6]
            if c.sum() > 0:
                lab = ("background", "fluke", "fin", "pectoral", "head", "body")[int(np.argmax(c))]
                o["tip_part_mostly"] = lab
                o["tip_part_mostly_fin"] = int(lab == "fin")
                o["tip_part_mostly_head"] = int(lab == "head")
                o["tip_part_mostly_body"] = int(lab == "body")
            o["no_fin_px_near_instance"] = int(r.get("hr", {}).get("pixels_near", {}).get("dorsal_fin", 0) == 0)
        out[key] = (stem, o)
    ER.DAYS = CP.DAYS_ALL
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="parts_seg_r05_a")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    split = json.loads((paths.SPLITS_DIR / "days.json").read_text())
    days = split["train"] + split["val"]
    rows = {n: CP.rows_of(n, days) for n in (a.base, a.arm)}
    parts_up = {}
    for day in days:
        for l in open(CP.DAYS_ALL / day / "parts_up" / "index.jsonl"):
            r = json.loads(l)
            for i in r.get("instances", []):
                parts_up[i["key"]] = i.get("parts", [])
    t = geometry.load_folded_table(paths.FOLDED_TABLE)
    table = {"x_min_m": float(t["x_min_m"]), "length_m": float(t["length_m"])}
    res = {"base": a.base, "arm": a.arm, "three_way_breakdown": {}, "instance_measures": {}, "clicks_2025_03_18_hr": {}}
    for variant in ("hr", "full"):
        rb, ra = CP.choose(rows[a.base], variant, "two_part"), CP.choose(rows[a.arm], variant, "two_part")
        res["three_way_breakdown"][variant] = three_way_breakdown(rows[a.base], rows[a.arm], variant, rb, ra)
        mb, ma = instance_measures(rows[a.base], variant, parts_up), instance_measures(rows[a.arm], variant, parts_up)
        res["instance_measures"][variant] = {k: boot(mb[k], ma[k]) for k in mb if mb[k]}
    cb, ca = click_outcomes(a.base, table), click_outcomes(a.arm, table)
    names = sorted({n for _, o in cb.values() for n in o if n != "tip_part_mostly"})
    for n in names:
        b = {k: (s, o[n]) for k, (s, o) in cb.items() if n in o}
        x = {k: (s, o[n]) for k, (s, o) in ca.items() if n in o}
        res["clicks_2025_03_18_hr"][n] = boot(b, x)
    from collections import Counter
    res["clicks_2025_03_18_hr"]["tip_part_classes"] = {a.base: dict(Counter(o.get("tip_part_mostly") for _, o in cb.values() if "tip_part_mostly" in o)),
                                                       a.arm: dict(Counter(o.get("tip_part_mostly") for _, o in ca.values() if "tip_part_mostly" in o))}
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["instance_measures"], indent=0)[:3000])
    print(json.dumps(res["clicks_2025_03_18_hr"], indent=0))
    for variant in ("hr", "full"):
        for truth in ("t_eye", "t_vc"):
            for kname, g in res["three_way_breakdown"][variant][truth].items():
                print(variant, truth, kname, {k: (v.get("n"), v.get("base_acc"), v.get("arm_acc"), v.get("ci95")) for k, v in g.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
