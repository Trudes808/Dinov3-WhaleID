"""Score the part segmentation on real photos (training and validation days only; never the evaluation days).

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.parts_seg.evaluate_real --out experiments/parts_seg_r05_a/real_metrics.json \
        [--name parts_seg_r05_a] [--sheets SCRATCH_DIR]

a. instance level, three-way fluke / dorsal fin / other, against
   (1) the by-eye labels experiments/view_class/view_labels.json (confident codes F, D, O, FD; marginal = any lower-case
       code, scored separately) and
   (2) the view classifier's confident answers DAY/safe/view_class.json (P >= 0.9 yes, <= 0.1 no);
   baseline on the same instances: the cut-out flags of parts_hr/index.jsonl ("fluke" / "fin").
   The decision rule (fluke share >= tf, else fin share >= td, else other; shares of the instance's pixels) is chosen
   on the TRAINING days (by-eye + classifier, equal weight) and reported on the validation days separately.
b. points against the researcher's clicks (Whale-Review-Web data/2025-01-02 and 2025-03-18 pair_marks.json, read only).
c. breakdowns of the instance-level errors (truth, size, one vs several instances, catalogue group type).
d. contact sheets (optional, --sheets: pictures of real photos, never inside the repository).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..common import paths
from .labels import CLASSES, PALETTE
from .predict import EVAL_DAYS, day_list

ROOT = paths.REPO_ROOT
DAYS = paths.EXPERIMENTS_DIR / "days_all"
LAB = ROOT.parent / "Whale-Review-Web" / "data"
CLICK_DAYS = ("2025-01-02", "2025-03-18")
THREE = ("fluke", "dorsal", "other")
NEAR_PX = 8.0            # "inside or within a few pixels": 8 px of the 2560-px image (4 px of the 1280 map)


# ---------------------------------------------------------------------------------------------------------- loading
def load_all(name: str, days: list[str]) -> list[dict]:
    split = json.loads((paths.SPLITS_DIR / "days.json").read_text())
    cat = json.loads((paths.SPLITS_DIR / "labels_all_days.json").read_text())["days"]
    eye = json.loads((ROOT / "experiments" / "view_class" / "view_labels.json").read_text())
    rows = []
    for day in days:
        d = DAYS / day
        pred = json.loads((d / name / "instances.json").read_text())["instances"]
        index = [json.loads(l) for l in open(d / "parts_hr" / "index.jsonl") if l.strip()]
        vc = json.loads((d / "safe" / "view_class.json").read_text())["views"] if (d / "safe" / "view_class.json").exists() else {}
        drop = set(json.loads((d / "safe" / "not_whale.json").read_text()).get("drop", [])) if (d / "safe" / "not_whale.json").exists() else set()
        photos, groups = cat.get(day, {}).get("photos", {}), cat.get(day, {}).get("groups", {})
        for rec in index:
            n_inst = len([i for i in rec.get("instances", []) if i["key"] not in drop])
            animal = photos.get(rec["stem"].split("__")[-1])
            g = animal.split("_")[-1][:3] if animal else None
            gtype = (groups.get(g) or ("O" if g and g.startswith("O") else "?")) if animal else "no catalogue id"
            for inst in rec.get("instances", []):
                key = inst["key"]
                if key not in pred:
                    continue
                e = eye.get(key)
                rows.append({"day": day, "split": "train" if day in split["train"] else "val", "key": key, "stem": rec["stem"],
                             "bbox": inst.get("bbox_2560"), "area": inst.get("area"), "base_parts": inst.get("parts", []),
                             "not_whale": key in drop, "n_inst": n_inst, "gtype": gtype, "eye": e["code"] if e else None,
                             "eye_round": e["round"] if e else None, "vc": vc.get(key), "pred": pred[key]})
    return rows


def eye_three(code: str | None):
    """confident three-way truth from a by-eye code, or None (marginal / X / none)."""
    if not code or code == "X" or any(c.islower() for c in code):
        return None
    return "fluke" if "F" in code else "dorsal" if "D" in code else "other"


def eye_marginal(code: str | None) -> bool:
    return bool(code) and code != "X" and any(c.islower() for c in code)


def vc_three(v):
    if not v:
        return None
    pf, pd = v
    if pf >= 0.9:
        return "fluke"
    if pf <= 0.1:
        if pd >= 0.9:
            return "dorsal"
        if pd <= 0.1:
            return "other"
    return None


def base_three(parts) -> str:
    return "fluke" if "fluke" in parts else "dorsal" if "fin" in parts else "other"


def shares(p: dict) -> tuple[float, float]:
    px = p.get("pixels_near", p["pixels"])
    tot = max(p["covered"], 1)
    return px["fluke"] / tot, px["dorsal_fin"] / tot


def rule_three(p: dict, tf, td: float) -> str:
    """tf = fluke share minimum, or (minimum, r): fluke also needs fluke share >= r x pectoral-fin share (raised
    pectoral fins come out as a mixture of the two classes)."""
    tf, rp = tf if isinstance(tf, (tuple, list)) else (tf, 0.0)
    f, d = shares(p)
    px = p.get("pixels_near", p["pixels"])
    pec = px["pectoral_fin"] / max(p["covered"], 1)
    return "fluke" if (f >= tf and f >= rp * pec) else "dorsal" if d >= td else "other"


# ---------------------------------------------------------------------------------------------------------- scoring
def confusion(pairs) -> dict:
    m = np.zeros((3, 3), int)
    for t, p in pairs:
        m[THREE.index(t), THREE.index(p)] += 1
    n = int(m.sum())
    return {"n": n, "acc": float(np.trace(m) / n) if n else None, "matrix_rows_true_fluke_dorsal_other": m.tolist()}


def score_set(rows, variant, tf, td, truth) -> dict:
    sel = [r for r in rows if r.get(truth) is not None and variant in r["pred"]]
    return {"ours": confusion([(r[truth], rule_three(r["pred"][variant], tf, td)) for r in sel]),
            "baseline": confusion([(r[truth], base_three(r["base_parts"])) for r in sel])}


def choose_rule(rows, variant) -> tuple[float, float, dict]:
    grid = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3]
    best, table = None, {}
    tr = [r for r in rows if r["split"] == "train" and variant in r["pred"]]
    eye = [r for r in tr if r["t_eye"]]
    vc = [r for r in tr if r["t_vc"]]
    for tf in [(g, rp) for g in grid for rp in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)]:
        for td in grid:
            a1 = np.mean([r["t_eye"] == rule_three(r["pred"][variant], tf, td) for r in eye]) if eye else 0
            a2 = np.mean([r["t_vc"] == rule_three(r["pred"][variant], tf, td) for r in vc]) if vc else 0
            sc = 0.5 * (a1 + a2)
            table[f"{tf[0]}x{tf[1]}/{td}"] = [round(float(a1), 4), round(float(a2), 4)]
            if best is None or sc > best[0]:
                best = (sc, tf, td)
    return best[1], best[2], table


def size_bin(r) -> str:
    if not r["bbox"]:
        return "?"
    x1, y1, x2, y2 = r["bbox"]
    L = max(x2 - x1, y2 - y1)
    return "<200 px" if L < 200 else "200-400 px" if L < 400 else "400-800 px" if L < 800 else ">=800 px"


def breakdown(rows, variant, tf, td, truth, key) -> dict:
    out = {}
    for r in rows:
        if r.get(truth) is None or variant not in r["pred"]:
            continue
        b = key(r)
        o = out.setdefault(b, {"n": 0, "ours": 0, "baseline": 0})
        o["n"] += 1
        o["ours"] += int(rule_three(r["pred"][variant], tf, td) == r[truth])
        o["baseline"] += int(base_three(r["base_parts"]) == r[truth])
    return {b: {"n": o["n"], "ours_acc": round(o["ours"] / o["n"], 3), "baseline_acc": round(o["baseline"] / o["n"], 3)} for b, o in sorted(out.items())}


# ---------------------------------------------------------------------------------------------------------- clicks
def class_map(day, key, variant, name):
    """(class map, a, ox, oy): map pixel = (p2560 - (ox, oy)) * a."""
    stem = key.rsplit("~", 1)[0]
    if variant == "full":
        m = np.asarray(Image.open(DAYS / day / name / "full" / f"{stem}.png"))
        H, W = Image.open(DAYS / day / "images" / f"{stem}.jpg").size[::-1]
        return m, m.shape[1] / W, 0.0, 0.0
    m = np.asarray(Image.open(DAYS / day / name / "hr" / f"{key}.png"))
    return m, None, None, None


def instance_in_map(day, stem, k, shape, a, ox, oy, table):
    from .predict import instances_of

    masks, (x0, y0, s) = instances_of(day, stem, table)
    if k >= len(masks):
        return None
    M = np.array([[a / s, 0, (x0 - ox) * a], [0, a / s, (y0 - oy) * a]])
    inst = cv2.warpAffine(masks[k].astype(np.uint8), M, (shape[1], shape[0]), flags=cv2.INTER_NEAREST) > 0
    return inst


def click_metrics(name: str, variant: str, table: dict) -> dict:
    out = []
    for day in CLICK_DAYS:
        if not (DAYS / day / name / "instances.json").exists():
            continue
        labs = json.loads((LAB / day / "pair_marks.json").read_text())
        labs = labs["labels"] if isinstance(labs, dict) else labs
        photos = {}
        for l in labs:
            if l.get("verdict") == "skip":
                continue
            for sd in ("a", "b"):
                kp = (l.get("keypoints") or {}).get(sd) or {}
                if kp and l.get(f"key_{sd}"):
                    photos.setdefault(l[f"key_{sd}"], {}).update(kp)
        inst_json = json.loads((DAYS / day / name / "instances.json").read_text())["instances"]
        for key, kp in photos.items():
            if key not in inst_json:
                continue
            stem, k = key.rsplit("~", 1)
            k = int(k)
            W, H = Image.open(DAYS / day / "images" / f"{stem}.jpg").size
            if variant == "full":
                cm, a, ox, oy = class_map(day, key, "full", name)
            else:
                rec = inst_json[key]
                if "hr_window" not in rec:
                    continue
                hx, hy, side, osz = rec["hr_window"]
                cm = np.asarray(Image.open(DAYS / day / name / "hr" / f"{key}.png"))
                a, ox, oy = osz / side, hx, hy
            inst = instance_in_map(day, stem, k, cm.shape, a, ox, oy, table)
            if inst is None:
                continue
            near = cv2.dilate(inst.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
            pts = {n: np.array([(p["x"] * W - ox) * a, (p["y"] * H - oy) * a]) for n, p in kp.items()}
            rec = {"day": day, "key": key, "clicked": sorted(kp)}
            fin_all = cm == 2
            fin = (fin_all & near).astype(np.uint8)
            n, lab, st, _ = cv2.connectedComponentsWithStats(fin, connectivity=8)
            region = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))) if n > 1 else np.zeros_like(inst)

            def dist(mask, p):
                if not mask.any():
                    return None
                x, y = int(round(p[0])), int(round(p[1]))
                if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]:
                    return 0.0
                ys, xs = np.nonzero(mask)
                return float(np.min(np.hypot(xs - p[0], ys - p[1])) / a)        # 2560 px

            if "dorsal_fin_tip" in pts:
                rec["tip_to_fin_region_px"] = dist(region.astype(bool), pts["dorsal_fin_tip"])
                rec["tip_to_any_fin_px"] = dist(fin_all, pts["dorsal_fin_tip"])
                if "dorsal_fin_front" in pts and "dorsal_fin_back" in pts:
                    base = float(np.hypot(*(pts["dorsal_fin_front"] - pts["dorsal_fin_back"])) / a)
                    rec["fin_base_px"] = base
                    if region.any() and base > 1:
                        ys, xs = np.nonzero(region)
                        rec["region_extent_over_base"] = float((max(np.ptp(xs), np.ptp(ys)) + 1) / a / base)
                        fin_c = (pts["dorsal_fin_front"] + pts["dorsal_fin_back"] + pts["dorsal_fin_tip"]) / 3
                        rec["centroid_off_over_base"] = float(np.hypot(xs.mean() - fin_c[0], ys.mean() - fin_c[1]) / a / base)
                        rec["tip_to_region_over_base"] = rec["tip_to_fin_region_px"] / base
            fl = (cm == 1) & near
            for n_ in ("fluke_notch", "fluke_tip_left", "fluke_tip_right"):
                if n_ in pts:
                    rec[f"{n_}_to_fluke_px"] = dist(fl, pts[n_])
            if "fluke_tip_left" in pts and "fluke_tip_right" in pts:
                rec["fluke_span_px"] = float(np.hypot(*(pts["fluke_tip_left"] - pts["fluke_tip_right"])) / a)
            out.append(rec)
    return summarise_clicks(out), out


def summarise_clicks(recs) -> dict:
    def frac(vals, thr):
        v = [x for x in vals if x is not None]
        return {"n": len(vals), "with_region": len(v), "within": round(float(np.mean([x <= thr for x in v])) if v else 0, 3),
                "within_of_all": round(float(sum(x <= thr for x in v) / max(len(vals), 1)), 3)}

    tips = [r for r in recs if "tip_to_fin_region_px" in r]
    s = {"fin_tip_in_largest_fin_region": frac([r["tip_to_fin_region_px"] for r in tips], NEAR_PX),
         "fin_tip_in_any_fin_pixel": frac([r["tip_to_any_fin_px"] for r in tips], NEAR_PX),
         "fin_tip_in_any_fin_pixel_0px": frac([r["tip_to_any_fin_px"] for r in tips], 0.0)}
    based = [r for r in tips if "region_extent_over_base" in r]
    if based:
        ext = np.array([r["region_extent_over_base"] for r in based]); off = np.array([r["centroid_off_over_base"] for r in based])
        s["fin_region_extent_over_base"] = {"n": len(based), "median": round(float(np.median(ext)), 3), "q25": round(float(np.percentile(ext, 25)), 3),
                                            "q75": round(float(np.percentile(ext, 75)), 3)}
        s["fin_centroid_off_over_base"] = {"n": len(based), "median": round(float(np.median(off)), 3), "over_1": round(float(np.mean(off > 1)), 3),
                                           "over_0.5": round(float(np.mean(off > 0.5)), 3)}
        s["fin_tip_within_quarter_base"] = round(float(np.mean([r["tip_to_region_over_base"] <= 0.25 for r in based])), 3)
    for n_ in ("fluke_notch", "fluke_tip_left", "fluke_tip_right"):
        v = [r[f"{n_}_to_fluke_px"] for r in recs if f"{n_}_to_fluke_px" in r]
        s[f"{n_}_in_fluke"] = frac(v, NEAR_PX)
    both = [r for r in recs if r.get("fluke_tip_left_to_fluke_px") is not None or r.get("fluke_tip_right_to_fluke_px") is not None]
    flk = [r for r in recs if "fluke_tip_left_to_fluke_px" in r and "fluke_tip_right_to_fluke_px" in r]
    s["fluke_all_three_points_in"] = round(float(np.mean([all((r.get(f"{n}_to_fluke_px") is not None and r[f"{n}_to_fluke_px"] <= NEAR_PX)
                                                             for n in ("fluke_notch", "fluke_tip_left", "fluke_tip_right") if f"{n}_to_fluke_px" in r)
                                                         for r in flk])), 3) if flk else None
    s["fluke_photos_with_both_tips"] = len(flk)
    return s


# ---------------------------------------------------------------------------------------------------------- sheets
def overlay(rgb, cls):
    pal = np.zeros((256, 3), np.uint8)
    pal[: len(PALETTE) // 3] = np.asarray(PALETTE, np.uint8).reshape(-1, 3)
    col = pal[cls]
    m = (cls > 0) & (cls < 255)
    o = rgb.copy()
    o[m] = (0.4 * rgb[m] + 0.6 * col[m]).astype(np.uint8)
    return o


def tile(r, name, variant, size=200):
    day, key = r["day"], r["key"]
    stem = key.rsplit("~", 1)[0]
    bgr = cv2.imread(str(DAYS / day / "images" / f"{stem}.jpg"))
    H, W = bgr.shape[:2]
    x1, y1, x2, y2 = r["bbox"]
    side = max(x2 - x1, y2 - y1) * 1.3 + 20
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    gx0, gy0 = cx - side / 2, cy - side / 2
    M = np.array([[size / side, 0, -gx0 * size / side], [0, size / side, -gy0 * size / side]])
    img = cv2.warpAffine(bgr, M, (size, size), flags=cv2.INTER_AREA)
    if variant == "full":
        cm = np.asarray(Image.open(DAYS / day / name / "full" / f"{stem}.png"))
        a = cm.shape[1] / W
        Mc = np.array([[size / side / a, 0, -gx0 * size / side], [0, size / side / a, -gy0 * size / side]])
    else:
        hx, hy, hside, osz = r["pred"]["hr_window"]
        cm = np.asarray(Image.open(DAYS / day / name / "hr" / f"{key}.png"))
        a = osz / hside
        Mc = np.array([[size / side / a, 0, (hx - gx0) * size / side], [0, size / side / a, (hy - gy0) * size / side]])
    cls = cv2.warpAffine(cm, Mc, (size, size), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    ov = overlay(img[..., ::-1], cls)[..., ::-1]
    t = np.concatenate([img, np.ascontiguousarray(ov)], 1)
    return t


def make_sheet(rows, name, variant, path: Path, title: str, cols=4, size=200):
    tiles, legend = [], []
    for k, r in enumerate(rows[:24]):
        t = tile(r, name, variant, size)
        lbl = f"{k + 1}"
        cv2.putText(t, lbl, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(t, lbl, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        tiles.append(t)
        legend.append({"tile": k + 1, "key": r["key"], "day": r["day"], "truth_eye": r.get("eye"), "truth_vc": r.get("t_vc"),
                       "vc": r.get("vc"), "pred": r.get("p_three"), "shares_fluke_fin": [round(x, 4) for x in shares(r["pred"][variant])],
                       "pixels": r["pred"][variant]["pixels"], "baseline": base_three(r["base_parts"]), "n_inst": r["n_inst"], "gtype": r["gtype"]})
    if not tiles:
        return None
    h, w = tiles[0].shape[:2]
    nrows = (len(tiles) + cols - 1) // cols
    sheet = np.full((30 + nrows * (h + 4), cols * (w + 4), 3), 255, np.uint8)
    cv2.putText(sheet, title, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    for k, t in enumerate(tiles):
        rr, cc = divmod(k, cols)
        sheet[30 + rr * (h + 4): 30 + rr * (h + 4) + h, cc * (w + 4): cc * (w + 4) + w] = t
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    path.with_suffix(".json").write_text(json.dumps({"title": title, "variant": variant, "tiles": legend}, indent=1))
    return path


def multi_sheet(rows, name, variant, path: Path, size=300, cols=3, seed=0):
    """photos with >= 2 whale instances: the whole region of all instances, the photo next to the overlay."""
    by = {}
    for r in rows:
        if r["n_inst"] >= 2 and not r["not_whale"]:
            by.setdefault((r["day"], r["stem"]), []).append(r)
    keys = sorted(by)
    random.Random(seed).shuffle(keys)
    tiles, legend = [], []
    for (day, stem) in keys[:18]:
        rs = by[(day, stem)]
        bx = np.array([r["bbox"] for r in rs if r["bbox"]])
        fake = dict(rs[0], bbox=[bx[:, 0].min(), bx[:, 1].min(), bx[:, 2].max(), bx[:, 3].max()])
        t = tile(fake, name, "full", size)
        cv2.putText(t, str(len(tiles) + 1), (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(t)
        legend.append({"tile": len(tiles), "day": day, "stem": stem,
                       "instances": [{"key": r["key"], "truth_vc": r.get("t_vc"), "truth_eye": r.get("eye"), "pred": r.get("p_three")} for r in rs]})
    if not tiles:
        return None
    h, w = tiles[0].shape[:2]
    nrows = (len(tiles) + cols - 1) // cols
    sheet = np.full((30 + nrows * (h + 4), cols * (w + 4), 3), 255, np.uint8)
    cv2.putText(sheet, "photos with two or more whale instances (full-frame variant; box = all instances)", (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    for k, t in enumerate(tiles):
        rr, cc = divmod(k, cols)
        sheet[30 + rr * (h + 4): 30 + rr * (h + 4) + h, cc * (w + 4): cc * (w + 4) + w] = t
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    path.with_suffix(".json").write_text(json.dumps({"variant": "full", "tiles": legend}, indent=1))
    return path


# ---------------------------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--name", default="parts_seg_r05_a")
    ap.add_argument("--days", default="train,val")
    ap.add_argument("--sheets", type=Path, default=None)
    ap.add_argument("--sheet-variant", default=None, help="full / hr (default: the better on the val days)")
    a = ap.parse_args(argv)
    if a.sheets is not None and str(a.sheets.resolve()).startswith(str(ROOT.resolve())):
        raise SystemExit("sheets show real photos: write them outside the repository")
    days = day_list(a.days)
    assert not set(days) & EVAL_DAYS
    rows = load_all(a.name, days)
    for r in rows:
        r["t_eye"] = eye_three(r["eye"])
        r["t_vc"] = vc_three(r["vc"]) if not r["not_whale"] else None
        r["t_eye_marg"] = r["eye"] if eye_marginal(r["eye"]) else None
    from ..common import geometry
    t = geometry.load_folded_table(paths.FOLDED_TABLE)
    table = {"x_min_m": float(t["x_min_m"]), "length_m": float(t["length_m"])}
    res = {"instances": len(rows), "days": days, "truth_counts": {}}
    for split in ("train", "val"):
        sr = [r for r in rows if r["split"] == split]
        res["truth_counts"][split] = {"instances": len(sr), "eye_confident": sum(r["t_eye"] is not None for r in sr),
                                      "eye_confident_r0": sum(r["t_eye"] is not None and r["eye_round"] == "r0" for r in sr),
                                      "eye_marginal": sum(r["t_eye_marg"] is not None for r in sr), "vc_confident": sum(r["t_vc"] is not None for r in sr)}
    res["variants"] = {}
    for variant in ("full", "hr"):
        tf, td, grid = choose_rule(rows, variant)
        v = {"rule": {"fluke_share_min": tf[0], "fluke_over_pectoral_min": tf[1], "fin_share_min": td,
                      "text": "fluke if fluke share >= a and >= b x pectoral share, else dorsal if fin share >= c, else other (shares of the instance's pixels incl. predicted background)",
                      "chosen_on": "train days, mean of by-eye and classifier three-way accuracy"}}
        for split in ("train", "val", "all"):
            sr = [r for r in rows if split == "all" or r["split"] == split]
            r0 = [r for r in sr if r["eye_round"] == "r0"]
            v[split] = {"eye": score_set(sr, variant, tf, td, "t_eye"), "eye_random_r0": score_set(r0, variant, tf, td, "t_eye"),
                        "classifier": score_set(sr, variant, tf, td, "t_vc")}
            # marginal by-eye codes: does the model call the marginal part?
            marg = [r for r in sr if r["t_eye_marg"] and variant in r["pred"]]
            v[split]["eye_marginal"] = {
                "n": len(marg),
                "marginal_fluke_called_fluke": f"{sum(rule_three(r['pred'][variant], tf, td) == 'fluke' for r in marg if 'f' in r['eye'])}/{sum('f' in r['eye'] for r in marg)}",
                "marginal_fin_called_fin_or_fluke": f"{sum(rule_three(r['pred'][variant], tf, td) != 'other' for r in marg if 'd' in r['eye'])}/{sum('d' in r['eye'] for r in marg)}",
                "baseline_marginal_fluke_called_fluke": f"{sum(base_three(r['base_parts']) == 'fluke' for r in marg if 'f' in r['eye'])}/{sum('f' in r['eye'] for r in marg)}"}
        val = [r for r in rows if r["split"] == "val"]
        allr = rows
        v["breakdown_val_classifier"] = {
            "by_truth": breakdown(val, variant, tf, td, "t_vc", lambda r: r["t_vc"]),
            "by_size": breakdown(val, variant, tf, td, "t_vc", size_bin),
            "by_instances_in_photo": breakdown(val, variant, tf, td, "t_vc", lambda r: "1" if r["n_inst"] <= 1 else ">=2"),
            "by_group_type": breakdown(val, variant, tf, td, "t_vc", lambda r: r["gtype"])}
        v["breakdown_all_classifier"] = {
            "by_truth": breakdown(allr, variant, tf, td, "t_vc", lambda r: r["t_vc"]),
            "by_size": breakdown(allr, variant, tf, td, "t_vc", size_bin),
            "by_instances_in_photo": breakdown(allr, variant, tf, td, "t_vc", lambda r: "1" if r["n_inst"] <= 1 else ">=2"),
            "by_group_type": breakdown(allr, variant, tf, td, "t_vc", lambda r: r["gtype"])}
        v["breakdown_all_eye"] = {
            "by_truth": breakdown(allr, variant, tf, td, "t_eye", lambda r: r["t_eye"]),
            "by_size": breakdown(allr, variant, tf, td, "t_eye", size_bin),
            "by_instances_in_photo": breakdown(allr, variant, tf, td, "t_eye", lambda r: "1" if r["n_inst"] <= 1 else ">=2"),
            "by_group_type": breakdown(allr, variant, tf, td, "t_eye", lambda r: r["gtype"])}
        summ, recs = click_metrics(a.name, variant, table)
        v["clicks"] = summ
        res["variants"][variant] = v
        print(variant, "rule", tf, td, "val eye", v["val"]["eye"]["ours"]["acc"], "base", v["val"]["eye"]["baseline"]["acc"],
              "val vc", v["val"]["classifier"]["ours"]["acc"], "base", v["val"]["classifier"]["baseline"]["acc"], flush=True)
    better = max(("full", "hr"), key=lambda vv: (res["variants"][vv]["val"]["eye"]["ours"]["acc"] or 0) + (res["variants"][vv]["val"]["classifier"]["ours"]["acc"] or 0))
    res["better_variant_on_val"] = better
    a.out.write_text(json.dumps(res, indent=1))
    print("wrote", a.out, "better variant", better)
    if a.sheets is not None:
        variant = a.sheet_variant or better
        rr = res["variants"][variant]["rule"]
        tf, td = (rr["fluke_share_min"], rr["fluke_over_pectoral_min"]), rr["fin_share_min"]
        a.sheets.mkdir(parents=True, exist_ok=True)
        for r in rows:
            r["p_three"] = rule_three(r["pred"][variant], tf, td) if variant in r["pred"] else None
            r["truth"] = r["t_eye"] or r["t_vc"]            # by-eye first, then the confident classifier
        rng = random.Random(0)
        scored = [r for r in rows if r["truth"] and r["p_three"] and r["bbox"]]
        groups = {
            "01_fluke_missed_called_other": [r for r in scored if r["truth"] == "fluke" and r["p_three"] == "other"],
            "02_fluke_missed_called_fin": [r for r in scored if r["truth"] == "fluke" and r["p_three"] == "dorsal"],
            "03_fin_missed_called_other": [r for r in scored if r["truth"] == "dorsal" and r["p_three"] == "other"],
            "04_fluke_where_fin_view": [r for r in scored if r["truth"] == "dorsal" and r["p_three"] == "fluke"],
            "05_fluke_where_other_view": [r for r in scored if r["truth"] == "other" and r["p_three"] == "fluke"],
            "06_fin_where_other_view": [r for r in scored if r["truth"] == "other" and r["p_three"] == "dorsal"],
            "07_correct_random": [r for r in scored if r["truth"] == r["p_three"]],
        }
        made = []
        counts = {g: len(v) for g, v in groups.items()}
        for g, lst in groups.items():
            rng.shuffle(lst)
            p = make_sheet(lst, a.name, variant, a.sheets / f"{g}.jpg", f"{g} ({len(lst)} cases, {variant} variant; left photo, right predicted parts)")
            if p:
                made.append(str(p))
        p = multi_sheet(rows, a.name, variant, a.sheets / "08_multi_instance_photos.jpg")
        if p:
            made.append(str(p))
        (a.sheets / "sheets_index.json").write_text(json.dumps({"variant": variant, "counts": counts, "sheets": made}, indent=1))
        print("sheets", made, counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
