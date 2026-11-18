"""What is "the dorsal fin" for the researcher's clicks, the render label and the models? One frame for all.

    PYTHONPATH=. .venv/bin/python experiments/parts_weak/fin_definition.py [--models parts_seg_r05_a parts_seg_r05_w1]
        -> experiments/parts_weak/fin_definition.json, sheet SHEETS/fin_definition.jpg (photos: scratchpad only)

Frame of one fin = its triangle tip T, front base F, back base B (the researcher's clicks on real photos; the factory's
2D keypoints dorsal_fin_tip / dorsal_fin_front / dorsal_fin_back on run05 renders): "along" = position on the base line
from B (0) to F (1); "up" = height above the base line over the tip's height (base line 0, tip 1).
Per fin:  base / height (|F - B| over the tip's height above the base line);
          for the painted fin pixels near the triangle (render: the training label; real: a model's class map, hr
          window): how far down (5th percentile of "up"), how far forward / back (95th / 5th percentile of "along"),
          region length / base (the cut-outs' 0.28 measure: longer side of the region's box over |F - B|), and the share
          of the triangle covered.
Real fins: 2025-01-02 (W1 was trained on the tip part of these clicks: W1's numbers there are not independent) and
2025-03-18 (never trained on). Renders: run05 validation shots with all three keypoints visible.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from model_fine_tuning.common import paths  # noqa: E402
from model_fine_tuning.common.data import WhaleRun  # noqa: E402
from model_fine_tuning.parts_seg.labels import PALETTE  # noqa: E402
from model_fine_tuning.parts_weak import DAYS, LAB, SHEETS  # noqa: E402

SHADOW = Path("/var/tmp/sat3737-parts-cache/parts_weak/days_shadow")
LABELS = Path("/var/tmp/sat3737-parts-cache/run05/labels")
RUN05 = paths.WHALE_FACTORY_ROOT / "data" / "renders" / "run05"
OUT = Path(__file__).resolve().parent / "fin_definition.json"


def frame_stats(T, F, B, fin: np.ndarray) -> dict:
    u = F - B
    base = float(np.hypot(*u))
    if base < 3:
        return {}
    u = u / base
    w = (T - B) - ((T - B) @ u) * u
    h = float(np.hypot(*w))
    if h < 3:
        return {}
    n = w / h
    out = {"base_over_height": base / h, "base_px": base, "height_px": h, "tip_along": float((T - B) @ u / base)}
    ys, xs = np.nonzero(fin)
    if len(xs):
        P = np.stack([xs, ys], 1).astype(float) - B
        al, up = P @ u / base, P @ n / h
        near = (up > -0.6) & (up < 1.6) & (al > -0.6) & (al < 1.6)
        if near.sum() >= 3:
            al, up = al[near], up[near]
            pts = np.stack([xs[near], ys[near]], 1)
            out.update(painted=True, down=float(np.percentile(up, 5)), front=float(np.percentile(al, 95)), back=float(np.percentile(al, 5)),
                       length_over_base=float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])) + 1) / base)
            tri = np.zeros(fin.shape, np.uint8)
            cv2.fillPoly(tri, [np.round(np.stack([T, F, B])).astype(np.int32)], 1)
            out["triangle_covered"] = float((tri.astype(bool) & fin).sum() / max(tri.sum(), 1))
            fin_near = np.zeros(fin.shape, bool)
            fin_near[pts[:, 1], pts[:, 0]] = True
            out["iou_triangle"] = float((fin_near & tri.astype(bool)).sum() / max((fin_near | tri.astype(bool)).sum(), 1))
            out["painted_outside_triangle"] = float((fin[ys[near], xs[near]] & ~tri.astype(bool)[ys[near], xs[near]]).mean())
            # unclipped: the whole painted region connected to the triangle (the window above clips large bands)
            n_c, lab = cv2.connectedComponents(fin.astype(np.uint8), connectivity=8)
            touch = cv2.dilate(tri, np.ones((7, 7), np.uint8)) > 0
            ids = np.unique(lab[touch & fin])
            ids = ids[ids > 0]
            if len(ids):
                comp = np.isin(lab, ids)
                cy, cx = np.nonzero(comp)
                Q = np.stack([cx, cy], 1).astype(float) - B
                cal, cup = Q @ u / base, Q @ n / h
                out.update(comp_down=float(np.percentile(cup, 5)), comp_front=float(np.percentile(cal, 95)), comp_back=float(np.percentile(cal, 5)),
                           comp_length_over_base=float(max(np.ptp(cx), np.ptp(cy)) + 1) / base,
                           comp_outside_triangle=float((comp & ~tri.astype(bool)).sum() / comp.sum()),
                           comp_iou_triangle=float((comp & tri.astype(bool)).sum() / max((comp | tri.astype(bool)).sum(), 1)),
                           comp_area_over_triangle=float(comp.sum() / max(tri.sum(), 1)))
            return out
    out["painted"] = False
    return out


def clicks(day: str) -> dict:
    labs = json.loads((LAB / day / "pair_marks.json").read_text())
    labs = labs["labels"] if isinstance(labs, dict) else labs
    kp = {}
    for l in labs:
        if l.get("verdict") == "skip":
            continue
        for sd in ("a", "b"):
            k = (l.get("keypoints") or {}).get(sd) or {}
            if k and l.get(f"key_{sd}"):
                kp.setdefault(l[f"key_{sd}"], {}).update(k)
    return {k: v for k, v in kp.items() if all(n in v for n in ("dorsal_fin_tip", "dorsal_fin_front", "dorsal_fin_back"))}


def real(day: str, models: list[str]) -> tuple[list, list]:
    rows, tiles = [], []
    ck = clicks(day)
    insts = {}
    for m in models:
        root = SHADOW if (SHADOW / day / m / "instances.json").exists() else DAYS
        insts[m] = (root, json.loads((root / day / m / "instances.json").read_text())["instances"])
    for key, kp in sorted(ck.items()):
        stem = key.rsplit("~", 1)[0]
        if not all(key in insts[m][1] and "hr_window" in insts[m][1][key] for m in models):
            continue
        W, H = Image.open(DAYS / day / "images" / f"{stem}.jpg").size
        row = {"day": day, "key": key}
        maps = {}
        for m in models:
            root, ins = insts[m]
            hx, hy, side, osz = ins[key]["hr_window"]
            a = osz / side
            P = {n: np.array([(kp[n]["x"] * W - hx) * a, (kp[n]["y"] * H - hy) * a]) for n in ("dorsal_fin_tip", "dorsal_fin_front", "dorsal_fin_back")}
            cm = np.asarray(Image.open(root / day / m / "hr" / f"{key}.png"))
            row[m] = frame_stats(P["dorsal_fin_tip"], P["dorsal_fin_front"], P["dorsal_fin_back"], cm == 2)
            maps[m] = (cm, P, (hx, hy, side, osz))
        rows.append(row)
        tiles.append((day, stem, maps))
    return rows, tiles


def renders(n_max: int = 400) -> tuple[list, list]:
    run = WhaleRun(RUN05, paths.SPLITS_DIR / "run05.json")
    rows, tiles = [], []
    for sid in run.ids("val"):
        m = run.meta(sid)
        k = m.get("keypoints") or {}
        need = ("dorsal_fin_tip", "dorsal_fin_front", "dorsal_fin_back")
        if not all(n in k and k[n].get("state") == "visible" and k[n].get("in_frame") for n in need):
            continue
        p = LABELS / f"{sid}.png"
        if not p.exists():
            continue
        lab = np.asarray(Image.open(p))
        P = {n: np.array(k[n]["uv"], float) for n in need}
        st = frame_stats(P["dorsal_fin_tip"], P["dorsal_fin_front"], P["dorsal_fin_back"], lab == 2)
        if not st:
            continue
        rows.append({"id": sid, "category": m.get("category"), "label": st})
        if len(tiles) < 12 and st.get("painted") and st["height_px"] > 25:
            tiles.append((sid, run, lab, P))
        if len(rows) >= n_max:
            break
    return rows, tiles


def summ(vals: list[dict]) -> dict:
    out = {"n": len(vals), "painted": round(float(np.mean([v.get("painted", False) for v in vals])), 3) if vals else None}
    for f in ("base_over_height", "tip_along", "down", "front", "back", "length_over_base", "triangle_covered", "painted_outside_triangle", "iou_triangle",
              "comp_down", "comp_front", "comp_back", "comp_length_over_base", "comp_outside_triangle", "comp_iou_triangle", "comp_area_over_triangle"):
        x = [v[f] for v in vals if f in v]
        if x:
            out[f] = {"median": round(float(np.median(x)), 3), "q25": round(float(np.percentile(x, 25)), 3), "q75": round(float(np.percentile(x, 75)), 3)}
    return out


def overlay(rgb, cls):
    pal = np.zeros((256, 3), np.uint8)
    pal[: len(PALETTE) // 3] = np.asarray(PALETTE, np.uint8).reshape(-1, 3)
    m = (cls > 0) & (cls < 255)
    o = rgb.copy()
    o[m] = (0.45 * rgb[m] + 0.55 * pal[cls][m]).astype(np.uint8)
    return o


def crop_tile(img, cls, P, size=220):
    pts = np.stack([P["dorsal_fin_tip"], P["dorsal_fin_front"], P["dorsal_fin_back"]])
    c = pts.mean(0)
    s = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])) * 1.8 + 20
    M = np.array([[size / s, 0, -(c[0] - s / 2) * size / s], [0, size / s, -(c[1] - s / 2) * size / s]])
    a = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR)
    b = overlay(a, cv2.warpAffine(cls, M, (size, size), flags=cv2.INTER_NEAREST, borderValue=0))
    q = np.round((np.c_[pts, np.ones(3)] @ M.T)).astype(np.int32)
    cv2.polylines(b, [q], True, (255, 255, 255), 1)
    for (x, y), col in zip(q, ((255, 0, 255), (0, 255, 255), (255, 255, 0))):
        cv2.circle(b, (int(x), int(y)), 3, col, -1)
    return a, b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["parts_seg_r05_a", "parts_seg_r05_w1"])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--sheet-name", default="fin_definition.jpg")
    a = ap.parse_args()
    res = {"frame": "along: back base 0 -> front base 1; up: base line 0 -> tip height 1", "real": {}, "render_label": None}
    real_tiles = []
    for day in ("2025-01-02", "2025-03-18"):
        rows, tiles = real(day, a.models)
        res["real"][day] = {"n_fins": len(rows), "clicks": summ([r[a.models[0]] for r in rows]),
                            **{m: summ([r[m] for r in rows if r[m]]) for m in a.models}}
        if day == "2025-01-02":
            real_tiles = tiles
    rrows, rtiles = renders()
    res["render_label"] = summ([r["label"] for r in rrows])
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    # sheet: 12 real fins of 2025-01-02 (photo | baseline | W1) and 12 renders (render | label)
    rng = np.random.default_rng(0)
    pick = [real_tiles[i] for i in rng.choice(len(real_tiles), min(12, len(real_tiles)), replace=False)]
    rows_img = []
    for day, stem, maps in pick:
        rgb = cv2.cvtColor(cv2.imread(str(DAYS / day / "images" / f"{stem}.jpg")), cv2.COLOR_BGR2RGB)
        panels = []
        for k, m in enumerate(a.models):
            cm, P, (hx, hy, side, osz) = maps[m]
            M = np.array([[osz / side, 0, -hx * osz / side], [0, osz / side, -hy * osz / side]])
            win = cv2.warpAffine(rgb, M, (int(osz), int(osz)), flags=cv2.INTER_LINEAR)
            ph, ov = crop_tile(win, cm, P)
            if k == 0:
                panels.append(ph)
            panels.append(ov)
        rows_img.append(np.hstack([np.pad(p, ((2, 2), (2, 2), (0, 0)), constant_values=255) for p in panels]))
    rend = []
    for sid, run, lab, P in rtiles:
        ph, ov = crop_tile(run.image(sid), lab, P)
        rend.append(np.hstack([np.pad(p, ((2, 2), (2, 2), (0, 0)), constant_values=255) for p in (ph, ov)]))
    W1 = max(r.shape[1] for r in rows_img)
    left = np.vstack([np.pad(r, ((0, 0), (0, W1 - r.shape[1]), (0, 0)), constant_values=255) for r in rows_img])
    right = np.vstack(rend)
    Hh = max(left.shape[0], right.shape[0])
    left = np.pad(left, ((0, Hh - left.shape[0]), (0, 20), (0, 0)), constant_values=255)
    right = np.pad(right, ((0, Hh - right.shape[0]), (0, 0), (0, 0)), constant_values=255)
    sheet = np.hstack([left, right])
    head = np.full((30, sheet.shape[1], 3), 255, np.uint8)
    cv2.putText(head, "left: 2025-01-02 clicked fins: photo | " + " | ".join(a.models) + "    right: run05 renders: render | training label   "
                "(white triangle = tip / front base (cyan) / back base (yellow); green = fin)", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    SHEETS.mkdir(parents=True, exist_ok=True)
    p = SHEETS / a.sheet_name
    cv2.imwrite(str(p), cv2.cvtColor(np.vstack([head, sheet]), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
