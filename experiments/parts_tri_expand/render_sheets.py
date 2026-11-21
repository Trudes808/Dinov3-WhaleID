"""Sheets (scratchpad only): triexp_render_examples.jpg (check21, 24 fins at random, seed 0: render | triangle | A on the
exact mask | A on S6h | exact fin + hump label with the blade outlined) and triexp_failures.jpg (the worst check20 +
check21 fins by IoU with S6h, with the cause measured: other whale / own fluke / not whale / under the line).

    .venv/bin/python experiments/parts_tri_expand/render_sheets.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_fins as RF  # noqa: E402
from model_fine_tuning.parts_weak import tri_expand as TE  # noqa: E402
from real_fins import SHEETS, grid_sheet  # noqa: E402

OUT = Path(__file__).resolve().parent


def tile(r, img, panels, size=180, title=""):
    x0, y0, x1, y1 = r["win"]
    pts = np.stack([r["Tw"], r["Fw"], r["Bw"]])
    fr = TE.frame(r["Tw"], r["Fw"], r["Bw"])
    c = pts.mean(0)
    s = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), fr["h_eff"]) * 2.2 + 12
    M = np.array([[size / s, 0, -(c[0] - s / 2) * size / s], [0, size / s, -(c[1] - s / 2) * size / s]])
    win = img[y0 * 2:y1 * 2, x0 * 2:x1 * 2]
    win = cv2.resize(win, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
    base = cv2.warpAffine(win, M, (size, size), flags=cv2.INTER_LINEAR)
    q = np.round(np.c_[pts, np.ones(3)] @ M.T).astype(np.int32)
    out = []
    for kind, d, col in panels:
        t = base.copy()
        if d is not None:
            m = cv2.warpAffine(d.astype(np.uint8), M, (size, size), flags=cv2.INTER_NEAREST) > 0
            ov = t.copy()
            ov[m] = col
            t = cv2.addWeighted(t, 0.55, ov, 0.45, 0)
        if kind == "tri":
            ov = t.copy()
            cv2.fillPoly(ov, [q], (0, 200, 255))
            t = cv2.addWeighted(t, 0.6, ov, 0.4, 0)
        if kind == "label":
            bm = cv2.warpAffine(r["fin"].astype(np.uint8), M, (size, size), flags=cv2.INTER_NEAREST)
            cs, _ = cv2.findContours(bm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(t, cs, -1, (255, 255, 255), 1)
        if kind != "photo":
            cv2.polylines(t, [q], True, (0, 255, 255), 1, cv2.LINE_AA)
        out.append(t)
    row = np.hstack([np.pad(x, ((2, 2), (2, 2), (0, 0)), constant_values=255) for x in out])
    lab = np.full((18, row.shape[1], 3), 255, np.uint8)
    cv2.putText(lab, title, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([lab, row])


def image(st, sid):
    return cv2.imread(str(RF.FACTORY / st / "dataset" / "images" / f"{sid}.png"))


def main() -> None:
    p = json.loads((OUT / "rule.json").read_text())["params"]
    tiles = []
    st = "r06r_check21"
    fins = [r for r in RF.load_set(st, ("gt", "s6h")) if r["all_visible"]]
    rows = {x["key"]: x for x in json.loads((RF.CACHE / f"render_rows_{st}.json").read_text())}
    rng = np.random.default_rng(0)
    for i in sorted(rng.choice(len(fins), 24, replace=False)):
        r = fins[i]
        e_gt = TE.expand(r["Tw"], r["Fw"], r["Bw"], r["mask"]["gt"], **p)
        e_s6 = TE.expand(r["Tw"], r["Fw"], r["Bw"], r["mask"]["s6h"], **p)
        x = rows[r["key"]]
        t = f"{r['key']} view {x['sight_deg']:.0f} deg  IoU tri {x['tri']['fin_hump'][0]:.2f} A {x['exp']['gt']['fin_hump'][0]:.2f} A-S6h {x['exp']['s6h']['fin_hump'][0]:.2f}"
        tiles.append(tile(r, image(st, r["sid"]), [("photo", None, None), ("tri", None, None), ("e", e_gt, (40, 40, 255)), ("e", e_s6, (40, 40, 255)),
                                                    ("label", r["dorsal"], (60, 200, 60))], title=t))
    head = [f"Renders r06r_check21 (report set), 24 fins at random (seed 0). Rule A {p}.",
            "render | clicked-style triangle from the true fin points | A on the exact whale mask (red) | A on the S6h mask run on the render | exact run06 dorsal label (green; white = blade)"]
    SHEETS.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(SHEETS / "triexp_render_examples.jpg"), grid_sheet(tiles, 2, head), [cv2.IMWRITE_JPEG_QUALITY, 88])
    # failures: 16 worst (S6h) over both sets, cause from the measured leak pixels
    worst = []
    for st in ("r06r_check20", "r06r_check21"):
        fins = {r["key"]: r for r in RF.load_set(st, ("gt", "s6h"))}
        for x in json.loads((RF.CACHE / f"render_rows_{st}.json").read_text()):
            if "s6h" in x["exp"]:
                worst.append((x["exp"]["s6h"]["fin_hump"][0] if np.isfinite(x["exp"]["s6h"]["fin_hump"][0]) else 0.0, st, x, fins[x["key"]]))
    worst.sort(key=lambda w: w[0])
    tiles = []
    for v, st, x, r in worst[:16]:
        e_s6 = TE.expand(r["Tw"], r["Fw"], r["Bw"], r["mask"]["s6h"], **p)
        e_gt = TE.expand(r["Tw"], r["Fw"], r["Bw"], r["mask"]["gt"], **p)
        lk = x["exp"]["s6h"]["leak_px"]
        miss = 1 - (x["exp"]["s6h"]["fin_hump"][1] if np.isfinite(x["exp"]["s6h"]["fin_hump"][1]) else 0)
        cause = max(lk, key=lk.get) if sum(lk.values()) > 0.25 * max(x["exp"]["s6h"]["px"], 1) else "missed (mask / view)"
        t = f"{st[-7:]} {x['key']} IoU S6h {v:.2f} exact {x['exp']['gt']['fin_hump'][0]:.2f}; {cause}; missed {miss:.0%}; view {x['sight_deg']:.0f}; whales {x['n_animals']}; base/h {x['base_over_h']:.1f}"
        tiles.append(tile(r, image(st, r["sid"]), [("photo", None, None), ("e", e_gt, (40, 40, 255)), ("e", e_s6, (40, 40, 255)), ("label", r["dorsal"], (60, 200, 60))], title=t))
    head = ["Failures: the 16 renders (check20 + check21) with the lowest IoU of rule A on S6h against the run06 fin + hump label.",
            "render | A on the exact mask | A on S6h | exact label (green, white = blade). Cause = the largest source of extra pixels when they are over a quarter, else 'missed'."]
    cv2.imwrite(str(SHEETS / "triexp_failures_renders.jpg"), grid_sheet(tiles, 2, head), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("sheets written")


if __name__ == "__main__":
    main()
