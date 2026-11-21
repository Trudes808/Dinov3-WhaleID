"""The frozen rule (rule.json) on the researcher's clicked fins of the two click days (2025-01-02, 2025-03-18; both
training days, read only), with the candidate mask S6h and the live mask (seg_r04_ft4, P >= 0.5), at 1280 px.

    .venv/bin/python experiments/parts_tri_expand/real_fins.py stats         -> real_stats.json (numbers only)
    .venv/bin/python experiments/parts_tri_expand/real_fins.py sheet         -> SHEETS/triexp_real_examples.jpg (40 at random,
                                                                                 seed 0, + the 8 worst)
    .venv/bin/python experiments/parts_tri_expand/real_fins.py tally-sheet   -> SHEETS/triexp_tally_*.jpg (60 at random, seed 1)
Photos and sheets never go into the repository.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "parts_weak"))
from fin_definition import clicks  # noqa: E402
from model_fine_tuning.parts_weak import DAYS  # noqa: E402
from model_fine_tuning.parts_weak import tri_expand as TE  # noqa: E402

OUT = Path(__file__).resolve().parent
PRED = Path("/var/tmp/sat3737-parts-cache/seg_splash/pred")
SHEETS = Path("/tmp/claude-1072903897/-home-sat3737-NOAA-whales/4fbba7ff-3ba5-4b50-9cb7-970c81250409/scratchpad/triexp/sheets")
CLICK_DAYS = ("2025-01-02", "2025-03-18")
NEED = ("dorsal_fin_tip", "dorsal_fin_front", "dorsal_fin_back")


def params():
    r = json.loads((OUT / "rule.json").read_text())
    return {"A": r["params"], "B": r["params_B"]}


def masks(day: str, stem: str) -> dict:
    s6 = cv2.imread(str(PRED / "s6h" / day / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
    r4 = cv2.imread(str(PRED / "r04" / day / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
    return {"s6h": s6 > 0, "live": r4 >= 128}


def fins():
    out = []
    for day in CLICK_DAYS:
        for key, kp in sorted(clicks(day).items()):
            out.append({"day": day, "key": key, "stem": key.rsplit("~", 1)[0], "kp": {n: (kp[n]["x"], kp[n]["y"]) for n in NEED}})
    return out


def apply(f: dict, p: dict) -> dict:
    M = masks(f["day"], f["stem"])
    H, W = M["s6h"].shape
    T, F, B = (np.array([f["kp"][n][0] * W, f["kp"][n][1] * H]) for n in NEED)
    tri = TE.triangle(T, F, B, (H, W))
    res = {"T": T, "F": F, "B": B, "tri": tri, "masks": M, "E": {}}
    for mk, m in M.items():
        res["E"][mk] = TE.expand(T, F, B, m, **p["A"])
        res["E"][mk + "_B"] = TE.expand(T, F, B, m, **p["B"])
    return res


def cmd_stats(a) -> None:
    p = params()
    rows = []
    for f in fins():
        r = apply(f, p)
        fr = TE.frame(r["T"], r["F"], r["B"])
        ta = int(r["tri"].sum())
        row = {"day": f["day"], "key": f["key"], "tri_px_1280": ta, "base_px_1280": fr["base"], "h_px_1280": fr["h"], "base_over_h": fr["base"] / fr["h"]}
        for mk in ("s6h", "live", "s6h_B", "live_B"):
            e, m = r["E"][mk], r["masks"][mk.replace("_B", "")]
            tm = int((r["tri"] & m).sum())
            row[mk] = {"tri_outside_mask": 1 - tm / max(ta, 1), "exp_over_tri": int(e.sum()) / max(ta, 1),
                       "added_over_tri": int((e & ~r["tri"]).sum()) / max(ta, 1), "exp_px": int(e.sum())}
        rows.append(row)
    def q(x):
        x = np.asarray(x)
        return {"median": round(float(np.median(x)), 3), "q25": round(float(np.percentile(x, 25)), 3), "q75": round(float(np.percentile(x, 75)), 3),
                "mean": round(float(np.mean(x)), 3)}
    summ = {"params": p, "n_fins": len(rows), "by_day": {d: sum(r["day"] == d for r in rows) for d in CLICK_DAYS},
            "fin_size_1280": {"h_px": q([r["h_px_1280"] for r in rows]), "base_over_h": q([r["base_over_h"] for r in rows])}}
    for mk in ("s6h", "live", "s6h_B", "live_B"):
        summ[mk] = {k: q([r[mk][k] for r in rows]) for k in ("tri_outside_mask", "exp_over_tri", "added_over_tri")}
        summ[mk]["fins_with_tri_outside_mask_over_0.25"] = int(sum(r[mk]["tri_outside_mask"] > 0.25 for r in rows))
        summ[mk]["fins_exp_over_tri_over_2"] = int(sum(r[mk]["exp_over_tri"] > 2 for r in rows))
    (OUT / "real_stats.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=1))
    print(json.dumps(summ, indent=1))


# ------------------------------------------------------------------------------------------------------ sheets
def crop_box(r, scale_up=2.0):
    pts = np.stack([r["T"], r["F"], r["B"]])
    c = pts.mean(0)
    fr = TE.frame(r["T"], r["F"], r["B"])
    s = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), fr["h_eff"]) * scale_up + 16
    return c, s


def tile(img2560, r, layers, size=240, title=""):
    """img2560: the photo; r: apply() result (1280 frame). layers: list of (kind, data)."""
    c, s = crop_box(r)
    k = 2.0                                                   # 1280 -> 2560
    M = np.array([[size / (s * k), 0, -(c[0] * k - s * k / 2) * size / (s * k)], [0, size / (s * k), -(c[1] * k - s * k / 2) * size / (s * k)]])
    base = cv2.warpAffine(img2560, M, (size, size), flags=cv2.INTER_AREA, borderValue=(0, 0, 0))
    M1 = M.copy()
    M1[:, :2] *= k                                            # 1280-frame -> tile
    out = []
    for kind, d in layers:
        t = base.copy()
        if kind == "tri":
            q = np.round(np.c_[np.stack([r["T"], r["F"], r["B"]]), np.ones(3)] @ M1.T).astype(np.int32)
            ov = t.copy()
            cv2.fillPoly(ov, [q], (0, 200, 255))
            t = cv2.addWeighted(t, 0.6, ov, 0.4, 0)
            cv2.polylines(t, [q], True, (0, 255, 255), 1, cv2.LINE_AA)
        elif kind == "mask":
            m = cv2.warpAffine(d.astype(np.uint8), M1, (size, size), flags=cv2.INTER_NEAREST) > 0
            ov = t.copy()
            ov[m] = (40, 40, 255)
            t = cv2.addWeighted(t, 0.55, ov, 0.45, 0)
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(t, cs, -1, (60, 60, 255), 1)
            q = np.round(np.c_[np.stack([r["T"], r["F"], r["B"]]), np.ones(3)] @ M1.T).astype(np.int32)
            cv2.polylines(t, [q], True, (0, 255, 255), 1, cv2.LINE_AA)
        elif kind == "whale":
            m = cv2.warpAffine(d.astype(np.uint8), M1, (size, size), flags=cv2.INTER_NEAREST) > 0
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(t, cs, -1, (255, 255, 0), 1)
        out.append(t)
    row = np.hstack([np.pad(x, ((2, 2), (2, 2), (0, 0)), constant_values=255) for x in out])
    lab = np.full((18, row.shape[1], 3), 255, np.uint8)
    cv2.putText(lab, title, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([lab, row])


def grid_sheet(tiles, ncol, head):
    w = max(t.shape[1] for t in tiles)
    h = max(t.shape[0] for t in tiles)
    tiles = [np.pad(t, ((0, h - t.shape[0]), (0, w - t.shape[1] + 10), (0, 0)), constant_values=255) for t in tiles]
    rows = []
    for i in range(0, len(tiles), ncol):
        r = tiles[i:i + ncol]
        r += [np.full_like(tiles[0], 255)] * (ncol - len(r))
        rows.append(np.hstack(r))
    sheet = np.vstack(rows)
    hd = np.full((24 * len(head) + 8, sheet.shape[1], 3), 255, np.uint8)
    for i, line in enumerate(head):
        cv2.putText(hd, line, (6, 20 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([hd, sheet])


def cmd_sheet(a) -> None:
    p = params()
    st = json.loads((OUT / "real_stats.json").read_text())["rows"]
    F = fins()
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(len(F), 40, replace=False).tolist())
    rest = [i for i in range(len(F)) if i not in pick]
    # 8 worst (S6h): the 4 largest growths over the triangle and the 4 largest triangle shares outside the mask
    by_grow = sorted(rest, key=lambda i: -st[i]["s6h"]["exp_over_tri"])[:4]
    by_out = sorted([i for i in rest if i not in by_grow], key=lambda i: -st[i]["s6h"]["tri_outside_mask"])[:4]
    tiles = []
    for tag, idx in (("random", pick), ("worst: grows most", by_grow), ("worst: triangle off the mask", by_out)):
        for i in idx:
            f = F[i]
            r = apply(f, p)
            img = cv2.imread(str(DAYS / f["day"] / "images" / f"{f['stem']}.jpg"))
            s = st[i]
            title = f"{tag} #{i} {f['day'][5:]}  A x{s['s6h']['exp_over_tri']:.2f}  B x{s['s6h_B']['exp_over_tri']:.2f}  tri off mask {s['s6h']['tri_outside_mask']:.0%}"
            tiles.append(tile(img, r, [("photo", None), ("tri", None), ("mask", r["E"]["s6h"]), ("mask", r["E"]["s6h_B"]), ("mask", r["E"]["live"])], size=190, title=title))
    head = [f"Expanded triangle on the researcher's clicked fins (2025-01-02, 2025-03-18). A = fixed on renders' fin + hump label {p['A']}; B = fixed on the fin-only label {p['B']}.",
            "Per fin: photo | clicked triangle (yellow) | A with the candidate whale mask S6h (red) | B with S6h (nothing below the base line) | A with the live mask.  xN = area over the triangle.",
            "First 40 at random (seed 0), last 8 the worst: 4 that grow most over the triangle, 4 with most of the triangle off the S6h mask."]
    SHEETS.mkdir(parents=True, exist_ok=True)
    out = SHEETS / "triexp_real_examples.jpg"
    cv2.imwrite(str(out), grid_sheet(tiles, 2, head), [cv2.IMWRITE_JPEG_QUALITY, 88])
    (SHEETS / "triexp_real_examples.json").write_text(json.dumps({"random": [F[i]["key"] for i in pick], "grow": [F[i]["key"] for i in by_grow],
                                                                  "off_mask": [F[i]["key"] for i in by_out]}, indent=1))
    print(out)


def cmd_tally(a) -> None:
    p = params()
    F = fins()
    rng = np.random.default_rng(1)
    pick = sorted(rng.choice(len(F), 60, replace=False).tolist())
    (SHEETS / "triexp_tally_sample.json").write_text(json.dumps([F[i]["key"] for i in pick], indent=1))
    for part in range(3):
        tiles = []
        for j, i in enumerate(pick[part * 20:(part + 1) * 20]):
            f = F[i]
            r = apply(f, p)
            img = cv2.imread(str(DAYS / f["day"] / "images" / f"{f['stem']}.jpg"))
            tiles.append(tile(img, r, [("photo", None), ("mask", r["E"]["s6h"]), ("whale", r["masks"]["s6h"]), ("mask", r["E"]["live"]), ("whale", r["masks"]["live"])],
                              size=230, title=f"T{part * 20 + j + 1:02d} #{i} {f['key'][-30:]}"))
        head = ["Tally sample (seed 1): photo | expanded S6h | S6h whale outline (cyan) | expanded live | live whale outline"]
        cv2.imwrite(str(SHEETS / f"triexp_tally_{part + 1}.jpg"), grid_sheet(tiles, 2, head), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("tally sheets written")


def cmd_failures(a) -> None:
    """triexp_failures.jpg = the render failures (render_sheets.py) above the real clicked fins where rule A grows most
    over the triangle (12) or where most of the triangle is off the S6h mask (4), over all 197 fins."""
    p = params()
    st = json.loads((OUT / "real_stats.json").read_text())["rows"]
    F = fins()
    grow = sorted(range(len(F)), key=lambda i: -st[i]["s6h"]["exp_over_tri"])[:12]
    off = sorted([i for i in range(len(F)) if i not in grow], key=lambda i: -st[i]["s6h"]["tri_outside_mask"])[:4]
    tiles = []
    for tag, idx in (("grows most", grow), ("triangle off the mask", off)):
        for i in idx:
            f = F[i]
            r = apply(f, p)
            img = cv2.imread(str(DAYS / f["day"] / "images" / f"{f['stem']}.jpg"))
            s = st[i]
            tiles.append(tile(img, r, [("photo", None), ("tri", None), ("mask", r["E"]["s6h"]), ("whale", r["masks"]["s6h"]), ("mask", r["E"]["s6h_B"])], size=180,
                              title=f"real, {tag}: #{i} {f['day'][5:]} A x{s['s6h']['exp_over_tri']:.2f} tri off mask {s['s6h']['tri_outside_mask']:.0%}"))
    head = ["Real clicked fins where rule A grows most over the triangle (12) or the triangle is most off the S6h mask (4), all 197 fins.",
            "photo | clicked triangle | A on S6h (red) | S6h whale outline (cyan) | B on S6h"]
    real = grid_sheet(tiles, 2, head)
    rend = cv2.imread(str(SHEETS / "triexp_failures_renders.jpg"))
    w = max(real.shape[1], rend.shape[1])
    pad = lambda x: np.pad(x, ((0, 0), (0, w - x.shape[1]), (0, 0)), constant_values=255)  # noqa: E731
    cv2.imwrite(str(SHEETS / "triexp_failures.jpg"), np.vstack([pad(rend), np.full((20, w, 3), 255, np.uint8), pad(real)]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(SHEETS / "triexp_failures.jpg")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["stats", "sheet", "tally-sheet", "failures"])
    a = ap.parse_args()
    {"stats": cmd_stats, "sheet": cmd_sheet, "tally-sheet": cmd_tally, "failures": cmd_failures}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
