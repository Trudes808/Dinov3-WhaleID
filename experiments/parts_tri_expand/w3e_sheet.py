"""triA_model_examples.jpg (scratchpad only): W3c, W3d, W3e (seed 0) on the same 30 held-out clicked fins of 2025-03-18
as triexp_model_examples.jpg (same key list and seed), and on 20 render fins of r06r_check21 (seed 0; the per-whale
"hr" windows of w3e_scores.py, as real photos are run).

    PYTHONPATH=. .venv/bin/python experiments/parts_tri_expand/w3e_sheet.py

Real rows: photo | W3c | W3d | W3e (red = predicted dorsal) | rule-A truth on S6h (green); yellow = clicked triangle.
Render rows: render | W3c | W3d | W3e | NEW render label (rule A, green) with the OLD label outlined in magenta.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "parts_weak"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_fine_tuning.parts_weak import tri_expand as TE  # noqa: E402
import pred_fill as PF  # noqa: E402
import w3e_scores as WS  # noqa: E402

SHEETS = Path("/tmp/claude-1072903897/-home-sat3737-NOAA-whales/4fbba7ff-3ba5-4b50-9cb7-970c81250409/scratchpad/triA/sheets")
ARMS = ["parts_seg_r05_w3c", "parts_seg_r05_w3d", "parts_seg_r05_w3e"]
OLD_ARMS = ["parts_seg_r05_w3c", "parts_seg_r05_w3c_fill", "parts_seg_r05_w3d"]      # the earlier sheet's key filter
Z = 200


def overlay(t, m, col):
    ov = t.copy()
    ov[m] = col
    return cv2.addWeighted(t, 0.55, ov, 0.45, 0)


def real_rows():
    from fin_definition import clicks
    p = PF.rule()
    day = "2025-03-18"
    ins_old = {m: json.loads((PF.SHADOW / day / m / "instances.json").read_text())["instances"] for m in OLD_ARMS}
    keys = [k for k in sorted(clicks(day)) if all(k in ins_old[m] and "hr_window" in ins_old[m][k] for m in OLD_ARMS)]
    ck = clicks(day)
    pick = sorted(np.random.default_rng(0).choice(len(keys), 30, replace=False).tolist())
    ins = {m: json.loads((PF.SHADOW / day / m / "instances.json").read_text())["instances"] for m in ARMS}
    tiles = []
    for i in pick:
        key = keys[i]
        stem = key.rsplit("~", 1)[0]
        kp = ck[key]
        M = PF.s6h(day, stem)
        H, W = M.shape
        T, F, B = (np.array([kp[x]["x"] * W, kp[x]["y"] * H]) for x in ("dorsal_fin_tip", "dorsal_fin_front", "dorsal_fin_back"))
        E = TE.expand(T, F, B, M, **p)
        hx, hy, side, osz = ins[ARMS[0]][key]["hr_window"]
        aa, s = osz / side, 2560.0 / max(H, W)
        A = np.array([[aa * s, 0, -hx * aa], [0, aa * s, -hy * aa]])
        img = cv2.warpAffine(cv2.imread(str(REPO / "experiments" / "days_all" / day / "images" / f"{stem}.jpg")),
                             np.array([[aa, 0, -hx * aa], [0, aa, -hy * aa]]), (int(osz), int(osz)), flags=cv2.INTER_AREA)
        q = np.round(np.stack([(x * s - [hx, hy]) * aa for x in (T, F, B)])).astype(np.int32)
        fr = TE.frame(*q.astype(float))
        c = q.mean(0)
        L = max(np.ptp(q[:, 0]), np.ptp(q[:, 1]), fr["h_eff"]) * 2.4 + 16
        Zm = np.array([[Z / L, 0, -(c[0] - L / 2) * Z / L], [0, Z / L, -(c[1] - L / 2) * Z / L]])
        layers = [None] + [(np.asarray(Image.open(PF.SHADOW / day / m / "hr" / f"{key}.png")) == PF.FIN) for m in ARMS] + \
                 [cv2.warpAffine(E.astype(np.uint8), A, (int(osz), int(osz)), flags=cv2.INTER_NEAREST) > 0]
        row = []
        for j, Lr in enumerate(layers):
            t = cv2.warpAffine(img, Zm, (Z, Z), flags=cv2.INTER_LINEAR)
            if Lr is not None:
                m = cv2.warpAffine(Lr.astype(np.uint8), Zm, (Z, Z), flags=cv2.INTER_NEAREST) > 0
                t = overlay(t, m, (40, 40, 255) if j < 4 else (60, 200, 60))
                qq = np.round(np.c_[q, np.ones(3)] @ Zm.T).astype(np.int32)
                cv2.polylines(t, [qq], True, (0, 255, 255), 1, cv2.LINE_AA)
            row.append(np.pad(t, ((2, 2), (2, 2), (0, 0)), constant_values=255))
        r = np.hstack(row)
        lab = np.full((16, r.shape[1], 3), 255, np.uint8)
        cv2.putText(lab, f"real {key[-34:]}", (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(np.vstack([lab, r]))
    return tiles


def render_rows(n=20):
    F = WS.fins()
    pick = sorted(np.random.default_rng(0).choice(len(F), n, replace=False).tolist())
    ds = WS.FAC / WS.ST / "dataset"
    wins = {m: json.loads((WS.CACHE / "render_parts_hr" / m / f"{WS.ST}_windows.json").read_text()) for m in ARMS}
    tiles = []
    for i in pick:
        f = F[i]
        sid, k = f["sid"], f["k"]
        img = cv2.imread(str(ds / "images" / f"{sid}.png"))
        inst, parts, did = WS.rd(ds / "instances" / f"{sid}.png"), WS.rd(ds / "parts" / f"{sid}.png"), WS.rd(ds / "dorsal_triA" / f"{sid}.png")
        fr = TE.frame(f["T"], f["F"], f["B"])
        pts = np.stack([f["T"], f["F"], f["B"]])
        c = (pts.min(0) + pts.max(0)) / 2
        L = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), fr["h_eff"]) * 2.4 + 16
        x0, y0 = int(np.floor(c[0] - L / 2)), int(np.floor(c[1] - L / 2))
        x1, y1 = x0 + int(np.ceil(L)), y0 + int(np.ceil(L))
        Zm = np.array([[Z / (x1 - x0), 0, 0], [0, Z / (y1 - y0), 0]])

        def crop(a, interp=cv2.INTER_NEAREST):
            pad = cv2.copyMakeBorder(a, 2000, 2000, 2000, 2000, cv2.BORDER_CONSTANT, value=0)
            return cv2.resize(pad[y0 + 2000:y1 + 2000, x0 + 2000:x1 + 2000], (Z, Z), interpolation=interp)
        base = crop(img, cv2.INTER_LINEAR)
        own = inst == k
        q = np.round(np.c_[pts - [x0, y0], np.ones(3)] @ Zm.T).astype(np.int32)
        row = [base]
        for m in ARMS:
            cls = cv2.imread(str(WS.CACHE / "render_parts_hr" / m / WS.ST / f"{sid}~{k}.png"), cv2.IMREAD_UNCHANGED)
            full = WS.to2560(cls, wins[m][f"hr|{sid}~{k}"], None, x0, y0, x1, y1)
            t = overlay(base, cv2.resize(((full == PF.FIN)).astype(np.uint8), (Z, Z), interpolation=cv2.INTER_NEAREST) > 0, (40, 40, 255))
            cv2.polylines(t, [q], True, (0, 255, 255), 1, cv2.LINE_AA)
            row.append(t)
        t = overlay(base, crop(((did == k)).astype(np.uint8)) > 0, (60, 200, 60))
        oldm = crop((((parts == 2) & own)).astype(np.uint8))
        cnt, _ = cv2.findContours(oldm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(t, cnt, -1, (255, 0, 255), 1)
        cv2.polylines(t, [q], True, (0, 255, 255), 1, cv2.LINE_AA)
        row.append(t)
        r = np.hstack([np.pad(x, ((2, 2), (2, 2), (0, 0)), constant_values=255) for x in row])
        lab = np.full((16, r.shape[1], 3), 255, np.uint8)
        cv2.putText(lab, f"render {sid}:{k} view {f['sight_deg']:.0f} deg", (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(np.vstack([lab, r]))
    return tiles


def main() -> int:
    from real_fins import grid_sheet
    tiles = real_rows() + render_rows()
    head = ["Rows 1-30: held-out clicked fins 2025-03-18 (same 30 as triexp_model_examples.jpg). Rows 31-50: r06r_check21 render fins, seed 0, per-whale hr window.",
            "photo | W3c | W3d | W3e (red = predicted dorsal, seed-0 models) | truth: real = rule A on S6h (green); render = NEW label (green), OLD label outline magenta; yellow = fin triangle"]
    SHEETS.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(SHEETS / "triA_model_examples.jpg"), grid_sheet(tiles, 2, head), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(SHEETS / "triA_model_examples.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
