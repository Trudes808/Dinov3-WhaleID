"""By-eye sheets (scratchpad only, never the repo): per instance [window with the predicted points | rectified by
two points (shape kept) | rectified by three points (affine) | today's cut-out v3].

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.pose_side.sheets --pred P.json --name A_val [--days 2025-01-15,...]

Canonical frames. Fin: front base left, back base right, tip up (a fin travelling right is mirrored, as the fin
cut-outs are). Fluke: seen from behind with the stock on top: the whale's left tip on the right, right tip on the left
(the cut-out v3 convention), never mirrored, so a fluke whose notch ends below the tips had its tips called the wrong
way round (or is seen from the front).
Point colours by the predicted state: green visible, blue under water, orange hidden by splash, red behind a body,
grey outside; the ring is thick when P(visible) >= 0.5.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from . import CUT, SHEETS, guard
from .data import KP, STATES, load_jpg

T = 256
COL = [(0, 220, 0), (0, 120, 255), (255, 160, 0), (255, 40, 40), (150, 150, 150)]


def rect(frame, P, part, how):
    if part == "fin":
        src = np.array([P["dorsal_fin_front"], P["dorsal_fin_back"], P["dorsal_fin_tip"]], np.float32)
        dst = np.array([[T * 0.32, T * 0.72], [T * 0.68, T * 0.72], [T * 0.55, T * 0.36]], np.float32)
    else:
        src = np.array([P["fluke_tip_left"], P["fluke_tip_right"], P["fluke_notch"]], np.float32)
        dst = np.array([[T * 0.88, T * 0.5], [T * 0.12, T * 0.5], [T * 0.5, T * 0.42]], np.float32)
    if how == "robust":                                     # transfer.frame_M 'robust', canonical -> tile
        from .transfer import frame_M
        F = frame_M({n: np.asarray(v, float) for n, v in P.items()}, part, "robust")
        G = np.array([[0.36 * T, 0, 0.32 * T], [0, 0.36 * T, 0.72 * T]]) if part == "fin" else np.array([[-0.38 * T, 0, 0.5 * T], [0, -0.38 * T, 0.5 * T]])
        M = G[:, :2] @ F; M[:, 2] += G[:, 2]
    elif how == "aff3":
        M = cv2.getAffineTransform(src, dst)
    else:
        a = complex(*(src[1] - src[0])); b = complex(*(dst[1] - dst[0])); s = b / a
        R = np.array([[s.real, -s.imag], [s.imag, s.real]])
        if part == "fin":
            tip = R @ (src[2] - src[0]) + dst[0]
            if tip[1] > dst[0][1]:                              # tip below the base line: mirror about the base line
                F = np.array([[1, 0], [0, -1.0]]); R = F @ R
        M = np.hstack([R, (dst[0] - R @ src[0])[:, None]])
    return cv2.warpAffine(frame, M, (T, T), flags=cv2.INTER_AREA, borderValue=(40, 40, 40))


def tile_points(frame, P, meta, win):
    cx, cy, side = win; s = T / side; M = np.array([[s, 0, T / 2 - cx * s], [0, s, T / 2 - cy * s]])
    t = cv2.warpAffine(frame, M, (T, T), flags=cv2.INTER_AREA, borderValue=(40, 40, 40))
    for n in KP:
        st = np.array(meta["state"][n]); k = int(st.argmax()); x, y = M[:, :2] @ np.array(P[n]) + M[:, 2]
        if 0 <= x < T and 0 <= y < T:
            cv2.circle(t, (int(x), int(y)), 5, COL[k], 2 if st[0] >= 0.5 else 1)
            cv2.putText(t, n.split("_")[-1][:5], (int(x) + 5, int(y) - 3), 0, 0.33, COL[k], 1)
    return t


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pred", type=Path, required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--days", default="2025-01-15,2025-02-27,2025-03-14"); ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args(); days = guard(a.days.split(","))
    P = json.loads(a.pred.read_text()); meta = json.loads(a.pred.with_name(a.pred.stem + "_meta.json").read_text())["rows"]
    SHEETS.mkdir(parents=True, exist_ok=True)
    for part, cdir in (("fluke", "fluke"), ("fin", "finB")):
        rows = []
        for key, m in meta.items():
            if m["day"] not in days or not (CUT / m["day"] / cdir / f"{key}.jpg").exists():
                continue
            frame = cv2.cvtColor(load_jpg(m["day"], key.rsplit("~", 1)[0]), cv2.COLOR_RGB2BGR)
            pts = {n: np.array(v) for n, v in P[key].items()}
            cut = cv2.resize(cv2.imread(str(CUT / m["day"] / cdir / f"{key}.jpg")), (T, T), interpolation=cv2.INTER_AREA)
            t0 = tile_points(frame, P[key], m, m["window"])
            rows.append(np.hstack([t0, rect(frame, pts, part, "sim2"), rect(frame, pts, part, "robust"), cut]))
            if len(rows) >= a.n:
                break
        if rows:
            out = SHEETS / f"rectified_{a.name}_{part}.jpg"; cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88]); print(out, len(rows))


if __name__ == "__main__":
    main()
