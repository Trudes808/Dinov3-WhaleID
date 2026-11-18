"""The cut-outs against the researcher's clicks (Whale-Review-Web/data/<day>/pair_marks.json; training days 2025-01-02
and 2025-03-18 only). Read-only; prints and writes a small json.

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.cutouts_v3.clickcheck experiments/cutouts_v3/clickcheck.json

Fins (front base, back base and tip clicked): centre of the cut-out from the fin centre (mean of the three clicks) in
fin bases; the fin base's share of the cut-out's width; the tip inside the cut-out; for A (parts_up), B, C.
Flukes (both tips and the notch clicked): in the cut-out, is the whale's LEFT tip on the right (the never-mirrored
view from behind turned stock-up puts it there) and is the notch above the line of the tips (stock on top); for the
old cut-out (parts_up, its vertical flip applied) and v3.
"""
import json, sys
from pathlib import Path

import numpy as np

from . import CACHE, DAYS, ROOT

LAB = ROOT.parent / "Whale-Review-Web" / "data"
CLICK_DAYS = ("2025-01-02", "2025-03-18")


def clicks(day):
    z = json.loads((LAB / day / "pair_marks.json").read_text()); L = z["labels"] if isinstance(z, dict) else z; out = {}
    for l in L:
        if l.get("verdict") == "skip":
            continue
        for sd in ("a", "b"):
            kp = (l.get("keypoints") or {}).get(sd) or {}
            if kp and l.get(f"key_{sd}"):
                out.setdefault(l[f"key_{sd}"], {}).update({k: (v["x"], v["y"]) for k, v in kp.items()})
    return out


def apply(M, p):
    return M[:, :2] @ np.asarray(p, float) + M[:, 2]


def main():
    res = {"fin": {v: [] for v in ("A", "finB", "finC")}, "fluke": {v: [] for v in ("old", "v3")}}
    for day in CLICK_DAYS:
        d = DAYS / day
        for key, kp in clicks(day).items():
            sf = None
            for part in ("body", "fin", "fluke"):
                f = d / "parts_up" / f"{key}__{part}.npz"
                if f.exists():
                    sf = float(np.load(f)["scale_full"]); break
            if sf is None:
                continue
            from PIL import Image
            W, H = Image.open(d / "images" / f"{key.rsplit('~', 1)[0]}.jpg").size
            raw = {k: np.array([x * W * sf, y * H * sf]) for k, (x, y) in kp.items()}
            if all(k in raw for k in ("dorsal_fin_front", "dorsal_fin_back", "dorsal_fin_tip")):
                c = (raw["dorsal_fin_front"] + raw["dorsal_fin_back"] + raw["dorsal_fin_tip"]) / 3; base = np.linalg.norm(raw["dorsal_fin_front"] - raw["dorsal_fin_back"])
                srcs = {"A": (d / "parts_up" / f"{key}__fin.npz", 1536)}
                srcs.update({v: (CACHE / day / v / f"{key}.npz", 768) for v in ("finB", "finC")})
                if not srcs["A"].__getitem__(0).exists() or not (CACHE / day / "finB" / f"{key}.npz").exists():
                    continue                                                 # the same fins in all three
                for v, (f, size) in srcs.items():
                    M = np.load(f)["M"].astype(float); sc = float(np.hypot(M[0, 0], M[0, 1]))
                    ctr = np.linalg.solve(M[:, :2], np.array([size / 2, size / 2]) - M[:, 2])
                    tip = apply(M, raw["dorsal_fin_tip"])
                    res["fin"][v].append(dict(off=float(np.linalg.norm(ctr - c) / base), share=float(base * sc / size),
                                              tip_in=bool((0 <= tip).all() and (tip < size).all()), key=key))
            if all(k in raw for k in ("fluke_tip_left", "fluke_tip_right", "fluke_notch")):
                fo = d / "parts_up" / f"{key}__fluke.npz"; fn = CACHE / day / "fluke" / f"{key}.npz"
                if not fo.exists() or not fn.exists():
                    continue
                for v, f, size in (("old", fo, 1536), ("v3", fn, 768)):
                    z = np.load(f); M = z["M"].astype(float)
                    L, R, N = (apply(M, raw[k]) for k in ("fluke_tip_left", "fluke_tip_right", "fluke_notch"))
                    if v == "old" and bool(z["flipped_v"]):
                        L[1], R[1], N[1] = size - L[1], size - R[1], size - N[1]
                    res["fluke"][v].append(dict(left_on_right=bool(L[0] > R[0]), notch_up=bool(N[1] < (L[1] + R[1]) / 2), key=key))
    out = {}
    for v, rr in res["fin"].items():
        if rr:
            off = np.array([r["off"] for r in rr]); sh = np.array([r["share"] for r in rr])
            out[f"fin_{v}"] = dict(n=len(rr), centre_off_median=round(float(np.median(off)), 3), off_over_1=round(float(np.mean(off > 1)), 3), off_over_half=round(float(np.mean(off > 0.5)), 3),
                                   base_share_median=round(float(np.median(sh)), 3), base_share_q=[round(float(x), 3) for x in np.percentile(sh, [25, 75])], tip_inside=round(float(np.mean([r["tip_in"] for r in rr])), 3))
    for v, rr in res["fluke"].items():
        if rr:
            out[f"fluke_{v}"] = dict(n=len(rr), left_tip_on_right=round(float(np.mean([r["left_on_right"] for r in rr])), 3), notch_above_tips=round(float(np.mean([r["notch_up"] for r in rr])), 3),
                                     consistent=round(float(np.mean([r["left_on_right"] and r["notch_up"] for r in rr])), 3))
    print(json.dumps(out, indent=1))
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
