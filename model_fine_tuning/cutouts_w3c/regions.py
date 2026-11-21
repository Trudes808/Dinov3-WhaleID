"""The W3c dorsal region of every whale instance, from the class maps eval_w.py wrote (SHADOW/<day>/<model>/hr), no
RAW decode, no GPU.

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.cutouts_w3c.regions train,val --seed s0 [--workers 10]

Per instance key (DAY/parts_up/index.jsonl), as parts_seg/predict.py region_stats finds the fin: dorsal-class pixels
of the hr map inside the instance mask grown by one cse-crop pixel + 2 map px, largest 8-connected region. Written to
CACHE/regions/<seed>/<day>.json: per key
  view      view classifier class (fluke if P(fluke) >= thr, else dorsal if P(dorsal) >= thr, else other)
  up_fin    parts_up has a fin cut-out
  reg       None, or {px (map px), area2560, n_regions, c (centroid, 2560 px), a (map px per 2560 px),
            L_chart / H_chart  extent along / across the parts_up fin cut-out's x axis (only with up_fin),
            L_pca / H_pca      along / across the region's own principal axis,
            L_inst / H_inst    along / across the instance's principal axis,   (2560 px; extent = ptp + 1 map px)
            chart_x / chart_y  [min, max] of the region along the parts_up chart's axes (only with up_fin; added 2026-11-18 pm)}
  A         (only with up_fin) parts_up fin cut-out: centre (2560 px), side (2560 px), mirrored, chart x axis
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np

from . import CACHE, DAYS, MODELS, SHADOW, day_list

UP_SIZE = 1536


def view_of(vc: dict, key: str) -> str:
    thr = float(vc.get("threshold", 0.5)); pf, pd = vc["views"].get(key, (0.0, 0.0))
    return "fluke" if pf >= thr else "dorsal" if pd >= thr else "other"


def ext(P: np.ndarray, u: np.ndarray, pad: float) -> float:
    return float(np.ptp(P @ u)) + pad if len(P) else 0.0


def axis(P: np.ndarray) -> np.ndarray:
    if len(P) < 3:
        return np.array([1.0, 0.0])
    _, _, vt = np.linalg.svd(P - P.mean(0), full_matrices=False)
    return vt[0]


def up_info(day: str, key: str):
    f = DAYS / day / "parts_up" / f"{key}__fin.npz"
    if not f.exists():
        return None
    z = np.load(f); M = z["M"].astype(float); sf = float(z["scale_full"]); sc = float(np.hypot(M[0, 0], M[0, 1]))
    ctr_raw = np.linalg.solve(M[:, :2], np.array([UP_SIZE / 2, UP_SIZE / 2]) - M[:, 2])
    return {"c": (ctr_raw / sf).tolist(), "side": UP_SIZE / sc / sf, "mirrored": bool(z["mirrored"]), "ex": (M[0, :2] / sc).tolist(), "ey": (M[1, :2] / sc).tolist(), "scale_full": sf}


def work(job):
    day, stem, keys, model = job
    from PIL import Image
    from ..common import geometry, paths
    from ..parts_seg.predict import instances_of
    import model_fine_tuning.parts_seg.predict as pr
    cv2.setNumThreads(1)
    pr.DAYS = DAYS
    out = {}
    try:
        table = geometry.load_folded_table(paths.FOLDED_TABLE)
        masks, (x0, y0, s) = instances_of(day, stem, table)
        recs = INST[day]; vc = VC[day]
        for key in keys:
            k = int(key.rsplit("~", 1)[1])
            r = {"view": view_of(vc, key), "p": list(vc["views"].get(key, (0.0, 0.0))), "up_fin": (DAYS / day / "parts_up" / f"{key}__fin.npz").exists(), "reg": None}
            A = up_info(day, key) if r["up_fin"] else None
            if A:
                r["A"] = A
            rec = recs.get(key)
            mp = SHADOW / day / model / "hr" / f"{key}.png"
            if k >= len(masks) or not rec or "hr_window" not in rec or not mp.exists():
                r["no_map"] = True; out[key] = r; continue
            cls = np.asarray(Image.open(mp)); hx, hy, side, osz = rec["hr_window"]; a = osz / side
            H, W = cls.shape
            Mi = np.array([[a / s, 0, (x0 - hx) * a], [0, a / s, (y0 - hy) * a]])
            inst = cv2.warpAffine(masks[k].astype(np.uint8), Mi, (W, H), flags=cv2.INTER_NEAREST) > 0
            rr = int(np.ceil(a / s)) + 2
            near = cv2.dilate(inst.astype(np.uint8), np.ones((2 * rr + 1, 2 * rr + 1), np.uint8)) > 0
            fin = ((cls == 2) & near).astype(np.uint8)
            n, lab, st, _ = cv2.connectedComponentsWithStats(fin, connectivity=8)
            if n > 1:
                j = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
                ys, xs = np.nonzero(lab == j)
                P = np.stack([xs / a + hx, ys / a + hy], 1)
                iy, ix = np.nonzero(masks[k]); Pi = np.stack([ix / s + x0, iy / s + y0], 1)
                g = {"px": int(len(xs)), "area2560": float(len(xs) / a / a), "n_regions": int(n - 1), "c": P.mean(0).tolist(), "a": float(a)}
                pad = 1.0 / a
                for name, u in (("pca", axis(P)), ("inst", axis(Pi))) + ((("chart", np.array(A["ex"])),) if A else ()):
                    v = np.array([-u[1], u[0]])
                    g[f"L_{name}"] = ext(P, u, pad); g[f"H_{name}"] = ext(P, v, pad)
                if A:                                       # region extent along the parts_up chart's x / y axes (2560 px, half a map px of pad)
                    for nm, u in (("x", np.array(A["ex"])), ("y", np.array(A["ey"]))):
                        pr = P @ u; g[f"chart_{nm}"] = [float(pr.min() - pad / 2), float(pr.max() + pad / 2)]
                f = rec.get("hr", {}).get("fin")
                if f:
                    g["c_check"] = float(np.hypot(*(np.array(f["centroid"]) - P.mean(0))))
                r["reg"] = g
            out[key] = r
        return stem, out, None
    except Exception as e:  # noqa: BLE001
        import traceback
        return stem, out, f"{type(e).__name__}: {e} {traceback.format_exc()[-300:]}"


INST: dict = {}
VC: dict = {}


def _init(day, model):
    INST[day] = json.loads((SHADOW / day / model / "instances.json").read_text())["instances"]
    VC[day] = json.loads((DAYS / day / "safe" / "view_class.json").read_text())


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("days"); ap.add_argument("--seed", default="s0", choices=list(MODELS)); ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args(argv); model = MODELS[a.seed]
    od = CACHE / "regions" / a.seed; od.mkdir(parents=True, exist_ok=True)
    for day in day_list(a.days):
        jobs = []
        for l in open(DAYS / day / "parts_up" / "index.jsonl"):
            r = json.loads(l); ks = [i["key"] for i in r.get("instances", [])]
            if ks:
                jobs.append((day, r["stem"], ks, model))
        res, nerr = {}, 0
        with get_context("spawn").Pool(min(a.workers, 10), initializer=_init, initargs=(day, model)) as pool:
            for stem, o, err in pool.imap_unordered(work, jobs, chunksize=4):
                res.update(o)
                if err:
                    nerr += 1; print("  error", stem[-40:], err, flush=True)
        (od / f"{day}.json").write_text(json.dumps(res))
        fins = [r for r in res.values() if r["view"] == "dorsal" and r["up_fin"]]
        print(f"{a.seed} {day}: {len(res)} instances, dorsal-view with parts_up fin {len(fins)}, with W3c region {sum(r['reg'] is not None for r in fins)}, "
              f"no map {sum(bool(r.get('no_map')) for r in res.values())}, errors {nerr}", flush=True)


if __name__ == "__main__":
    main()
