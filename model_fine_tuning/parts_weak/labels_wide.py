"""Widened dorsal label for the run05 renders: the dorsal part = fin AND hump, with the researcher's proportions.

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.parts_weak.labels_wide build [--workers 3]
        -> /var/tmp/sat3737-parts-cache/run05/labels_wide/<id>.png (palette PNGs like parts_seg.labels' cache)
           + labels_wide_summary.json (area new / old, base / height in the image)
    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.parts_weak.labels_wide test --n 40

parts_seg/labels.py is not changed (its default stays the label every existing model and cache uses); this module takes
the cached label of a shot and only re-draws class 2. Definition (user's decision 2026-11-17: the dorsal part covers fin
and hump, "about twice as large"): in the fin's own plane (labels.dorsal_mask: back base B at 0, the base chord B -> F
along e1, height e2 towards the tip, n across), with the rig's fin height h:
  * the front base is moved forward along the back to a_front = max(|F - B|, R x h), R = 2.7 = the median base /
    height of the researcher's 197 clicked fins (2025-01-02 and 2025-03-18; quartiles 1.9 / 3.5; the tip sits at 0.13
    of the base from the back base; the rig's own keypoints give 1.5);
  * dorsal = the old fin label (blade + the rig's hump strip) OR every whale pixel whose posed surface point lies
    between the back base and the new front base (0 <= a <= a_front), within HUMP_BAND x h sideways of the fin plane,
    and no deeper than DEPTH x h below the ridge (the highest surface point of the body's midline strip, |n| < 0.22 x
    the rig's chord, at that position along the back, from the pixels themselves): "down to the back line" with a depth
    so the region is a region and not a line (with the old slab alone it was a line: sheet wide_test.jpg);
  * only body, head or fin pixels change class (fluke, pectoral fin, background, ignore are kept).
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from PIL import Image

from ..parts_seg import labels as L

RUN05 = Path("/home/sat3737/NOAA_whales/Whale-Image-Data-Factory/data/renders/run05")
OLD = Path("/var/tmp/sat3737-parts-cache/run05/labels")
NEW = Path("/var/tmp/sat3737-parts-cache/run05/labels_wide")
R_BASE = 2.7
DEPTH = 1.5                                       # max depth below the ridge, x the fin height, tapering to 0 at the new front base
THIN = 0.15                                       # depth of the strip along the back line behind the blade root, x the fin height
HUMP_BAND = 1.0                                  # sideways limit of the hump part, x the fin height (the old slab is 0.22 x chord)


def plane(meta):
    poly = L.fin_outline(meta)
    if poly is None:
        return None
    front, back = poly[0], poly[-1]
    tip = L._kp(meta, "dorsal_fin_tip")
    e1 = front - back
    chord = float(np.linalg.norm(e1))
    if chord < 1e-3:
        return None
    e1 /= chord
    up = tip - back - np.dot(tip - back, e1) * e1
    h = float(np.linalg.norm(up))
    if h < 1e-3:
        return None
    e2 = up / h
    return back, e1, e2, np.cross(e1, e2), chord, h, float((tip - back) @ e1)


def widen(ds: Path, sid: str, old: np.ndarray, r_base: float = R_BASE, depth: float = DEPTH, hump_band: float = HUMP_BAND):
    meta = json.loads((ds / "meta" / f"{sid}.json").read_text())
    new = old.copy()
    info = {"old_px": int((old == 2).sum())}
    pl = plane(meta)
    if pl is None:
        info["new_px"] = info["old_px"]
        return new, info
    back, e1, e2, n, chord, h, a_tip = pl
    D, ddef = L._xyz(ds / "canon_deformed" / f"{sid}.npz")
    whale = (old == 2) | (old == 4) | (old == 5)
    ok = ddef & whale
    ys, xs = np.nonzero(ok)
    if not len(xs):
        info["new_px"] = info["old_px"]
        return new, info
    P = D[ys, xs].astype(np.float64) - back
    a, b, d = P @ e1, P @ e2, P @ n
    a_front = max(chord, r_base * h)
    band = np.abs(d) < L.FIN_BAND * chord
    inside = (a >= 0) & (a <= a_front)
    body = band & inside & (old[ys, xs] != 2)                # the ridge is read from the midline strip only
    sel = (np.abs(d) < hump_band * h) & inside
    # ridge of the body along the back: highest surface point per 1-cm bin of a (running max over +-3 cm)
    nb = int(np.ceil(a_front)) + 2
    ridge = np.full(nb, -np.inf)
    if body.any():
        ib = np.clip(a[body].astype(int), 0, nb - 1)
        np.maximum.at(ridge, ib, b[body])
        k = 3
        pad = np.concatenate([np.full(k, -np.inf), ridge, np.full(k, -np.inf)])
        ridge = np.max(np.stack([pad[i:i + nb] for i in range(2 * k + 1)]), 0)
    ia = np.clip(a.astype(int), 0, nb - 1)
    # depth: behind the blade's front root only a thin strip along the back line (the clicked triangle ends at the back
    # line under the blade); in front of it the hump part, full depth x h at the root tapering to 0 at the new front
    poly = L.fin_outline(meta)
    a_root = float(np.clip((poly[1] - back) @ e1, 0.0, a_front))
    taper = np.clip((a_front - a) / max(a_front - a_root, 1e-3), 0, 1)
    dep = np.where(a >= a_root, depth * h * taper, THIN * h)
    take = sel & (b >= ridge[ia] - dep)
    new[ys[take], xs[take]] = 2
    info.update(new_px=int((new == 2).sum()), chord_cm=chord, height_cm=h, a_front_cm=a_front, tip_along_new=a_tip / a_front)
    # image-plane proportions: the new front base = the visible surface pixel nearest to the 3D point B + a_front e1
    k = meta["keypoints"]
    if all(k.get(x, {}).get("state") == "visible" and k[x].get("in_frame") for x in ("dorsal_fin_tip", "dorsal_fin_front", "dorsal_fin_back")):
        Fn = back + a_front * e1
        dist = np.linalg.norm(D[ys, xs] - Fn, axis=1)
        j = int(np.argmin(dist))
        if dist[j] < 0.15 * a_front:
            T, B = np.array(k["dorsal_fin_tip"]["uv"]), np.array(k["dorsal_fin_back"]["uv"])
            Fo = np.array(k["dorsal_fin_front"]["uv"])
            F2 = np.array([xs[j], ys[j]], float)

            def boh(F):
                u = F - B
                base = float(np.hypot(*u))
                if base < 2:
                    return None
                u = u / base
                w = (T - B) - ((T - B) @ u) * u
                hh = float(np.hypot(*w))
                return (base / hh, float((T - B) @ u / base)) if hh > 2 else None

            o, w = boh(Fo), boh(F2)
            if o and w:
                info.update(img_base_over_height_old=o[0], img_base_over_height_new=w[0], img_tip_along_new=w[1], new_front_uv=F2.tolist())
    return new, info


def _one(sid):
    ds = RUN05 / "dataset"
    out = NEW / f"{sid}.png"
    try:
        old = np.asarray(Image.open(OLD / f"{sid}.png"))
        new, info = widen(ds, sid, old)
        L.save_palette(new, out)
        return sid, info, None
    except Exception as e:  # noqa: BLE001
        return sid, {}, f"{type(e).__name__}: {e}"


def summarise(infos: list[dict]) -> dict:
    old = np.array([i["old_px"] for i in infos], float)
    new = np.array([i["new_px"] for i in infos], float)
    has = old > 0
    q = lambda v: [round(float(x), 3) for x in np.percentile(v, [25, 50, 75])] if len(v) else None  # noqa: E731
    img = [i for i in infos if "img_base_over_height_new" in i]
    return {"shots": len(infos), "shots_with_fin": int(has.sum()), "total_area_new_over_old": round(float(new[has].sum() / old[has].sum()), 3),
            "per_shot_area_new_over_old_q25_50_75": q(new[has] / old[has]),
            "image_base_over_height_old_q25_50_75": q([i["img_base_over_height_old"] for i in img]),
            "image_base_over_height_new_q25_50_75": q([i["img_base_over_height_new"] for i in img]),
            "image_tip_along_new_q25_50_75": q([i["img_tip_along_new"] for i in img]), "image_n": len(img),
            "clicks_reference": {"base_over_height_q25_50_75": [1.9, 2.7, 3.54], "tip_along_q25_50_75": [0.047, 0.133, 0.238], "n": 197},
            "rule": {"R_base_over_height": R_BASE, "depth_below_ridge_x_height": DEPTH, "ridge_band_x_rig_chord": L.FIN_BAND, "hump_band_x_height": HUMP_BAND}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "test"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args(argv)
    ids = [json.loads(l)["id"] for l in open(RUN05 / "dataset" / "manifest.jsonl") if l.strip()]
    if a.cmd == "test":
        ids = ids[:: max(1, len(ids) // a.n)][: a.n]
    NEW.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    infos, errs = [], []
    with get_context("fork").Pool(a.workers) as pool:
        for k, (sid, info, err) in enumerate(pool.imap_unordered(_one, ids, chunksize=8)):
            if err:
                errs.append((sid, err))
            else:
                infos.append(dict(info, id=sid))
            if (k + 1) % 1000 == 0:
                print(f"{k + 1}/{len(ids)} {time.time() - t0:.0f}s", flush=True)
    s = summarise(infos)
    s.update(errors=len(errs), error_examples=errs[:3], seconds=round(time.time() - t0))
    print(json.dumps(s, indent=1))
    if a.cmd == "build":
        (NEW.parent / "labels_wide_summary.json").write_text(json.dumps(s, indent=1))
        (NEW.parent / "labels_wide_infos.json").write_text(json.dumps(infos))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
