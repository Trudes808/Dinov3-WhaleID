"""Read-only: the body map against the researcher's own clicks on real photos (Whale-Review-Web data/<day>/pair_marks.json:
dorsal fin tip / front base / back base, fluke tips and notch, other whales in the frame). Training days only.

    PYTHONPATH=. .venv/bin/python experiments/bodymap_review/bodymap_clicks.py OUT.json [--days 2025-01-02,2025-03-18]

Per photo with clicks: where each click lands in the map (which whale instance, the body fraction u there, inside what the
cut-out code takes for the dorsal fin / the fluke), how the dorsal-fin cut-out the pipeline makes compares with the fin
the researcher marked (centre, size), the head direction the map gives against the fin's own front and back base, the
map's left / right at the clicked fluke tips, and what became of the other whales the researcher marked.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
DAYS = ROOT / "experiments" / "days_all"; LAB = ROOT.parent / "Whale-Review-Web" / "data"
ap = argparse.ArgumentParser(); ap.add_argument("out", type=Path); ap.add_argument("--days", default="2025-01-02,2025-03-18"); a = ap.parse_args()
from PIL import Image
from model_fine_tuning.common import geometry, paths
from model_fine_tuning.cluster.parts import body_axis_angle, split_instances
t = geometry.load_folded_table(paths.FOLDED_TABLE); X0, LEN = float(t["x_min_m"]), float(t["length_m"])
out = []
for day in a.days.split(","):
    labs = json.loads((LAB / day / "pair_marks.json").read_text()); labs = labs["labels"] if isinstance(labs, dict) else labs
    photos, others = {}, {}
    for l in labs:
        if l.get("verdict") == "skip":
            continue
        for sd in ("a", "b"):
            key = l.get(f"key_{sd}")
            if not key:
                continue
            kp = (l.get("keypoints") or {}).get(sd) or {}
            if kp:
                photos.setdefault(key, {}).update(kp)
            ow = (l.get("other_whales") or {}).get(sd)
            if ow is not None and (kp or ow):
                others[key] = ow                                              # the latest record of the photo wins
    drop = set(json.loads((DAYS / day / "safe" / "not_whale.json").read_text()).get("drop", [])) if (DAYS / day / "safe" / "not_whale.json").exists() else set()
    for key, kp in photos.items():
        stem, k = key.rsplit("~", 1); k = int(k); f = DAYS / day / "cse" / f"{stem}_canon.npz"
        if not f.exists():
            continue
        z = np.load(f); x0, y0, s = z["x0y0s"]; use = z["use"]; signed = z["signed_mm"].astype(np.float32) / 1000.0; u = (signed[..., 0] - X0) / LEN
        W, H = Image.open(DAYS / day / "images" / f"{stem}.jpg").size; masks = split_instances(use, 0.1, u=u)
        if k >= len(masks):
            continue
        pix = [np.stack(np.nonzero(m)[::-1], 1).astype(np.float32) for m in masks]            # (x, y) per instance
        fin_all = (u > 0.36) & (u < 0.54) & (signed[..., 2] > 0.9); fluke_all = (u < 0.12) & (np.abs(signed[..., 1]) > 0.5)

        def at(p):
            """crop px of a click, the nearest whale pixel and what the map says there"""
            c = np.array([(p["x"] * W - x0) * s, (p["y"] * H - y0) * s], np.float32); best = (1e9, -1, None)
            for i, q in enumerate(pix):
                d = np.hypot(*(q - c).T); j = int(d.argmin())
                if d[j] < best[0]:
                    best = (float(d[j]), i, q[j].astype(int))
            d, i, q = best; x, y = int(q[0]), int(q[1])
            return {"c": c.tolist(), "dist": d, "inst": i, "u": float(u[y, x]), "y": float(signed[y, x, 1]), "z": float(signed[y, x, 2]), "fin": bool(fin_all[y, x]), "fluke": bool(fluke_all[y, x])}

        m = masks[k]; ys, xs = np.nonzero(m); long_px = float(max(np.ptp(xs), np.ptp(ys)) + 1)
        rec = {"day": day, "key": key, "n_inst": len([i for i in range(len(masks)) if f"{stem}~{i}" not in drop]), "long_px": long_px, "s": float(s), "pts": {n: at(p) for n, p in kp.items()}}
        fb = m & fin_all
        if fb.sum() >= max(400, 0.03 * m.sum()):                                       # the pipeline makes a dorsal-fin cut-out
            fy, fx = np.nonzero(fb); rec["fin_cut"] = {"c": [float(fx.mean()), float(fy.mean())], "extent": float(max(np.ptp(fx), np.ptp(fy)) + 1), "share": float(fb.sum() / m.sum())}
        rec["head"] = float(body_axis_angle(m, signed))
        rec["others"] = [{n: at(p) for n, p in o.items()} for o in (others.get(key) or [])]
        out.append(rec)
    print(day, len([r for r in out if r["day"] == day]), "photos with clicks", flush=True)
a.out.write_text(json.dumps(out))
