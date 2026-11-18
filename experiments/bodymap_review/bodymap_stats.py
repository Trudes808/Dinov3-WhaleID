"""Read-only: what the body map (DAY/cse/<stem>_canon.npz) says per whale instance on real photos, for a comparison of
photos with one whale against photos with several. No model is run; nothing of a day is changed.

    PYTHONPATH=. .venv/bin/python experiments/bodymap_review/bodymap_stats.py OUT.json [--days train,val] [--workers 8]

Per photo: mask blobs, instances (cluster/parts.py split_instances, as parts_hr), the map's own mirror test
(points.jsonl), the catalogue's group type. Per instance: size in the network's 512-px crop, how well the body
fraction runs along the blob's long axis, which way the head points against the flank the map says is showing
(a whale at the surface is upright, so "head left + right flank" is an error), the mirror disagreement inside the
instance, the part flags the cut-outs came out with and the view classifier's fluke / dorsal answer.
"""
import argparse, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
DAYS = ROOT / "experiments" / "days_all"


def one(job):
    day, stem, rec, pt, table = job
    import cv2
    from model_fine_tuning.cluster.parts import body_axis_angle, split_instances
    try:
        z = np.load(DAYS / day / "cse" / f"{stem}_canon.npz"); use = z["use"]; side = z["side"]; signed = z["signed_mm"].astype(np.float32) / 1000.0
        dis = z["disagree_mm"].astype(np.float32) / 1000.0 if "disagree_mm" in z.files else None
        x0, y0, s = z["x0y0s"]
    except Exception as e:  # noqa: BLE001
        return {"day": day, "stem": stem, "error": str(e)[:80]}
    if use.sum() < 400:
        return {"day": day, "stem": stem, "blobs": 0, "inst": []}
    u = (signed[..., 0] - table["x_min_m"]) / table["length_m"]
    n, lab, st, _ = cv2.connectedComponentsWithStats(use.astype(np.uint8), connectivity=8)
    areas = sorted((int(a) for a in st[1:, cv2.CC_STAT_AREA]), reverse=True)
    blobs = [a for a in areas if a >= max(400, 0.1 * areas[0])]
    masks = split_instances(use, 0.1, u=u)
    out = {"day": day, "stem": stem, "blobs": len(blobs), "small_blobs": len(areas) - len(blobs), "n_inst": len(masks), "n_index": len(rec.get("instances", [])), "s": float(s),
           "crop_fill": float(use.mean()), "flip_m": pt.get("flip_disagreement_median_m"), "side_flip": pt.get("side_flip_agreement"), "rot_flip_deg": pt.get("rot_direct_flip_consistency_deg"), "inst": []}
    for k, m in enumerate(masks):
        ys, xs = np.nonzero(m); pts = np.stack([xs, ys], 1).astype(float); c = pts.mean(0); pts -= c
        _, sv, vt = np.linalg.svd(pts[:: max(1, len(pts) // 4000)], full_matrices=False); proj = pts @ vt[0]
        uu = u[ys, xs]; corr = float(abs(np.corrcoef(proj, uu)[0, 1])) if proj.std() > 1e-6 and uu.std() > 1e-6 else 0.0
        head = body_axis_angle(m, signed); sd = int(np.sign(side[m].sum())) if np.abs(side[m]).sum() else 0
        nz = side[m][side[m] != 0]; pur = float(max((nz > 0).mean(), (nz < 0).mean())) if len(nz) else None
        turn = (head - (0.0 if sd >= 0 else 180.0) + 180.0) % 360.0 - 180.0
        fin = m & (u > 0.36) & (u < 0.54) & (signed[..., 2] > 0.9); fluke = m & (u < 0.12) & (np.abs(signed[..., 1]) > 0.5)
        nb = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)[0] - 1
        # how many separate dorsal-fin patches of some size the map marks inside one instance (two = probably two whales in it)
        nf, labf, stf, _ = cv2.connectedComponentsWithStats(fin.astype(np.uint8), connectivity=8); fa = sorted((int(a) for a in stf[1:, cv2.CC_STAT_AREA]), reverse=True)
        fin_patches = len([a for a in fa if a >= max(150, 0.25 * fa[0])]) if fa else 0
        q = np.percentile(uu, [5, 50, 95])
        out["inst"].append({"k": k, "area": int(m.sum()), "long_px": float(max(np.ptp(xs), np.ptp(ys)) + 1), "elong": float(sv[0] / max(sv[1], 1e-6)), "axis_deg": float(np.degrees(np.arctan2(vt[0][1], vt[0][0]))),
                            "u_corr": corr, "u5": float(q[0]), "u50": float(q[1]), "u95": float(q[2]), "head": float(head), "side": sd, "side_purity": pur, "turn": float(turn),
                            "flip_m": float(np.median(dis[m])) if dis is not None else None, "fin_px": int(fin.sum()), "fluke_px": int(fluke.sum()), "fin_patches": fin_patches, "pieces": int(nb),
                            "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]})
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out", type=Path); ap.add_argument("--days", default="train,val"); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    from model_fine_tuning.common import geometry, paths
    t = geometry.load_folded_table(paths.FOLDED_TABLE); table = {"x_min_m": float(t["x_min_m"]), "length_m": float(t["length_m"])}
    split = json.loads((ROOT / "model_fine_tuning/splits/days.json").read_text()); L = json.loads((ROOT / "model_fine_tuning/splits/labels_all_days.json").read_text())["days"]
    days = [d for x in a.days.split(",") for d in (split[x] if x in split else [x])]
    rows = []
    with Pool(a.workers) as pool:
        for day in days:
            d = DAYS / day
            if not (d / "parts_hr" / "index.jsonl").exists():
                continue
            index = {json.loads(l)["stem"]: json.loads(l) for l in open(d / "parts_hr" / "index.jsonl")}
            pts = {Path(json.loads(l)["image"]).stem: json.loads(l) for l in open(d / "cse" / "points.jsonl")} if (d / "cse" / "points.jsonl").exists() else {}
            drop = set(json.loads((d / "safe" / "not_whale.json").read_text()).get("drop", [])) if (d / "safe" / "not_whale.json").exists() else set()
            vc = json.loads((d / "safe" / "view_class.json").read_text())["views"] if (d / "safe" / "view_class.json").exists() else {}
            cat = L.get(day, {}); photos = cat.get("photos", {}); groups = cat.get("groups", {})
            res = pool.map(one, [(day, stem, rec, pts.get(stem, {}), table) for stem, rec in index.items()], chunksize=8)
            for r in res:
                if "inst" not in r:
                    rows.append(r); continue
                rec = index[r["stem"]]; animal = photos.get(r["stem"].split("__")[-1]); g = animal.split("_")[-1][:3] if animal else None
                r["animal"] = animal; r["gtype"] = (groups.get(g) or ("O" if g and g.startswith("O") else "?")) if animal else None
                for i, x in enumerate(r["inst"]):
                    ir = rec["instances"][i] if i < len(rec.get("instances", [])) else {}
                    x["key"] = ir.get("key"); x["parts"] = ir.get("parts", []); x["not_whale"] = ir.get("key") in drop; x["vc"] = vc.get(ir.get("key"))
                rows.append(r)
            print(day, len(res), flush=True)
    a.out.write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
