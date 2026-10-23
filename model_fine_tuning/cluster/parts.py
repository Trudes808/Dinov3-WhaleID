"""Pose-normalised part charts for identity matching: dorsal fin, fluke, and the whole visible
flank, cut from the full-resolution RAW with the canonical map.

    python -m model_fine_tuning.cluster.parts --index INDEX_JSONL --canon-dir PRED --out PARTS [--size 512] [--workers 10]

Per photo, up to three charts (jpg + npz with masks and view data):
  fin    the dorsal fin and its surround: mirrored to the right-flank view, head right, fin up,
         scaled to the fin box (chart.py), plus the fin silhouette isolated from the mask
  fluke  the fluke blades: rotated so the span is horizontal and the peduncle is at the top,
         mirrored so the whale's right tip is on the image right (from the resolved side map),
         scaled to the span; "underside" or "top" from the canonical height of the visible surface
  body   the whole visible whale: mirrored to the right-flank view, head right, scaled to its
         extent (flank scars, pigmentation, the general case)
Each npz carries view data used at match time: view_side, profile-ness (|nose_right| of the
direct rotation head), the part's pixel count, and the transform from the 2560-px image.

--instances: every connected component of the whale mask that is at least --min-instance-frac of the largest
(and 400 px) is a whale instance of its own (pairs, competitive groups, mother and calf), charted separately as
<stem>~k (k = 0 the largest), unless the components can be one whale: a component joins an instance whose body
coverage (canonical body fraction from the CSE map) it does not duplicate, so a back and a fluke in separate blobs
stay one whale and only two backs (two flukes, two heads) become two instances. The index records each instance's
area and bounding box in the 2560-px image so that cluster/tracklets.py can follow a whale through a burst and keep
two whales of one photo apart.
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from ..common import geometry, paths
from .chart import decode_full
from .finshape import isolate_fin


def body_axis_angle(body_crop: np.ndarray, signed_m: np.ndarray) -> float:
    """Direction of the head in the crop (degrees, y down): principal axis of the visible whale,
    signed by the body-fraction gradient along it."""
    by, bx = np.nonzero(body_crop)
    pts = np.stack([bx, by], 1).astype(float); pts -= pts.mean(0)
    _, _, vt = np.linalg.svd(pts, full_matrices=False)
    axis = vt[0]
    proj = pts @ axis
    c = np.corrcoef(proj, signed_m[by, bx, 0])[0, 1] if proj.std() > 1e-6 else 1.0
    h = axis * (1.0 if c >= 0 else -1.0)
    return float(np.degrees(np.arctan2(h[1], h[0])))


def make_transform(centre_crop, extent_crop, angle_deg, target_deg, x0, y0, s, scale_full, size, margin):
    cx, cy = centre_crop
    fx, fy = (cx / s + x0) * scale_full, (cy / s + y0) * scale_full
    out_scale = size / (extent_crop / s * scale_full * (1 + 2 * margin))
    M = cv2.getRotationMatrix2D((fx, fy), angle_deg - target_deg, out_scale)
    M[:, 2] += np.array([size / 2 - fx, size / 2 - fy])
    return M, float(out_scale)


def warp_masks(M, x0, y0, s, scale_full, size, *masks):
    crop_to_full = np.array([[scale_full / s, 0, x0 * scale_full], [0, scale_full / s, y0 * scale_full]], np.float64)
    M3 = (np.vstack([M, [0, 0, 1]]) @ np.vstack([crop_to_full, [0, 0, 1]]))[:2]
    return [cv2.warpAffine(m.astype(np.uint8), M3, (size, size), flags=cv2.INTER_NEAREST) > 0 for m in masks]


U_BINS = np.linspace(0.0, 1.0, 11)


def split_instances(use: np.ndarray, min_frac: float = 0.1, min_px: int = 400, u: np.ndarray | None = None, overlap_px: int = 200) -> list[np.ndarray]:
    """Whale instances from the mask components (largest first, at least min_frac of the largest and min_px).
    With the canonical body fraction u: a component joins the instance it can belong to, i.e. one whose body coverage
    it does not duplicate (a back and a fluke in separate blobs stay one whale; two backs are two whales: both hold
    >= overlap_px pixels in a shared u bin). Without u every component is its own instance."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(use.astype(np.uint8), connectivity=8)
    if n <= 1:
        return []
    areas = st[1:, cv2.CC_STAT_AREA]; order = np.argsort(-areas)
    comps = [lab == (k + 1) for k in order if areas[k] >= max(min_px, min_frac * areas[order[0]])]
    if u is None or len(comps) <= 1:
        return comps
    inst: list[np.ndarray] = []; hist: list[np.ndarray] = []
    for m in comps:
        h = np.histogram(np.clip(u[m], 0, 1), bins=U_BINS)[0]
        for i in range(len(inst)):
            if not np.any((hist[i] >= overlap_px) & (h >= overlap_px)):
                inst[i] = inst[i] | m; hist[i] = hist[i] + h
                break
        else:
            inst.append(m.copy()); hist.append(h)
    return inst


def extract_photo(rec: dict, canon_dir: Path, out: Path, table: dict, size: int, fin_min_fraction: float = 0.0, fluke_min_fraction: float = 0.0,
                  instances: bool = False, min_instance_frac: float = 0.1, only_parts: tuple = ("body", "fin", "fluke")) -> dict:
    """fin_min_fraction / fluke_min_fraction: besides the 400-pixel minimum, the part must hold this share of the
    whale's pixels (a patch of mislabelled pixels on a back must not produce a fluke chart)."""
    stem = Path(rec["image"]).stem
    z = np.load(canon_dir / f"{stem}_canon.npz")
    x0, y0, s = z["x0y0s"]
    use_all, side = z["use"], z["side"]
    signed = z["signed_mm"].astype(np.float32) / 1000.0
    if use_all.sum() < 400:
        return {"stem": stem, "parts": [], "instances": []}
    pose = rec.get("pose") or {}
    R = np.array(pose["rotation_direct"]) if pose.get("rotation_direct") else None
    profile = float(abs(R[1, 0])) if R is not None else float("nan")      # |nose_right|
    full = decode_full(rec["source"])
    H, W = full.shape[:2]
    img2560 = cv2.imread(str(canon_dir.parent / "images" / f"{stem}.jpg"))
    scale_full = W / img2560.shape[1]
    u = (signed[..., 0] - table["x_min_m"]) / table["length_m"]
    masks = split_instances(use_all, min_instance_frac, u=u) if instances else [use_all]
    inst_recs = []; main_parts = []
    for k, use in enumerate(masks):
        key = f"{stem}~{k}" if instances else stem
        made = extract_instance(key, use, side, signed, u, profile, full, x0, y0, s, scale_full, out, size, fin_min_fraction, fluke_min_fraction, only_parts)
        ys, xs = np.nonzero(use)
        bbox = [float(xs.min() / s + x0), float(ys.min() / s + y0), float((xs.max() + 1) / s + x0), float((ys.max() + 1) / s + y0)]   # 2560-px image
        inst_recs.append({"key": key, "parts": made, "area": int(use.sum()), "bbox_2560": [round(v, 1) for v in bbox]})
        if k == 0:
            main_parts = made
    return {"stem": stem, "parts": main_parts, "instances": inst_recs}


def extract_instance(stem: str, use: np.ndarray, side: np.ndarray, signed: np.ndarray, u: np.ndarray, profile: float, full: np.ndarray,
                     x0: float, y0: float, s: float, scale_full: float, out: Path, size: int, fin_min_fraction: float, fluke_min_fraction: float,
                     only_parts: tuple = ("body", "fin", "fluke")) -> list[str]:
    head_angle = body_axis_angle(use, signed)
    made = []

    def save(name, chart, masks: dict, meta: dict):
        cv2.imwrite(str(out / "charts" / f"{stem}__{name}.jpg"), cv2.cvtColor(chart, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        np.savez_compressed(out / f"{stem}__{name}.npz", **masks, **meta, profile=profile, stem=stem, part=name)
        made.append(name)

    # --- body: whole visible whale, right-flank view, head right
    ys, xs = np.nonzero(use)
    view_side = int(np.sign(side[use].sum())) if np.abs(side[use]).sum() else 0
    target = 0.0 if view_side >= 0 else 180.0
    if "body" in only_parts:
        M, sc = make_transform(((xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2), max(np.ptp(xs), np.ptp(ys)) + 1.0, head_angle, target, x0, y0, s, scale_full, size, 0.08)
        chart = cv2.warpAffine(full, M, (size, size), flags=cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC)
        (whale_c,) = warp_masks(M, x0, y0, s, scale_full, size, use)
        mirrored = view_side < 0
        if mirrored:
            chart, whale_c = chart[:, ::-1].copy(), whale_c[:, ::-1].copy()
        save("body", chart, {"whale": whale_c}, {"view_side": view_side, "mirrored": mirrored, "pixels": int(use.sum()), "M": M, "x0y0s": np.array([x0, y0, s]), "scale_full": scale_full})

    # --- fin: as chart.py, then the silhouette
    fin = use & (u > 0.36) & (u < 0.54) & (signed[..., 2] > 0.9)
    if "fin" in only_parts and fin.sum() >= max(400, fin_min_fraction * use.sum()):
        fy, fx = np.nonzero(fin)
        fside = int(np.sign(side[fin].sum())) if np.abs(side[fin]).sum() else view_side
        target = 0.0 if fside >= 0 else 180.0
        M, sc = make_transform((fx.mean(), fy.mean()), max(np.ptp(fx), np.ptp(fy)) + 1.0, head_angle, target, x0, y0, s, scale_full, size, 0.45)
        chart = cv2.warpAffine(full, M, (size, size), flags=cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC)
        whale_c, fin_c = warp_masks(M, x0, y0, s, scale_full, size, use, fin)
        mirrored = fside < 0
        if mirrored:
            chart, whale_c, fin_c = chart[:, ::-1].copy(), whale_c[:, ::-1].copy(), fin_c[:, ::-1].copy()
        sil = isolate_fin(whale_c)
        save("fin", chart, {"whale": whale_c, "fin_band": fin_c, "silhouette": sil if sil is not None else np.zeros_like(whale_c)},
             {"view_side": fside, "mirrored": mirrored, "pixels": int(fin.sum()), "M": M, "x0y0s": np.array([x0, y0, s]), "scale_full": scale_full})

    # --- fluke: blades, span horizontal, peduncle up, whale's right tip on the image right
    fluke = use & (u < 0.12) & (np.abs(signed[..., 1]) > 0.5)
    if "fluke" in only_parts and fluke.sum() >= max(400, fluke_min_fraction * use.sum()):
        ky, kx = np.nonzero(fluke)
        pts = np.stack([kx, ky], 1).astype(float); pts -= pts.mean(0)
        _, _, vt = np.linalg.svd(pts, full_matrices=False)
        span = vt[0]                                                     # the blade span direction in the crop
        # which end of the span is the whale's right tip: correlate the span coordinate with signed y (+ = right)
        c = np.corrcoef(pts @ span, signed[ky, kx, 1])[0, 1] if len(ky) > 10 else 1.0
        right = span * (1.0 if c >= 0 else -1.0)
        angle = float(np.degrees(np.arctan2(right[1], right[0])))         # rotate so the right tip points to +x
        M, sc = make_transform((kx.mean(), ky.mean()), max(np.ptp(kx), np.ptp(ky)) + 1.0, angle, 0.0, x0, y0, s, scale_full, size, 0.15)
        chart = cv2.warpAffine(full, M, (size, size), flags=cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC)
        whale_c, fluke_c = warp_masks(M, x0, y0, s, scale_full, size, use, fluke)
        # peduncle (increasing body fraction) should be at the top: if the body-fraction gradient points down, flip vertically
        A = np.stack([kx, ky, np.ones_like(kx)], 1).astype(float)
        g = np.linalg.lstsq(A, u[ky, kx], rcond=None)[0][:2]
        g_chart = (np.array([[M[0, 0], M[0, 1]], [M[1, 0], M[1, 1]]]) @ g)
        if g_chart[1] > 0:
            chart, whale_c, fluke_c = chart[::-1].copy(), whale_c[::-1].copy(), fluke_c[::-1].copy()
            flipped_v = True
        else:
            flipped_v = False
        underside = bool(np.mean(signed[ky, kx, 2]) < -0.3)
        save("fluke", chart, {"whale": whale_c, "fluke": fluke_c},
             {"view_side": 0, "mirrored": False, "flipped_v": flipped_v, "underside": underside, "pixels": int(fluke.sum()), "M": M, "x0y0s": np.array([x0, y0, s]), "scale_full": scale_full})
    return made


def _worker(args):
    rec, canon_dir, out, size, fin_min, fluke_min, instances, min_frac, only_parts = args
    table = geometry.load_folded_table(paths.FOLDED_TABLE)
    try:
        return extract_photo(rec, Path(canon_dir), Path(out), table, size, fin_min, fluke_min, instances, min_frac, only_parts)
    except Exception as e:  # noqa: BLE001
        return {"stem": Path(rec["image"]).stem, "error": str(e), "parts": [], "instances": []}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", type=Path, required=True, help="images/index.jsonl from convert_raw")
    p.add_argument("--canon-dir", type=Path, required=True, help="canon.predict output (<stem>_canon.npz, points.jsonl)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--fin-min-fraction", type=float, default=0.0, help="a fin chart also needs this share of the whale's pixels in the fin band")
    p.add_argument("--fluke-min-fraction", type=float, default=0.0, help="a fluke chart also needs this share of the whale's pixels on the blades")
    p.add_argument("--instances", action="store_true", help="chart every whale instance (mask component) separately as <stem>~k")
    p.add_argument("--min-instance-frac", type=float, default=0.1)
    p.add_argument("--update-multi", action="store_true", help="existing output: redo only the photos whose mask has several components, keep the rest")
    p.add_argument("--only-parts", default="body,fin,fluke", help="comma list of the chart types to cut (e.g. fin,fluke at native resolution)")
    args = p.parse_args(argv)
    recs = [json.loads(l) for l in open(args.index)]
    keep_rows = {}
    if args.update_multi and (args.out / "index.jsonl").exists():
        old_rows = {json.loads(l)["stem"]: json.loads(l) for l in open(args.out / "index.jsonl")}
        redo = set()
        for r in recs:
            stem = Path(r["image"]).stem; f = args.canon_dir / f"{stem}_canon.npz"
            if f.exists() and len(split_instances(np.load(f)["use"], args.min_instance_frac)) > 1:
                redo.add(stem)
        keep_rows = {k: v for k, v in old_rows.items() if k not in redo}
        for stem in redo:
            for f in list(args.out.glob(f"{stem}~*__*.npz")) + list((args.out / "charts").glob(f"{stem}~*__*.jpg")):
                f.unlink()
        recs = [r for r in recs if Path(r["image"]).stem in redo]
        print(f"update-multi: redoing {len(recs)} photos with several components, keeping {len(keep_rows)}", flush=True)
    poses = {Path(json.loads(l)["image"]).stem: json.loads(l) for l in open(args.canon_dir / "points.jsonl")} if (args.canon_dir / "points.jsonl").exists() else {}
    for r in recs:
        r["pose"] = poses.get(Path(r["image"]).stem)
    args.out.mkdir(parents=True, exist_ok=True); (args.out / "charts").mkdir(exist_ok=True)
    only = tuple(x for x in args.only_parts.split(",") if x)
    jobs = [(r, str(args.canon_dir), str(args.out), args.size, args.fin_min_fraction, args.fluke_min_fraction, args.instances, args.min_instance_frac, only)
            for r in recs if (args.canon_dir / f"{Path(r['image']).stem}_canon.npz").exists()]
    counts, errors, multi = {"body": 0, "fin": 0, "fluke": 0}, 0, 0
    with Pool(args.workers) as pool, open(args.out / "index.jsonl.tmp", "w") as handle:
        for row in keep_rows.values():
            handle.write(json.dumps(row) + "\n")
        for res in pool.imap_unordered(_worker, jobs, chunksize=2):
            if "error" in res:
                errors += 1; print("error", res["stem"], res["error"][:100], flush=True)
            for inst in res.get("instances") or [{"parts": res["parts"]}]:
                for n in inst["parts"]:
                    counts[n] += 1
            multi += len(res.get("instances", [])) > 1
            handle.write(json.dumps(res) + "\n")
    (args.out / "index.jsonl.tmp").replace(args.out / "index.jsonl")
    print(f"parts: {counts} ({errors} errors, {multi} photos with several instances) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
