"""Part charts at full sensor resolution: the dorsal fin (and its surround) cut from the RAW
file, normalised for pose with the canonical map (mirrored to a common side, rotated so the
head points left, scaled so the fin height is constant), then compared the way researchers
do: scars and barnacle marks as local keypoints with a geometric consistency check (SIFT +
ratio test + RANSAC affine), the trailing-edge shape as a profile, and position-aligned
DINOv3 patches on the normalised chart.

    python -m model_fine_tuning.cluster.chart extract --index INDEX_JSONL --canon-dir PRED --masks MASKS --out CHARTS [--size 768]
    python -m model_fine_tuning.cluster.chart score --charts CHARTS --timestamps TS --out OUT

score writes scores.npz in match.py's layout (sources chart_sift, chart_edge, chart_dino).
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from ..common import geometry, paths
from .fin import fin_pixels
from .match import time_term


def trailing_edge_profile(whale_c: np.ndarray, fin_c: np.ndarray, n: int = 48) -> np.ndarray:
    """Shape of the fin's trailing edge: walk the whale outline from the fin tip (highest fin
    pixel) down the trailing side (the left: the head is on the right) to the base line of the
    fin box, resample n points by arclength, express as (x, y) relative to the tip and divided
    by the fin height. Invariant to position and scale; the chart already fixed rotation/mirror."""
    cnt, _ = cv2.findContours(whale_c.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnt:
        return np.full(2 * n, np.nan, np.float32)
    c = max(cnt, key=cv2.contourArea)[:, 0, :].astype(float)       # (M, 2) x, y
    fy, fx = np.nonzero(fin_c)
    if len(fy) == 0:
        return np.full(2 * n, np.nan, np.float32)
    tip = np.array([fx[np.argmin(fy)], fy.min()], float)
    base_y = fy.max()
    i0 = int(np.argmin(np.linalg.norm(c - tip, axis=1)))
    # walk in the direction that moves left (trailing side, towards the tail) and down
    def walk(step):
        pts = [c[i0]]
        i = i0
        for _ in range(len(c)):
            i = (i + step) % len(c)
            pts.append(c[i])
            if c[i][1] >= base_y:
                break
        return np.array(pts)
    a, b = walk(1), walk(-1)
    path = a if (a[min(len(a) - 1, 15), 0] - a[0, 0]) < (b[min(len(b) - 1, 15), 0] - b[0, 0]) else b
    if len(path) < 5:
        return np.full(2 * n, np.nan, np.float32)
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    t = np.linspace(0, seg[-1], n)
    rs = np.stack([np.interp(t, seg, path[:, 0]), np.interp(t, seg, path[:, 1])], 1)
    height = max(base_y - tip[1], 1.0)
    return ((rs - tip) / height).reshape(-1).astype(np.float32)


def decode_full(raw_path: str) -> np.ndarray:
    import rawpy

    with rawpy.imread(raw_path) as raw:
        return raw.postprocess(use_camera_wb=True, half_size=False, output_bps=8)


def chart_transform(fin_crop: np.ndarray, body_crop: np.ndarray, signed_m: np.ndarray, x0: float, y0: float, s: float, scale_full: float,
                    size: int, view_side: int, margin: float = 0.45) -> tuple[np.ndarray, float]:
    """Affine (2x3) from the full-resolution image to the chart: translate to the fin centre,
    rotate so that, once left-side views are mirrored, the head points right and the fin up, scale so the fin
    box spans size / (1 + 2 margin). The head direction is the least-squares gradient of the
    body fraction over the whole visible whale (the fin alone spans too little of it)."""
    ys, xs = np.nonzero(fin_crop)
    cx, cy = xs.mean(), ys.mean()
    extent = max(np.ptp(xs), np.ptp(ys)) + 1.0
    by, bx = np.nonzero(body_crop)
    # body axis in the image = principal axis of the visible whale (robust); which end is the head =
    # the sign of the body-fraction gradient along that axis
    pts = np.stack([bx, by], 1).astype(float); pts -= pts.mean(0)
    _, _, vt = np.linalg.svd(pts, full_matrices=False)
    axis = vt[0]
    proj = pts @ axis
    sign = np.sign(np.corrcoef(proj, signed_m[by, bx, 0])[0, 1]) if proj.std() > 1e-6 else 1.0
    h = axis * (sign if sign != 0 else 1.0)
    angle = np.degrees(np.arctan2(h[1], h[0]))                     # direction of the head in the image (y down)
    fx, fy = (cx / s + x0) * scale_full, (cy / s + y0) * scale_full
    extent_full = extent / s * scale_full
    out_scale = size / (extent_full * (1 + 2 * margin))
    # a whale seen from its right flank shows its head on the viewer's right (fin up); seen from the
    # left flank the head is on the left, and the mirror applied afterwards puts it on the right too
    target = 0.0 if view_side >= 0 else 180.0
    M = cv2.getRotationMatrix2D((fx, fy), angle - target, out_scale)
    M[:, 2] += np.array([size / 2 - fx, size / 2 - fy])
    return M, float(out_scale)


def extract_photo(rec: dict, canon_dir: Path, masks_dir: Path, out: Path, table: dict, size: int, min_px: int = 400) -> dict | None:
    stem = Path(rec["image"]).stem
    z = np.load(canon_dir / f"{stem}_canon.npz")
    x0, y0, s = z["x0y0s"]
    use, side = z["use"], z["side"]
    signed = z["signed_mm"].astype(np.float32) / 1000.0
    fin = fin_pixels(signed, use, table)
    if fin.sum() < min_px:
        return None
    full = decode_full(rec["source"])
    H, W = full.shape[:2]
    img2560 = cv2.imread(str(canon_dir.parent / "images" / f"{stem}.jpg"))
    scale_full = W / img2560.shape[1]
    view_side = int(np.sign(side[fin].sum())) if np.abs(side[fin]).sum() else 0
    M, out_scale = chart_transform(fin, use, signed, x0, y0, s, scale_full, size, view_side)
    chart = cv2.warpAffine(full, M, (size, size), flags=cv2.INTER_AREA if out_scale < 1 else cv2.INTER_CUBIC)
    # masks: crop (512) -> full-res -> chart
    crop_to_full = np.array([[scale_full / s, 0, x0 * scale_full], [0, scale_full / s, y0 * scale_full]], np.float64)
    M3 = np.vstack([M, [0, 0, 1]]) @ np.vstack([crop_to_full, [0, 0, 1]])
    fin_c = cv2.warpAffine(fin.astype(np.uint8), M3[:2], (size, size), flags=cv2.INTER_NEAREST) > 0
    whale_c = cv2.warpAffine(use.astype(np.uint8), M3[:2], (size, size), flags=cv2.INTER_NEAREST) > 0
    mirrored = False
    if view_side < 0:
        chart, fin_c, whale_c, mirrored = chart[:, ::-1].copy(), fin_c[:, ::-1].copy(), whale_c[:, ::-1].copy(), True
    # trailing edge profile: after normalisation the head is on the left, so the trailing edge is the right boundary of
    # the whale mask inside the fin's rows, from the fin tip down to the base
    prof = trailing_edge_profile(whale_c, fin_c)
    grey = cv2.cvtColor(chart, cv2.COLOR_RGB2GRAY)
    region = cv2.dilate(whale_c.astype(np.uint8), np.ones((15, 15), np.uint8))   # whale incl. a thin margin
    sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=0.02)
    kps, desc = sift.detectAndCompute(grey, region * 255)
    kp_xy = np.array([k.pt for k in kps], np.float32) if kps else np.zeros((0, 2), np.float32)
    desc = desc.astype(np.float32) if desc is not None else np.zeros((0, 128), np.float32)
    cv2.imwrite(str(out / "charts" / f"{stem}.jpg"), cv2.cvtColor(chart, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    np.savez_compressed(out / f"{stem}.npz", fin=fin_c, whale=whale_c, profile=prof, kp_xy=kp_xy, kp_desc=desc, view_side=view_side, mirrored=mirrored,
                        out_scale=out_scale, fin_px=int(fin.sum()))
    return {"stem": stem, "kp": len(kp_xy), "view_side": view_side, "mirrored": mirrored}


def _extract_worker(args):
    rec, canon_dir, masks_dir, out, size = args
    table = geometry.load_folded_table(paths.FOLDED_TABLE)
    try:
        return extract_photo(rec, Path(canon_dir), Path(masks_dir), Path(out), table, size)
    except Exception as e:  # noqa: BLE001
        return {"stem": Path(rec["image"]).stem, "error": str(e)}


def cmd_extract(args) -> int:
    recs = [json.loads(l) for l in open(args.index)]
    args.out.mkdir(parents=True, exist_ok=True); (args.out / "charts").mkdir(exist_ok=True)
    jobs = [(r, str(args.canon_dir), str(args.masks), str(args.out), args.size) for r in recs if (args.canon_dir / f"{Path(r['image']).stem}_canon.npz").exists()]
    done, errors = 0, 0
    with Pool(args.workers) as pool:
        for res in pool.imap_unordered(_extract_worker, jobs, chunksize=2):
            if res is None:
                continue
            if "error" in res:
                errors += 1; print("error", res["stem"], res["error"][:120], flush=True); continue
            done += 1
            if done % 25 == 0:
                print(f"{done} charts", flush=True)
    print(f"charts: {done} ({errors} errors) -> {args.out}")
    return 0


def dino_chart_features(charts_dir: Path, stems: list[str], size: int) -> dict:
    import torch

    from ..common.backbone import DinoFeatures, normalize_image
    from ..common.data import to_tensor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = DinoFeatures("vitb16", layers=[11]).to(device).eval()
    out = {}
    with torch.no_grad():
        for s in stems:
            img = cv2.cvtColor(cv2.imread(str(charts_dir / "charts" / f"{s}.jpg")), cv2.COLOR_BGR2RGB)
            x = normalize_image(to_tensor(img)[None].to(device))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                feats, _ = backbone(x)
            F = feats[-1][0].float().cpu().numpy()
            F /= np.maximum(np.linalg.norm(F, axis=0, keepdims=True), 1e-6)
            out[s] = F.astype(np.float16)
    return out


def sift_score(a: dict, b: dict, ratio: float = 0.8, reproj: float = 12.0) -> tuple[float, int]:
    da, db = a["kp_desc"], b["kp_desc"]
    if len(da) < 8 or len(db) < 8:
        return np.nan, 0
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < ratio * n.distance]
    if len(good) < 4:
        return 0.0, 0
    pa = np.float32([a["kp_xy"][m.queryIdx] for m in good]); pb = np.float32([b["kp_xy"][m.trainIdx] for m in good])
    # the charts are already pose-normalised, so a consistent match is a small, near-identity similarity transform
    _, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=reproj, maxIters=2000, confidence=0.99)
    inliers = int(inl.sum()) if inl is not None else 0
    return inliers / np.sqrt(min(len(da), len(db))), inliers


def edge_score(a: dict, b: dict) -> float:
    pa, pb = a["profile"], b["profile"]
    ok = np.isfinite(pa) & np.isfinite(pb)
    return -float(np.sqrt(np.mean((pa[ok] - pb[ok]) ** 2))) if ok.sum() > 20 else np.nan


def dino_score(Fa: np.ndarray, Fb: np.ndarray, fin_a: np.ndarray, fin_b: np.ndarray, patch: int = 16, rng=None) -> float:
    """Position-aligned (same chart location, +-1 patch) minus misaligned cosine over fin patches."""
    g = Fa.shape[1]
    fa = fin_a.reshape(g, patch, g, patch).mean((1, 3)) > 0.3; fb = fin_b.reshape(g, patch, g, patch).mean((1, 3)) > 0.3
    both = fa & fb
    ys, xs = np.nonzero(both)
    if len(ys) < 12:
        return np.nan
    Fa32, Fb32 = Fa.astype(np.float32), Fb.astype(np.float32)
    vals = []
    by, bx = np.nonzero(fb)
    for y, x in zip(ys, xs):
        y0, y1, x0, x1 = max(0, y - 1), min(g, y + 2), max(0, x - 1), min(g, x + 2)
        nb = Fb32[:, y0:y1, x0:x1].reshape(Fb32.shape[0], -1)
        aligned = float(np.max(Fa32[:, y, x] @ nb))
        far = (np.abs(by - y) > 4) | (np.abs(bx - x) > 4)
        if far.sum() < 5:
            continue
        pick = rng.choice(np.nonzero(far)[0], min(16, int(far.sum())), replace=False)
        mis = float(np.mean(Fa32[:, y, x] @ Fb32[:, by[pick], bx[pick]]))
        vals.append(aligned - mis)
    return float(np.mean(vals)) if len(vals) >= 12 else np.nan


def cmd_score(args) -> int:
    files = sorted(f for f in args.charts.glob("*.npz")); stems = [f.stem for f in files]
    recs = {s: dict(np.load(f)) for s, f in zip(stems, files)}
    ts = json.loads(args.timestamps.read_text())
    n = len(stems)
    t = np.array([ts[s]["time"] for s in stems]); cam = np.array([ts[s]["camera"] for s in stems]); dt = np.abs(t[:, None] - t[None, :])
    side = np.array([int(recs[s]["view_side"]) for s in stems])
    same_view = (side[:, None] == side[None, :]) | (side[:, None] == 0) | (side[None, :] == 0)
    print(f"{n} charts; computing DINOv3 chart features", flush=True)
    F = dino_chart_features(args.charts, stems, args.size)
    rng = np.random.default_rng(0)
    mats = {k: np.full((n, n), np.nan) for k in ("chart_sift", "chart_edge", "chart_dino")}
    counts = np.zeros((n, n), np.int32)
    for i in range(n):
        for j in range(i + 1, n):
            if not same_view[i, j]:
                continue
            s_, inl = sift_score(recs[stems[i]], recs[stems[j]])
            mats["chart_sift"][i, j] = mats["chart_sift"][j, i] = s_; counts[i, j] = counts[j, i] = inl
            mats["chart_edge"][i, j] = mats["chart_edge"][j, i] = edge_score(recs[stems[i]], recs[stems[j]])
            mats["chart_dino"][i, j] = mats["chart_dino"][j, i] = dino_score(F[stems[i]], F[stems[j]], recs[stems[i]]["fin"], recs[stems[j]]["fin"], rng=rng)
        if i % 25 == 0:
            print(f"{i}/{n}", flush=True)
    out = {"dt_s": dt, "time_term": time_term(dt), "same_camera": cam[:, None] == cam[None, :]}
    for k, m in mats.items():
        fin = np.isfinite(m)
        z = (m - np.nanmean(m)) / (np.nanstd(m) + 1e-9) if fin.sum() > 10 else m
        out[f"{k}__same"] = z; out[f"{k}__flip"] = np.full((n, n), np.nan)
        out[f"{k}__n_same"] = np.where(fin, 1, 0); out[f"{k}__n_flip"] = np.zeros((n, n), np.int32)
    out["sift_inliers"] = counts
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "scores.npz", stems=np.array(stems), **out)
    json.dump({"sources": list(mats), "calib": {}}, open(args.out / "match_meta.json", "w"))
    print("wrote", args.out)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract"); e.add_argument("--index", type=Path, required=True); e.add_argument("--canon-dir", type=Path, required=True)
    e.add_argument("--masks", type=Path, required=True); e.add_argument("--out", type=Path, required=True); e.add_argument("--size", type=int, default=768); e.add_argument("--workers", type=int, default=8)
    s = sub.add_parser("score"); s.add_argument("--charts", type=Path, required=True); s.add_argument("--timestamps", type=Path, required=True); s.add_argument("--out", type=Path, required=True); s.add_argument("--size", type=int, default=768)
    args = p.parse_args(argv)
    return cmd_extract(args) if args.cmd == "extract" else cmd_score(args)


if __name__ == "__main__":
    raise SystemExit(main())
