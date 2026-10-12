"""Part-centric descriptor for the dorsal fin: the part photographers use to tell humpbacks
apart in back/flank photos. The fin is located on the canonical surface (body fraction 0.36 to
0.54, above the back line), cut out of the full-resolution photo with its own frame, mirrored
to a common viewing side using the resolved side, and described by (a) DINOv3 features of the
fin crop at high resolution (mean-pooled patch tokens on the fin, and the CLS token) and (b) its
silhouette (the trailing-edge profile from the segmentation mask, resampled along the fin
height). Pairs are compared by cosine of (a) and by the profile distance of (b).

    python -m model_fine_tuning.cluster.fin extract --images DIR --canon-dir PRED --masks MASKS --out FIN
    python -m model_fine_tuning.cluster.fin score --fin FIN --timestamps TS --out OUT
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from ..common import geometry, paths
from .match import time_term

U_LO, U_HI, Z_MIN = 0.36, 0.54, 0.9        # fin region on the canonical body (fractions of length, metres above the axis)


def fin_pixels(signed_m: np.ndarray, use: np.ndarray, table: dict) -> np.ndarray:
    u = (signed_m[..., 0] - table["x_min_m"]) / table["length_m"]
    return use & (u > U_LO) & (u < U_HI) & (signed_m[..., 2] > Z_MIN)


def extract_photo(stem: str, images_dir: Path, canon_dir: Path, masks_dir: Path, table: dict, backbone, device, size: int = 448, min_px: int = 400) -> dict | None:
    import torch

    from ..common.backbone import normalize_image
    from ..common.data import to_tensor

    z = np.load(canon_dir / f"{stem}_canon.npz")
    x0, y0, s = z["x0y0s"]
    use, side = z["use"], z["side"]
    signed = z["signed_mm"].astype(np.float32) / 1000.0
    fin = fin_pixels(signed, use, table)
    if fin.sum() < min_px:
        return None
    # fin box in crop coordinates -> full-image coordinates
    ys, xs = np.nonzero(fin)
    cx, cy = xs.mean(), ys.mean()
    half = max(xs.max() - xs.min(), ys.max() - ys.min()) * 0.75 + 8
    fx0, fy0 = (cx - half) / s + x0, (cy - half) / s + y0           # crop px -> full px: full = crop / s + (x0, y0)
    fs = size / (2 * half / s)
    image = cv2.cvtColor(cv2.imread(str(images_dir / f"{stem}.jpg"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    M = np.array([[fs, 0, -fx0 * fs], [0, fs, -fy0 * fs]], np.float64)
    crop = cv2.warpAffine(image, M, (size, size), flags=cv2.INTER_LINEAR)
    mask_full = cv2.imread(str(masks_dir / f"{stem}_mask.png"), cv2.IMREAD_UNCHANGED)
    if mask_full is None:
        return None
    if mask_full.shape != image.shape[:2]:
        mask_full = cv2.resize(mask_full, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_c = cv2.warpAffine((mask_full > 0).astype(np.uint8), M, (size, size), flags=cv2.INTER_NEAREST) > 0
    # the fin pixels live in the 512 crop: crop -> full (inverse of A) -> fin box (M)
    A = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], np.float64)
    crop_to_fin = (np.vstack([M, [0, 0, 1]]) @ np.vstack([cv2.invertAffineTransform(A), [0, 0, 1]]))[:2]
    fin_c = cv2.warpAffine(fin.astype(np.uint8), crop_to_fin, (size, size), flags=cv2.INTER_NEAREST) > 0
    view_side = int(np.sign(side[fin].sum())) if np.abs(side[fin]).sum() else 0
    flipped = False
    if view_side < 0:                                   # seen from the whale's left: mirror to the common (right-side) view
        crop, mask_c, fin_c, flipped = crop[:, ::-1].copy(), mask_c[:, ::-1].copy(), fin_c[:, ::-1].copy(), True
    with torch.no_grad():
        x = normalize_image(to_tensor(np.ascontiguousarray(crop))[None].to(device))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            feats, cls = backbone(x)
        F = feats[-1][0].float().cpu().numpy()
        cls = cls[0].float().cpu().numpy()
    g = size // 16
    pf = fin_c.reshape(g, 16, g, 16).mean((1, 3))                 # fraction of fin pixels per patch
    w = pf.reshape(-1)
    pooled = (F.reshape(F.shape[0], -1) * w[None]).sum(1) / max(w.sum(), 1e-6)
    # silhouette: the fin's own mask region (segmentation mask within the fin box), trailing-edge profile = rightmost
    # mask pixel per row within the fin's rows (common view: head to the left after mirroring is not guaranteed; use both edges)
    rows = np.nonzero(fin_c.any(1))[0]
    prof_l, prof_r = [], []
    for r in np.linspace(rows.min(), rows.max(), 32).astype(int):
        cols = np.nonzero(mask_c[r])[0]
        prof_l.append(cols.min() if len(cols) else np.nan); prof_r.append(cols.max() if len(cols) else np.nan)
    width = max(np.nanmax(prof_r) - np.nanmin(prof_l), 1.0)
    prof = (np.array([prof_l, prof_r], np.float32) - np.nanmin(prof_l)) / width
    return {"feat": pooled.astype(np.float32), "cls": cls.astype(np.float32), "profile": prof, "view_side": view_side, "flipped": flipped,
            "fin_px": int(fin.sum()), "crop": crop, "fin_mask": fin_c}


def cmd_extract(args) -> int:
    import torch

    from ..common.backbone import DinoFeatures

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = DinoFeatures("vitb16", layers=[11]).to(device).eval()
    table = geometry.load_folded_table(paths.FOLDED_TABLE)
    args.out.mkdir(parents=True, exist_ok=True); (args.out / "crops").mkdir(exist_ok=True)
    stems = sorted(f.name[: -len("_canon.npz")] for f in args.canon_dir.glob("*_canon.npz"))
    n = 0
    for stem in stems:
        try:
            rec = extract_photo(stem, args.images, args.canon_dir, args.masks, table, backbone, device)
        except Exception as e:  # noqa: BLE001
            print("skip", stem, e); rec = None
        if rec is None:
            continue
        crop = rec.pop("crop"); fm = rec.pop("fin_mask")
        vis = crop.copy(); vis[fm] = (0.5 * vis[fm] + 0.5 * np.array([255, 255, 0])).astype(np.uint8)
        cv2.imwrite(str(args.out / "crops" / f"{stem}.jpg"), cv2.cvtColor(np.concatenate([crop, vis], 1), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        np.savez_compressed(args.out / f"{stem}.npz", **rec)
        n += 1
    print(f"fin descriptors for {n} of {len(stems)} photos -> {args.out}")
    return 0


def cmd_score(args) -> int:
    files = sorted(f for f in args.fin.glob("*.npz"))
    stems = [f.stem for f in files]
    recs = [np.load(f) for f in files]
    ts = json.loads(args.timestamps.read_text())
    n = len(stems)
    t = np.array([ts[s]["time"] for s in stems]); cam = np.array([ts[s]["camera"] for s in stems]); dt = np.abs(t[:, None] - t[None, :])
    F = np.stack([r["feat"] for r in recs]); F /= np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-6)
    C = np.stack([r["cls"] for r in recs]); C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-6)
    P = np.stack([np.nan_to_num(r["profile"], nan=0.5).reshape(-1) for r in recs])
    side = np.array([int(r["view_side"]) for r in recs])
    out = {"dt_s": dt, "time_term": time_term(dt), "same_camera": cam[:, None] == cam[None, :]}
    same_view = (side[:, None] == side[None, :]) | (side[:, None] == 0) | (side[None, :] == 0)
    for name, S in (("fin_feat", F @ F.T), ("fin_cls", C @ C.T), ("fin_profile", -np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1))):
        z = (S - S[~np.eye(n, dtype=bool)].mean()) / (S[~np.eye(n, dtype=bool)].std() + 1e-9)
        out[f"{name}__same"] = np.where(same_view, z, np.nan)      # texture is side-specific: only compare fins seen from the same flank
        out[f"{name}__flip"] = np.where(same_view, np.nan, z) if name != "fin_profile" else z   # the silhouette can be compared across sides (mirrored)
        out[f"{name}__n_same"] = np.where(np.isfinite(out[f"{name}__same"]), 1, 0); out[f"{name}__n_flip"] = np.where(np.isfinite(out[f"{name}__flip"]), 1, 0)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "scores.npz", stems=np.array(stems), **out)
    json.dump({"sources": ["fin_feat", "fin_cls", "fin_profile"], "calib": {}}, open(args.out / "match_meta.json", "w"))
    print(f"fin scores for {n} photos -> {args.out}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract"); e.add_argument("--images", type=Path, required=True); e.add_argument("--canon-dir", type=Path, required=True)
    e.add_argument("--masks", type=Path, required=True); e.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("score"); s.add_argument("--fin", type=Path, required=True); s.add_argument("--timestamps", type=Path, required=True); s.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    return cmd_extract(args) if args.cmd == "extract" else cmd_score(args)


if __name__ == "__main__":
    raise SystemExit(main())
