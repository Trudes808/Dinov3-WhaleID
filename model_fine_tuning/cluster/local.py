"""Local aligned matching: compare the *patterns* at the same body locations, not the mean
look of a region. For a pair of photos, every DINOv3 patch (fine grid, 2x crop) of A is
compared with the patches of B that lie within `radius` on the canonical surface (same flank
under the hypothesis); its best cosine is the aligned similarity. The same patch compared with
B patches far away on the body gives the misaligned baseline. The pair score is the mean
(aligned - misaligned) over matched patches, symmetrised: it cancels the semantic sameness of
whale skin and keeps only what is specific to that spot. A SIFT variant does the same with
classical keypoints (scars, pigment edges): matches that pass the ratio test and land on the
same body spot, minus those that land elsewhere, per keypoint.

    python -m model_fine_tuning.cluster.local extract --images DIR --canon-dir PRED --out LOCAL [--upsample 2]
    python -m model_fine_tuning.cluster.local score --local LOCAL --atlas ATLAS --timestamps TS --out OUT [--radius 0.3]

The score step writes scores.npz in the same layout as match.py (local_dino__same/flip,
sift__same/flip, plus the time term), so cluster.py and pair_auc.py work on it unchanged.
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from ..common.data import warp_image, warp_params
from .match import time_term


# ---------------------------------------------------------------- extraction
def extract_photo(stem: str, images_dir: Path, canon_dir: Path, out_dir: Path, backbone, device, up: int, patch: int = 16, min_frac: float = 0.5) -> int:
    import torch

    from ..common.backbone import normalize_image
    from ..common.data import to_tensor

    z = np.load(canon_dir / f"{stem}_canon.npz")
    x0, y0, s = z["x0y0s"]
    use, side, signed = z["use"], z["side"], z["signed_mm"].astype(np.float32) / 1000.0
    C = use.shape[0]; S = C * up
    image = cv2.cvtColor(cv2.imread(str(images_dir / f"{stem}.jpg"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    crop = warp_image(image, warp_params(x0, y0, s * up), (S, S))
    nearest = lambda m: cv2.resize(m, (S, S), interpolation=cv2.INTER_NEAREST)
    use_u = nearest(use.astype(np.uint8)) > 0
    side_u = nearest(side.astype(np.int16)).astype(np.int8)
    pts_u = np.stack([cv2.resize(signed[..., k], (S, S), interpolation=cv2.INTER_NEAREST) for k in range(3)], -1)
    # patch grid
    with torch.no_grad():
        x = normalize_image(to_tensor(np.ascontiguousarray(crop))[None].to(device))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            feats, _ = backbone(x)
        F = feats[-1][0].float().cpu().numpy()                      # (768, S/16, S/16)
    g = S // patch
    frac = use_u.reshape(g, patch, g, patch).mean((1, 3))
    keep = frac >= min_frac
    gy, gx = np.nonzero(keep)
    pts = np.zeros((len(gy), 3), np.float32); sides = np.zeros(len(gy), np.int8)
    for n, (i, j) in enumerate(zip(gy, gx)):
        sl = (slice(i * patch, (i + 1) * patch), slice(j * patch, (j + 1) * patch))
        m = use_u[sl]
        pts[n] = pts_u[sl][m].mean(0)
        sd = side_u[sl][m]; sides[n] = np.sign(sd.sum()) if np.abs(sd).sum() else 0
    feat = F[:, gy, gx].T.astype(np.float16)
    # SIFT keypoints on the whale
    grey = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(nfeatures=1500)
    kps, desc = sift.detectAndCompute(grey, use_u.astype(np.uint8) * 255)
    if desc is None:
        kp_xy = np.zeros((0, 2), np.float32); desc = np.zeros((0, 128), np.float32); kp_pts = np.zeros((0, 3), np.float32); kp_side = np.zeros(0, np.int8)
    else:
        kp_xy = np.array([k.pt for k in kps], np.float32)
        ix, iy = np.clip(kp_xy[:, 0].astype(int), 0, S - 1), np.clip(kp_xy[:, 1].astype(int), 0, S - 1)
        kp_pts = pts_u[iy, ix]; kp_side = side_u[iy, ix]
    np.savez_compressed(out_dir / f"{stem}.npz", patch_xy=np.stack([gx, gy], 1).astype(np.int16), patch_pts=pts, patch_side=sides, patch_feat=feat,
                        kp_xy=kp_xy, kp_desc=desc.astype(np.float32), kp_pts=kp_pts, kp_side=kp_side, x0y0s=np.array([x0, y0, s * up]), size=S)
    return len(gy)


def cmd_extract(args) -> int:
    import torch

    from ..common.backbone import DinoFeatures

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = DinoFeatures("vitb16", layers=[11]).to(device).eval()
    args.out.mkdir(parents=True, exist_ok=True)
    stems = sorted(f.name[: -len("_canon.npz")] for f in args.canon_dir.glob("*_canon.npz") if (args.images / (f.name[: -len("_canon.npz")] + ".jpg")).exists())
    for n, stem in enumerate(stems):
        extract_photo(stem, args.images, args.canon_dir, args.out, backbone, device, args.upsample)
        if n % 50 == 0:
            print(f"{n}/{len(stems)}", flush=True)
    (args.out / "meta.json").write_text(json.dumps({"photos": len(stems), "upsample": args.upsample}))
    print("extracted", len(stems))
    return 0


# ---------------------------------------------------------------- scoring
_DATA = {}


def _load(local_dir: Path, stems: list[str]) -> dict:
    out = {}
    for s in stems:
        z = np.load(local_dir / f"{s}.npz")
        f = z["patch_feat"].astype(np.float32)
        f /= np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-6)
        d = z["kp_desc"]; d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)
        out[s] = {"pts": z["patch_pts"], "side": z["patch_side"], "feat": f, "kp_pts": z["kp_pts"], "kp_side": z["kp_side"], "kp_desc": d}
    return out


def signed(pts: np.ndarray, side: np.ndarray, flip: bool) -> np.ndarray:
    p = pts.copy()
    sgn = -side if flip else side
    p[:, 1] = np.abs(p[:, 1]) * np.where(sgn < 0, -1.0, 1.0)
    return p


def pair_score(a: dict, b: dict, flip: bool, radius: float, far: float, rng: np.random.Generator) -> tuple[float, int, float, int]:
    """(dino aligned-minus-misaligned, matched patches, sift score, sift aligned matches)."""
    from scipy.spatial import cKDTree

    pa, pb = signed(a["pts"], a["side"], False), signed(b["pts"], b["side"], flip)
    out_d, n_d = np.nan, 0
    if len(pa) >= 10 and len(pb) >= 10:
        tree = cKDTree(pb)
        vals = []
        for i, nb in enumerate(tree.query_ball_point(pa, radius)):
            if not nb:
                continue
            aligned = float(np.max(b["feat"][nb] @ a["feat"][i]))
            dist = np.linalg.norm(pb - pa[i], axis=1)
            faridx = np.nonzero(dist > far)[0]
            if len(faridx) < 5:
                continue
            pick = rng.choice(faridx, min(16, len(faridx)), replace=False)
            mis = float(np.mean(b["feat"][pick] @ a["feat"][i]))
            vals.append(aligned - mis)
        if len(vals) >= 20:
            out_d, n_d = float(np.mean(vals)), len(vals)
    out_s, n_s = np.nan, 0
    da, db = a["kp_desc"], b["kp_desc"]
    if len(da) >= 10 and len(db) >= 10:
        ka, kb = signed(a["kp_pts"], a["kp_side"], False), signed(b["kp_pts"], b["kp_side"], flip)
        sim = da @ db.T
        top2 = np.argpartition(-sim, 1, axis=1)[:, :2]
        best = top2[np.arange(len(da)), np.argmax(sim[np.arange(len(da))[:, None], top2], axis=1)]
        second = np.sort(sim[np.arange(len(da))[:, None], top2], axis=1)[:, 0]
        s1 = sim[np.arange(len(da)), best]
        ratio_ok = (1 - s1) < 0.8 * (1 - second)                   # Lowe's ratio on (1 - cos) distances
        d = np.linalg.norm(ka - kb[best], axis=1)
        near_tree = cKDTree(kb)
        in_overlap = np.array([len(x) > 0 for x in near_tree.query_ball_point(ka, far)])
        n_overlap = int(in_overlap.sum())
        if n_overlap >= 10:
            aligned = int(np.sum(ratio_ok & (d < radius) & in_overlap))
            wrong = int(np.sum(ratio_ok & (d > far) & in_overlap))
            out_s, n_s = (aligned - wrong) / n_overlap, aligned
    return out_d, n_d, out_s, n_s


def patch_contributions(a: dict, b: dict, flip: bool, radius: float = 0.3, far: float = 1.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Per-patch (aligned - misaligned) cosine of A's patches against B (nan where unmatched),
    and the same for B's patches against A: the material of the 'why linked' heatmap."""
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    out = []
    for src, dst, fl_src, fl_dst in ((a, b, False, flip), (b, a, flip, False)):
        ps, pd = signed(src["pts"], src["side"], fl_src), signed(dst["pts"], dst["side"], fl_dst)
        vals = np.full(len(ps), np.nan)
        if len(ps) >= 10 and len(pd) >= 10:
            tree = cKDTree(pd)
            for i, nb in enumerate(tree.query_ball_point(ps, radius)):
                if not nb:
                    continue
                dist = np.linalg.norm(pd - ps[i], axis=1)
                faridx = np.nonzero(dist > far)[0]
                if len(faridx) < 5:
                    continue
                pick = rng.choice(faridx, min(16, len(faridx)), replace=False)
                vals[i] = float(np.max(dst["feat"][nb] @ src["feat"][i])) - float(np.mean(dst["feat"][pick] @ src["feat"][i]))
        out.append(vals)
    return out[0], out[1]


def _worker(args):
    i, j, radius, far, seed = args
    rng = np.random.default_rng(seed)
    a, b = _DATA["recs"][i], _DATA["recs"][j]
    res = {}
    for h, flip in (("same", False), ("flip", True)):
        d1, n1, s1, m1 = pair_score(a, b, flip, radius, far, rng)
        d2, n2, s2, m2 = pair_score(b, a, flip, radius, far, rng)
        res[h] = (np.nanmean([d1, d2]) if not (np.isnan(d1) and np.isnan(d2)) else np.nan, n1 + n2,
                  np.nanmean([s1, s2]) if not (np.isnan(s1) and np.isnan(s2)) else np.nan, m1 + m2)
    return i, j, res


def _init(recs):
    _DATA["recs"] = recs


def cmd_score(args) -> int:
    from .match import load_atlas

    stems, atlas = load_atlas(args.atlas)
    data = _load(args.local, stems)
    recs = [data[s] for s in stems]
    n = len(stems)
    ts = json.loads(args.timestamps.read_text())
    t = np.array([ts[s]["time"] for s in stems], float); cam = np.array([ts[s]["camera"] for s in stems])
    dt = np.abs(t[:, None] - t[None, :])
    # candidate pairs: share at least a few atlas cells (either side)
    cellsets = [set(map(int, r["cell"])) for r in atlas]
    jobs = [(i, j, args.radius, args.far, i * n + j) for i in range(n) for j in range(i + 1, n) if len(cellsets[i] & cellsets[j]) >= args.min_shared_cells]
    print(f"{len(jobs)} candidate pairs of {n * (n - 1) // 2}", flush=True)
    mats = {f"{src}__{h}": np.full((n, n), np.nan) for src in ("local_dino", "sift") for h in ("same", "flip")}
    counts = {f"{src}__n_{h}": np.zeros((n, n), np.int32) for src in ("local_dino", "sift") for h in ("same", "flip")}
    with Pool(args.workers, initializer=_init, initargs=(recs,)) as pool:
        for k, (i, j, res) in enumerate(pool.imap_unordered(_worker, jobs, chunksize=16)):
            for h, (d, nd, s, ns) in res.items():
                mats[f"local_dino__{h}"][i, j] = mats[f"local_dino__{h}"][j, i] = d
                counts[f"local_dino__n_{h}"][i, j] = counts[f"local_dino__n_{h}"][j, i] = nd
                mats[f"sift__{h}"][i, j] = mats[f"sift__{h}"][j, i] = s
                counts[f"sift__n_{h}"][i, j] = counts[f"sift__n_{h}"][j, i] = ns
            if k % 5000 == 0:
                print(f"{k}/{len(jobs)}", flush=True)
    # z-score each matrix over its finite entries so the scale is comparable to match.py's
    for key in list(mats):
        m = mats[key]; fin = np.isfinite(m)
        if fin.sum() > 10:
            mats[key] = (m - np.nanmean(m)) / (np.nanstd(m) + 1e-9)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "scores.npz", stems=np.array(stems), dt_s=dt, time_term=time_term(dt), same_camera=cam[:, None] == cam[None, :], **mats, **counts)
    json.dump({"sources": ["local_dino", "sift"], "radius": args.radius, "far": args.far, "calib": {}}, open(args.out / "match_meta.json", "w"))
    for key, m in mats.items():
        print(f"{key:18s} scored pairs {int(np.isfinite(m).sum() // 2)}")
    print("wrote", args.out)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract"); e.add_argument("--images", type=Path, required=True); e.add_argument("--canon-dir", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True); e.add_argument("--upsample", type=int, default=2)
    s = sub.add_parser("score"); s.add_argument("--local", type=Path, required=True); s.add_argument("--atlas", type=Path, required=True)
    s.add_argument("--timestamps", type=Path, required=True); s.add_argument("--out", type=Path, required=True)
    s.add_argument("--radius", type=float, default=0.3); s.add_argument("--far", type=float, default=1.0)
    s.add_argument("--min-shared-cells", type=int, default=4); s.add_argument("--workers", type=int, default=24)
    args = p.parse_args(argv)
    return cmd_extract(args) if args.cmd == "extract" else cmd_score(args)


if __name__ == "__main__":
    raise SystemExit(main())
