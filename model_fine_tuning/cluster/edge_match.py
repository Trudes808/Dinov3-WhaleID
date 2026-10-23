"""Trailing-edge matching for dorsal fins and flukes at native resolution (the researchers' first cue).

    python -m model_fine_tuning.cluster.edge_match --day-dir DAY --labels labels.json [--part fin|fluke] [--min-size 150]

Edge extraction (parts_xr charts, 3072 px): the fin silhouette's outline from the tip down the trailing edge to the
base (charts have the head on the right, so the trailing edge is the left flank of the fin), or the fluke blades' lower
boundary across the visible span. The segmentation mask is coarse at this scale (4-6 px steps), so every outline point
is snapped to the strongest image gradient along its normal within +-12 px (the whale / water boundary) and the result
smoothed. Descriptor: the edge resampled by arclength to N points, expressed in a chord frame (tip -> base chord
rotated to the y axis, scaled by the chord length), as (a) the perpendicular deviation profile and (b) its curvature at
three smoothing scales. Matching: dynamic time warping with a Sakoe-Chiba band on the multi-scale curvature (the edge
stretches with pose and perspective; notches are local), score = -mean matched cost. Evaluated as pair AUC on labelled
charts (same animal in another burst vs other animals of the day), for all edges and for edges above --min-size px.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .finshape import isolate_fin

N = 96


def snap_to_gradient(pts: np.ndarray, gray: np.ndarray, search: int = 12) -> np.ndarray:
    """Move every point along the local normal to the strongest gradient within +-search px, then smooth."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy); H, W = mag.shape
    tang = np.gradient(pts, axis=0); tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-6)
    normal = np.stack([-tang[:, 1], tang[:, 0]], 1)
    out = pts.copy()
    offs = np.arange(-search, search + 1)
    for i, (p, n) in enumerate(zip(pts, normal)):
        cand = p[None, :] + offs[:, None] * n[None, :]
        xi = np.clip(np.round(cand[:, 0]).astype(int), 0, W - 1); yi = np.clip(np.round(cand[:, 1]).astype(int), 0, H - 1)
        m = mag[yi, xi] * np.exp(-(offs / search) ** 2)                               # prefer the mask's own edge when gradients tie
        out[i] = cand[int(np.argmax(m))]
    k = 5
    out = np.stack([np.convolve(np.pad(out[:, j], (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid") for j in range(2)], 1)
    return out


def resample(path: np.ndarray, n: int) -> np.ndarray:
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    if seg[-1] <= 0:
        return np.repeat(path[:1], n, 0)
    t = np.linspace(0, seg[-1], n)
    return np.stack([np.interp(t, seg, path[:, 0]), np.interp(t, seg, path[:, 1])], 1)


def edge_sharpness(path: np.ndarray, gray: np.ndarray) -> float:
    """Mean gradient magnitude along the path over the chart's 95th percentile: ~1 on a crisp whale / water edge, low on
    blur, foam or a path that runs over skin."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3); mag = np.hypot(gx, gy)
    xi = np.clip(np.round(path[:, 0]).astype(int), 0, mag.shape[1] - 1); yi = np.clip(np.round(path[:, 1]).astype(int), 0, mag.shape[0] - 1)
    return float(np.clip(mag[yi, xi].mean() / max(np.percentile(mag, 95), 1e-6), 0, 1.5))


def fin_trailing_edge(whale: np.ndarray, gray: np.ndarray, min_height: int = 60, fin_band: np.ndarray | None = None) -> tuple[np.ndarray, dict] | None:
    """Trailing-edge polyline (tip -> base, chart px) of the isolated fin, gradient-snapped. The morphological fin blob
    keeps the crest of a low back as well; the body map's fin band (dilated) cuts the blob down to the fin."""
    fin = isolate_fin(whale)
    if fin is None or fin.sum() < 400:
        return None
    if fin_band is not None and fin_band.any():
        k = max(int(0.04 * whale.shape[0]), 3)
        band = cv2.dilate(fin_band.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
        cut = fin & band
        if cut.sum() >= 400:
            n, lab, st, _ = cv2.connectedComponentsWithStats(cut.astype(np.uint8), connectivity=8)
            if n > 1:
                fin = lab == (1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))
    cnt, _ = cv2.findContours(fin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnt:
        return None
    c = max(cnt, key=cv2.contourArea)[:, 0, :].astype(float)
    ys, xs = np.nonzero(fin); tip = np.array([xs[np.argmin(ys)], ys.min()], float); base_y = ys.max(); height = base_y - tip[1]
    if height < min_height:
        return None
    bottom = c[c[:, 1] >= base_y - 0.15 * height]
    trail = bottom[np.argmin(bottom[:, 0])]                                            # head is on the right: trailing base = leftmost
    i_tip = int(np.argmin(np.linalg.norm(c - tip, axis=1))); i_trail = int(np.argmin(np.linalg.norm(c - trail, axis=1)))

    def walk(a, b, step):
        out = [c[a]]; i = a
        for _ in range(len(c)):
            i = (i + step) % len(c); out.append(c[i])
            if i == b:
                break
        return np.array(out)
    p1, p2 = walk(i_tip, i_trail, 1), walk(i_tip, i_trail, -1)
    # the trailing edge is the path whose mean x is smaller (left of the tip -> base chord's leading side)
    path = p1 if p1[:, 0].mean() < p2[:, 0].mean() else p2
    if len(path) < 10:
        return None
    path = resample(path, 4 * N)
    path = snap_to_gradient(path, gray)
    path = resample(path, N)
    return path, {"height": float(height), "tip": tip.tolist(), "base": trail.tolist(), "sharp": edge_sharpness(path, gray)}


def fluke_lower_edge(fluke: np.ndarray, gray: np.ndarray, min_span: int = 150) -> tuple[np.ndarray, dict] | None:
    """Lower (trailing) boundary of the fluke blades over the visible span, gradient-snapped."""
    ys, xs = np.nonzero(fluke)
    if len(xs) < 400:
        return None
    x0, x1 = xs.min(), xs.max()
    if x1 - x0 < min_span:
        return None
    cols = np.arange(x0, x1 + 1)
    low = np.array([(np.nonzero(fluke[:, x])[0].max() if fluke[:, x].any() else np.nan) for x in cols], float)
    ok = np.isfinite(low)
    if ok.mean() < 0.8:
        return None
    low = np.interp(cols, cols[ok], low[ok])
    path = np.stack([cols, low], 1).astype(float)
    path = resample(snap_to_gradient(resample(path, 4 * N), gray), N)
    return path, {"span": float(x1 - x0), "left": [float(x0), float(low[0])], "right": [float(x1), float(low[-1])], "sharp": edge_sharpness(path, gray)}


def descriptor(path: np.ndarray) -> dict:
    """Chord frame: translate the first point to the origin, rotate the chord (first -> last) onto +y, scale by its
    length; deviation = x in that frame; curvature of the deviation profile at three smoothing scales."""
    chord = path[-1] - path[0]; L = np.linalg.norm(chord)
    if L < 1e-6:
        return None
    ang = np.arctan2(chord[1], chord[0]) - np.pi / 2
    R = np.array([[np.cos(-ang), -np.sin(-ang)], [np.sin(-ang), np.cos(-ang)]])
    q = (path - path[0]) @ R.T / L
    dev = q[:, 0]
    curv = []
    for sigma in (1.5, 3.0, 6.0):
        sm = cv2.GaussianBlur(dev.reshape(-1, 1).astype(np.float32), (0, 0), sigma).ravel()
        curv.append(np.gradient(np.gradient(sm)))
    return {"dev": dev.astype(np.float32), "curv": np.stack(curv, 1).astype(np.float32), "q": q.astype(np.float32)}


def dtw_cost(a: np.ndarray, b: np.ndarray, band: float = 0.12) -> float:
    """Banded DTW mean cost between (n, k) sequences (squared Euclidean step cost)."""
    n = len(a); w = max(int(band * n), 2)
    D = np.full((n + 1, n + 1), np.inf); D[0, 0] = 0.0
    cost = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    for i in range(1, n + 1):
        lo, hi = max(1, i - w), min(n, i + w)
        for j in range(lo, hi + 1):
            D[i, j] = cost[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, n] / (2 * n))


def pair_scores(descs: list[dict], kind: str = "curv", band: float = 0.12, uniq: np.ndarray | None = None) -> np.ndarray:
    """(n, n) similarity = -DTW cost on the curvature (kind curv), the deviation profile (dev) or both; uniq (n_pts,)
    weights the sequence points (population uniqueness, 'less is more')."""
    n = len(descs); S = np.full((n, n), np.nan, np.float32)
    seqs = []
    for d in descs:
        x = d["curv"] * 40.0 if kind == "curv" else (d["dev"][:, None] if kind == "dev" else np.concatenate([d["dev"][:, None], d["curv"] * 40.0], 1))
        if uniq is not None:
            x = x * (0.3 + uniq)[:, None]
        seqs.append(x)
    for i in range(n):
        for j in range(i + 1, n):
            S[i, j] = S[j, i] = -dtw_cost(seqs[i], seqs[j], band)
    return S


def uniqueness(descs: list[dict]) -> np.ndarray:
    X = np.stack([d["dev"] for d in descs]); med = np.median(X, 0); mad = np.median(np.abs(X - med), 0) * 1.4826 + 1e-4
    z = np.abs(X - med) / mad
    return np.clip(np.median(z, 0) / max(np.percentile(np.median(z, 0), 95), 1e-6), 0, 1)   # per-position typical deviation of the population


def extract_day(day_dir: Path, part: str, xr_tag: str = "parts_xr", refined: bool = True) -> list[dict]:
    """refined: use the native-resolution re-segmentation (cluster/reseg.py) where it exists and agrees with the old mask."""
    from .reseg import refined_mask

    src = day_dir / xr_tag
    out = []
    for npz in sorted(src.glob(f"*__{part}.npz")):
        if npz.name.endswith(".reseg.npz"):
            continue
        key = npz.name[: -len(f"__{part}.npz")]
        z = np.load(npz); jpg = src / "charts" / f"{key}__{part}.jpg"
        gray = cv2.imread(str(jpg), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        whale = z["whale"]; rm = refined_mask(npz, part) if refined else None
        if rm is not None:
            whale = rm
        if part == "fin":
            r = fin_trailing_edge(whale, gray, fin_band=z["fin_band"] if "fin_band" in z.files else None)
        else:
            k = max(int(0.03 * whale.shape[0]), 3)
            fl = (whale & (cv2.dilate(z["fluke"].astype(np.uint8), np.ones((k, k), np.uint8)) > 0)) if rm is not None else z["fluke"]
            r = fluke_lower_edge(fl, gray)
        if r is None:
            continue
        path, meta = r; d = descriptor(path)
        if d is None:
            continue
        out.append({"key": key, "part": part, "path": path.astype(np.float32), "meta": meta, **d})
    return out


def main(argv=None) -> int:
    from sklearn.metrics import roc_auc_score

    from .bursts import make_bursts

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--day-dir", type=Path, required=True); p.add_argument("--labels", type=Path, required=True); p.add_argument("--part", default="fin")
    p.add_argument("--min-size", type=float, default=150.0, help="fin height / fluke span in chart px for the 'good edges' subset"); p.add_argument("--out", type=Path, default=None)
    p.add_argument("--max-charts", type=int, default=400); p.add_argument("--min-sharp", type=float, default=0.3, help="edge sharpness floor for the 'good edges' subset")
    p.add_argument("--no-refined", action="store_true", help="use the original segmentation masks even when re-segmentations exist")
    p.add_argument("--same-burst", action="store_true", help="also report the same-burst (consecutive frames) AUC: the consistency of the edge extraction")
    a = p.parse_args(argv)
    day = a.day_dir.name; L = json.loads(a.labels.read_text())["days"][day]["photos"]
    ts = json.loads((a.day_dir / "timestamps.json").read_text()); burst_of = {s: b["id"] for b in make_bursts(ts, 1.5) for s in b["photos"]}
    edges = extract_day(a.day_dir, a.part, refined=not a.no_refined)
    recs = []
    for e in edges:
        stem = e["key"].rsplit("~", 1)[0]
        if "~" in e["key"] and not e["key"].endswith("~0"):
            continue
        ident = L.get(stem.split("__")[-1], "?")
        if ident == "?" or ident.split("_")[1].startswith("O") or stem not in burst_of:
            continue
        e["ident"] = ident; e["burst"] = burst_of[stem]; e["size"] = e["meta"].get("height", e["meta"].get("span", 0.0)); e["sharp"] = e["meta"].get("sharp", 0.0); recs.append(e)
    recs = recs[: a.max_charts]
    print(f"{day} {a.part}: {len(edges)} edges extracted, {len(recs)} labelled", flush=True)
    if len(recs) < 10:
        return 0
    ident = np.array([r["ident"] for r in recs]); burst = np.array([r["burst"] for r in recs]); size = np.array([r["size"] for r in recs]); sharp = np.array([r["sharp"] for r in recs])
    print(f"  sharpness quartiles {np.percentile(sharp, [25, 50, 75]).round(2).tolist()}, size quartiles {np.percentile(size, [25, 50, 75]).round(0).tolist()}", flush=True)
    iu = np.triu_indices(len(recs), 1)
    same_b = (ident[iu[0]] == ident[iu[1]]) & (burst[iu[0]] == burst[iu[1]])
    valid = ~same_b                                                                        # same-burst pairs are not evidence
    y = (ident[iu[0]] == ident[iu[1]])[valid]
    uq = uniqueness(recs)
    results = {}
    for name, kw in (("curv", {"kind": "curv"}), ("dev", {"kind": "dev"}), ("both", {"kind": "both"}), ("curv_uniq", {"kind": "curv", "uniq": uq}), ("curv_band5", {"kind": "curv", "band": 0.05})):
        S = pair_scores(recs, **kw); s = S[iu][valid]
        good = (size[iu[0]] >= a.min_size) & (size[iu[1]] >= a.min_size) & (sharp[iu[0]] >= a.min_sharp) & (sharp[iu[1]] >= a.min_sharp)
        gv = good[valid]
        auc_all = roc_auc_score(y, s) if y.any() and (~y).any() else float("nan")
        auc_good = roc_auc_score(y[gv], s[gv]) if gv.sum() > 30 and y[gv].any() and (~y[gv]).any() else float("nan")
        results[name] = {"auc_all": auc_all, "auc_good": auc_good, "pairs_all": int(len(y)), "pairs_good": int(gv.sum()), "same_good": int(y[gv].sum())}
        extra = ""
        if a.same_burst:
            s_all = S[iu]; diff = ident[iu[0]] != ident[iu[1]]; sel = same_b | diff
            results[name]["auc_same_burst"] = float(roc_auc_score(same_b[sel], s_all[sel])); extra = f"  same-burst vs other animals {results[name]['auc_same_burst']:.3f}"
        print(f"  {name:10s} AUC all {auc_all:.3f} ({len(y)} pairs)  edges >= {a.min_size:.0f} px & sharp >= {a.min_sharp}: {auc_good:.3f} ({int(gv.sum())} pairs, {int(y[gv].sum())} same){extra}", flush=True)
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.out / f"edges_{a.part}.npz", keys=np.array([e["key"] for e in edges]), paths=np.stack([e["path"] for e in edges]), sizes=np.array([e["meta"].get("height", e["meta"].get("span", 0.0)) for e in edges]),
                            sharp=np.array([e["meta"].get("sharp", 0.0) for e in edges]))
        (a.out / f"edge_eval_{a.part}.json").write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
