"""Dorsal-fin silhouette matching on the oriented charts (chart.py output: head right, fin up).

The fin is isolated from the whale mask as the protrusion above the back: a wide horizontal
morphological opening removes it, the difference is the fin blob. Its outline, resampled by
arclength from the leading base corner over the tip to the trailing base corner and
normalised by the fin height, is the descriptor (DARWIN / Finscan style); the trailing half
is compared on its own too, and DINOv3's CLS token of a tight crop of the fin as a learned
shape+texture summary. Scores are written in match.py's layout (sources finshape_outline,
finshape_trailing, finshape_cls).

    python -m model_fine_tuning.cluster.finshape --charts CHARTS --timestamps TS --out OUT [--size 768]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .match import time_term


def isolate_fin(whale: np.ndarray, open_frac: float = 0.3, min_area: int = 300) -> np.ndarray | None:
    """Fin blob = whale mask minus its opening with a wide horizontal kernel (the back survives
    the opening, the fin does not). Keeps the component whose top is highest."""
    size = whale.shape[1]
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(open_frac * size)) | 1, 3))
    opened = cv2.morphologyEx(whale.astype(np.uint8), cv2.MORPH_OPEN, k) > 0
    blob = whale & ~opened
    n, lab, stats, _ = cv2.connectedComponentsWithStats(blob.astype(np.uint8), connectivity=8)
    best, best_top = None, None
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        top = stats[i, cv2.CC_STAT_TOP]
        if best is None or top < best_top:
            best, best_top = i, top
    return (lab == best) if best is not None else None


def outline_descriptor(fin: np.ndarray, n: int = 64) -> tuple[np.ndarray, np.ndarray, dict] | None:
    cnt, _ = cv2.findContours(fin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnt:
        return None
    c = max(cnt, key=cv2.contourArea)[:, 0, :].astype(float)
    ys, xs = np.nonzero(fin)
    tip = np.array([xs[np.argmin(ys)], ys.min()], float)
    base_y = ys.max()
    height = base_y - tip[1]
    if height < 20:
        return None
    # base corners: leftmost and rightmost contour points within the bottom 15 % of the fin
    bottom = c[c[:, 1] >= base_y - 0.15 * height]
    lead = bottom[np.argmax(bottom[:, 0])]      # head is on the right: leading edge base
    trail = bottom[np.argmin(bottom[:, 0])]     # trailing edge base
    i_lead = int(np.argmin(np.linalg.norm(c - lead, axis=1))); i_tip = int(np.argmin(np.linalg.norm(c - tip, axis=1))); i_trail = int(np.argmin(np.linalg.norm(c - trail, axis=1)))
    # walk from lead to trail passing through the tip (choose the direction that reaches the tip first)
    def walk(a, b, step):
        out = [c[a]]; i = a
        for _ in range(len(c)):
            i = (i + step) % len(c); out.append(c[i])
            if i == b:
                break
        return np.array(out)
    p1, p2 = walk(i_lead, i_trail, 1), walk(i_lead, i_trail, -1)
    def passes_tip(p):
        return np.min(np.linalg.norm(p - tip, axis=1)) < 2.0
    path = p1 if passes_tip(p1) and (not passes_tip(p2) or len(p1) <= len(p2)) else p2
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    if seg[-1] < 10:
        return None
    t = np.linspace(0, seg[-1], n)
    rs = np.stack([np.interp(t, seg, path[:, 0]), np.interp(t, seg, path[:, 1])], 1)
    norm = (rs - tip) / height
    k_tip = int(np.argmin(np.linalg.norm(rs - tip, axis=1)))
    trailing = rs[k_tip:]
    tseg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(trailing, axis=0), axis=1))] if len(trailing) > 1 else np.array([0.0])
    tt = np.linspace(0, tseg[-1], 32)
    trail_rs = np.stack([np.interp(tt, tseg, trailing[:, 0]), np.interp(tt, tseg, trailing[:, 1])], 1) if len(trailing) > 1 else np.full((32, 2), np.nan)
    stats = {"height": float(height), "base_width": float(lead[0] - trail[0]), "tip": tip.tolist(), "aspect": float((lead[0] - trail[0]) / height),
             "tip_offset": float((tip[0] - 0.5 * (lead[0] + trail[0])) / height)}
    return norm.reshape(-1).astype(np.float32), ((trail_rs - tip) / height).reshape(-1).astype(np.float32), stats


def tight_crop(chart: np.ndarray, fin: np.ndarray, size: int = 224, margin: float = 0.25) -> np.ndarray:
    ys, xs = np.nonzero(fin)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    half = max(xs.max() - xs.min(), ys.max() - ys.min()) * (0.5 + margin) + 4
    M = np.array([[size / (2 * half), 0, size / 2 - cx * size / (2 * half)], [0, size / (2 * half), size / 2 - cy * size / (2 * half)]], np.float64)
    return cv2.warpAffine(chart, M, (size, size), flags=cv2.INTER_AREA)


def main(argv=None) -> int:
    import torch

    from ..common.backbone import DinoFeatures, normalize_image
    from ..common.data import to_tensor

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--charts", type=Path, required=True); p.add_argument("--timestamps", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = DinoFeatures("vitb16", layers=[11]).to(device).eval()
    ts = json.loads(args.timestamps.read_text())
    args.out.mkdir(parents=True, exist_ok=True); (args.out / "fins").mkdir(exist_ok=True)
    recs = {}
    for f in sorted(args.charts.glob("*.npz")):
        z = np.load(f)
        chart = cv2.cvtColor(cv2.imread(str(args.charts / "charts" / f"{f.stem}.jpg")), cv2.COLOR_BGR2RGB)
        fin = isolate_fin(z["whale"])
        if fin is None:
            continue
        d = outline_descriptor(fin)
        if d is None:
            continue
        outline, trailing, stats = d
        crop = tight_crop(chart, fin)
        with torch.no_grad():
            x = normalize_image(to_tensor(crop)[None].to(device))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                _, cls = backbone(x)
        cls = cls[0].float().cpu().numpy(); cls /= max(np.linalg.norm(cls), 1e-6)
        vis = crop.copy()
        fin_c = tight_crop(fin.astype(np.uint8) * 255, fin)
        cnt, _ = cv2.findContours((fin_c > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(vis, cnt, -1, (255, 255, 0), 1)
        cv2.imwrite(str(args.out / "fins" / f"{f.stem}.jpg"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        recs[f.stem] = {"outline": outline, "trailing": trailing, "cls": cls, "view_side": int(z["view_side"]), **stats}
    stems = sorted(recs); n = len(stems)
    print(f"fin silhouettes for {n} of {len(list(args.charts.glob('*.npz')))} charts", flush=True)
    t = np.array([ts[s]["time"] for s in stems]); cam = np.array([ts[s]["camera"] for s in stems]); dt = np.abs(t[:, None] - t[None, :])
    side = np.array([recs[s]["view_side"] for s in stems])
    same_view = (side[:, None] == side[None, :]) | (side[:, None] == 0) | (side[None, :] == 0)
    O = np.stack([recs[s]["outline"] for s in stems]); T = np.stack([recs[s]["trailing"] for s in stems]); C = np.stack([recs[s]["cls"] for s in stems])
    A = np.array([[recs[s]["aspect"], recs[s]["tip_offset"]] for s in stems])
    out = {"dt_s": dt, "time_term": time_term(dt), "same_camera": cam[:, None] == cam[None, :]}
    scores = {"finshape_outline": -np.sqrt(np.nanmean((O[:, None, :] - O[None, :, :]) ** 2, axis=-1)),
              "finshape_trailing": -np.sqrt(np.nanmean((T[:, None, :] - T[None, :, :]) ** 2, axis=-1)),
              "finshape_cls": C @ C.T,
              "finshape_aspect": -np.linalg.norm(A[:, None, :] - A[None, :, :], axis=-1)}
    for k, m in scores.items():
        m = np.where(same_view, m, np.nan); off = ~np.eye(n, dtype=bool) & np.isfinite(m)
        z = (m - m[off].mean()) / (m[off].std() + 1e-9)
        out[f"{k}__same"] = z; out[f"{k}__flip"] = np.full((n, n), np.nan); out[f"{k}__n_same"] = np.isfinite(z).astype(np.int32); out[f"{k}__n_flip"] = np.zeros((n, n), np.int32)
    np.savez_compressed(args.out / "scores.npz", stems=np.array(stems), **out)
    json.dump({"sources": list(scores), "calib": {}}, open(args.out / "match_meta.json", "w"))
    json.dump({s: {k: v for k, v in r.items() if k not in ("outline", "trailing", "cls")} for s, r in recs.items()}, open(args.out / "fin_stats.json", "w"), indent=1)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
