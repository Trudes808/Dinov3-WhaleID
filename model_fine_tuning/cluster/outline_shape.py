"""Anchored OUTLINE shape descriptor: the native-resolution silhouette between detector landmarks.

The landmark detector (landmarks/parts.py, parts_v2) localises the fluke tips and notch and the fin tip / junctions on
real charts reasonably, but its 24-point edge chains do not transfer from the renders (cluster/shape_channel.py: pair
AUC at chance). The silhouette itself is known precisely from the native re-segmentation (cluster/reseg.py), and what
made the old edge matcher (cluster/edge_match.py) unstable was its endpoints. So: take the refined mask's outline and
cut it at the detector's anchors.

  fluke: the outline arc from the left tip to the right tip that passes the notch (the trailing edge), 64 points by
         arclength, in the tip -> tip frame (origin mid-tips, x along the span, scale = span); the mirrored arc is
         also tried when comparing (underside vs top view)
  fin:   the arc tip -> back junction (trailing edge) and tip -> front junction (leading edge), 32 points each, in the
         back -> front chord frame

Pair score = RMS distance between corresponding points. Evaluation (with --labels): AUC of same-animal vs other-animal
pairs for same-burst pairs (extraction stability), cross-burst pairs (the target) and cross-burst pairs within the
catalog group (where the identity embeddings fail). Writes the z-scored channel into channels.npz as
"oshape:<name>|fin" / "|fluke" with --write, and a sheet of the closest cross-burst pairs with --sheet.

    python -m model_fine_tuning.cluster.outline_shape --day-dir experiments/days_all/2025-02-27 --name parts_v2 \
        --labels model_fine_tuning/splits/labels_all_days.json --sheet docs/outline_pairs/2025-02-27.png --write
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

from .reseg import refined_mask
from .shape_channel import auc, bursts, zscore_similarity

PARTS = ("fin", "fluke")
ANCHORS = {"fluke": ("fluke_tip_left", "fluke_tip_right", "fluke_notch"), "fin": ("dorsal_fin_tip", "dorsal_fin_back", "dorsal_fin_front")}
N_POINTS = {"fluke": 64, "fin": 32}


def outline(mask: np.ndarray, near: np.ndarray) -> np.ndarray | None:
    """Closed contour (n, 2) of the mask component closest to the anchor points."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = [c[:, 0, :].astype(float) for c in cnts if len(c) >= 200]
    if not cnts:
        return None
    best, best_d = None, np.inf
    for c in cnts:
        d = np.mean([np.min(np.linalg.norm(c - p, axis=1)) for p in near])
        if d < best_d:
            best, best_d = c, d
    return best


def arc(contour: np.ndarray, i0: int, i1: int, i_through: int | None, i_avoid: int | None) -> np.ndarray:
    """The arc of the closed contour from index i0 to i1 that contains i_through (or avoids i_avoid)."""
    n = len(contour)
    fwd = np.arange(i0, i0 + ((i1 - i0) % n) + 1) % n
    bwd = np.arange(i0, i0 - ((i0 - i1) % n) - 1, -1) % n
    if i_through is not None:
        pick = fwd if i_through in set(fwd.tolist()) else bwd
    elif i_avoid is not None:
        pick = bwd if i_avoid in set(fwd.tolist()) else fwd
    else:
        pick = fwd if len(fwd) <= len(bwd) else bwd
    return contour[pick]


def resample(path: np.ndarray, n: int) -> np.ndarray:
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1); s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return np.repeat(path[:1], n, 0)
    t = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(t, s, path[:, 0]), np.interp(t, s, path[:, 1])], 1)


def descriptor(rec: dict, mask: np.ndarray, part: str, min_conf: float, max_snap: float, allow_hidden: bool = False, top_fraction: float = 1.0) -> dict | None:
    """top_fraction < 1 (fin): keep only the part of each edge above that fraction of the tip's height over the chord,
    where the shape is distinctive and the fuzzy junction endpoints matter least."""
    lm = rec["landmarks"]
    names = ANCHORS[part]
    ok_states = ("visible", "hidden") if allow_hidden else ("visible",)
    for n in names:
        if n not in lm or lm[n]["state"] not in ok_states or lm[n]["conf"] < min_conf:
            return None
    pts = np.array([lm[n]["uv"] for n in names], float)
    cnt = outline(mask, pts)
    if cnt is None:
        return None
    idx, snap = [], []
    for p in pts:
        d = np.linalg.norm(cnt - p, axis=1); i = int(np.argmin(d)); idx.append(i); snap.append(float(d[i]))
    if part == "fluke":
        a, b = cnt[idx[0]], cnt[idx[1]]; span = np.linalg.norm(b - a)
        if span < 50 or max(snap) > max_snap * span:
            return None
        path = arc(cnt, idx[0], idx[1], i_through=idx[2], i_avoid=None)
        ex = (b - a) / span; ey = np.array([-ex[1], ex[0]]); origin = (a + b) / 2
        local = ((resample(path, N_POINTS[part]) - origin) @ np.stack([ex, ey], 1)) / span
        notch = ((cnt[idx[2]] - origin) @ ey) / span
        if notch < 0:
            local[:, 1] *= -1.0
        return {"pts": local, "span": float(span), "snap": snap, "n": len(path)}
    tip, back, front = cnt[idx[0]], cnt[idx[1]], cnt[idx[2]]; chord = np.linalg.norm(front - back)
    if chord < 30 or max(snap) > max_snap * chord:
        return None
    trailing = arc(cnt, idx[0], idx[1], i_through=None, i_avoid=idx[2])
    leading = arc(cnt, idx[0], idx[2], i_through=None, i_avoid=idx[1])
    if len(trailing) > len(cnt) * 0.6 or len(leading) > len(cnt) * 0.6:
        return None                                       # went the long way round the whole body
    ex = (front - back) / chord; ey = np.array([-ex[1], ex[0]]); origin = (back + front) / 2
    R = np.stack([ex, ey], 1)
    sign = -1.0 if ((tip - origin) @ ey) < 0 else 1.0
    def local_arc(path):
        q = ((path - origin) @ R) / chord; q[:, 1] *= sign
        if top_fraction < 1.0:
            h_tip = q[0, 1]
            if h_tip <= 0.05:
                return None
            keep = q[:, 1] >= (1.0 - top_fraction) * h_tip
            last = int(np.argmin(keep)) if not keep.all() else len(q)      # first point below the cut, from the tip
            q = q[: max(last, 3)]
        return resample(q, N_POINTS[part])
    t_arc, l_arc = local_arc(trailing), local_arc(leading)
    if t_arc is None or l_arc is None:
        return None
    return {"pts": np.concatenate([t_arc, l_arc]), "span": float(chord), "snap": snap, "n": len(trailing) + len(leading)}


def mirror(d: dict) -> dict:
    pts = d["pts"][::-1].copy(); pts[:, 0] *= -1.0
    return {**d, "pts": pts}


def procrustes_rms(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """RMS distance after the best similarity transform (rotation, scale, translation) of each B[j] onto A.
    A: (n, 2), B: (m, n, 2). Umeyama in closed form, vectorised over m."""
    A0 = A - A.mean(0); B0 = B - B.mean(1, keepdims=True)
    H = np.einsum("mni,nj->mij", B0, A0) / A.shape[0]                 # (m, 2, 2) cross-covariance
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    S_adj = S.copy(); S_adj[:, 1] *= d
    varB = np.mean(np.sum(B0 * B0, -1), -1)
    scale = S_adj.sum(1) / np.maximum(varB, 1e-12)
    R = np.einsum("mij,mjk->mik", U, Vt)                              # B -> A rotation (acting on row vectors: x @ R)
    sign_fix = np.ones_like(S); sign_fix[:, 1] = d
    R = np.einsum("mij,mj,mjk->mik", U, sign_fix, Vt)
    Bt = np.einsum("mni,mij->mnj", B0, R) * scale[:, None, None]
    return np.sqrt(np.mean(np.sum((Bt - A0[None]) ** 2, -1), -1))


def distance_matrix(descs: dict, keys: list[str], part: str, align: str = "none") -> np.ndarray:
    pos = {k: i for i, k in enumerate(keys)}
    items = [(pos[k], d) for (k, p), d in descs.items() if p == part]
    n = len(keys); D = np.full((n, n), np.nan, np.float32)
    P = np.stack([d["pts"] for _, d in items]) if items else None
    if P is None:
        return D
    Pm = np.stack([mirror(d)["pts"] for _, d in items]) if part == "fluke" else None
    ids = np.array([i for i, _ in items])
    for a in range(len(items) - 1):
        if align == "procrustes":
            d = procrustes_rms(P[a], P[a + 1:])
            if Pm is not None:
                d = np.minimum(d, procrustes_rms(P[a], Pm[a + 1:]))
        else:
            diff = P[a][None] - P[a + 1:]; d = np.sqrt(np.mean(np.sum(diff * diff, -1), -1))
            if Pm is not None:
                dm = np.sqrt(np.mean(np.sum((P[a][None] - Pm[a + 1:]) ** 2, -1), -1)); d = np.minimum(d, dm)
        D[ids[a], ids[a + 1:]] = d; D[ids[a + 1:], ids[a]] = d
    return D


def evaluate(D: np.ndarray, keys: list[str], photos: dict, ts: dict) -> dict:
    burst_of = bursts(ts)
    ident = np.array([photos.get(k.rsplit("~", 1)[0].split("__")[-1], "?") if k.endswith("~0") or "~" not in k else "?" for k in keys])
    burst = np.array([burst_of.get(k.rsplit("~", 1)[0], -1) for k in keys])
    group = np.array([i.split("_")[1][:3] if "_" in i else "?" for i in ident])
    n = len(keys); iu = np.triu_indices(n, 1); d = D[iu]
    known = np.isfinite(d) & (ident[iu[0]] != "?") & (ident[iu[1]] != "?") & (burst[iu[0]] >= 0)
    known &= ~np.char.startswith(np.char.partition(ident[iu[0]], "_")[:, 2], "O") & ~np.char.startswith(np.char.partition(ident[iu[1]], "_")[:, 2], "O")
    same = ident[iu[0]] == ident[iu[1]]; same_burst = burst[iu[0]] == burst[iu[1]]; same_group = group[iu[0]] == group[iu[1]]
    s = -d
    return {"pairs": int(known.sum()),
            "auc_same_burst_vs_other_animals": auc(s[known & same & same_burst], s[known & ~same]),
            "auc_cross_burst": auc(s[known & same & ~same_burst], s[known & ~same & ~same_burst]),
            "auc_cross_burst_within_group": auc(s[known & same & ~same_burst], s[known & ~same & same_group & ~same_burst]),
            "positives_cross_burst": int((known & same & ~same_burst).sum()), "negatives_within_group": int((known & ~same & same_group & ~same_burst).sum())}


def sheet(descs: dict, D: np.ndarray, keys: list[str], part: str, day_dir: Path, preds: dict, photos: dict, ts: dict, out: Path, count: int = 10, tile: int = 320) -> None:
    pos = {k: i for i, k in enumerate(keys)}; burst_of = bursts(ts)
    ks = [k for (k, p) in descs if p == part]
    cand = []
    for a in range(len(ks)):
        for b in range(a + 1, len(ks)):
            ka, kb = ks[a], ks[b]
            if burst_of.get(ka.rsplit("~", 1)[0], -1) == burst_of.get(kb.rsplit("~", 1)[0], -2):
                continue
            v = D[pos[ka], pos[kb]]
            if np.isfinite(v):
                cand.append((float(v), ka, kb))
    cand.sort(); rows = []
    for v, ka, kb in cand[:count]:
        tiles = []
        for k in (ka, kb):
            img = cv2.imread(str(day_dir / "parts_xr" / "charts" / f"{k}__{part}.jpg"))
            if img is None:
                continue
            lm = preds[(k, part)]["landmarks"]
            for n in ANCHORS[part]:
                cv2.circle(img, tuple(int(t) for t in lm[n]["uv"]), 28, (0, 255, 0), 6)
            sc = tile / max(img.shape[:2]); tiles.append(cv2.resize(img, (int(img.shape[1] * sc), int(img.shape[0] * sc))))
        if len(tiles) == 2:
            ia, ib = photos.get(ka.rsplit("~", 1)[0].split("__")[-1], "?"), photos.get(kb.rsplit("~", 1)[0].split("__")[-1], "?")
            lab = "SAME" if ia == ib != "?" else "diff"
            row = np.concatenate(tiles, 1); cv2.putText(row, f"d={v:.3f} {lab} {ia[-6:]} vs {ib[-6:]}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2); rows.append(row)
    if rows:
        w = max(r.shape[1] for r in rows); rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
        out.parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(out), np.concatenate(rows, 0))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--day-dir", type=Path, required=True); p.add_argument("--name", default="parts_v2"); p.add_argument("--tag", default="cluster_v3")
    p.add_argument("--labels", type=Path, default=None); p.add_argument("--sheet", type=Path, default=None); p.add_argument("--write", action="store_true")
    p.add_argument("--min-conf", type=float, default=0.08); p.add_argument("--max-snap", type=float, default=0.06, help="anchor to outline distance allowed, fraction of the span / chord")
    p.add_argument("--align", default="none", choices=["none", "procrustes"], help="procrustes: best similarity transform per pair before the RMS");
    p.add_argument("--allow-hidden", action="store_true"); p.add_argument("--top-fraction", type=float, default=1.0); p.add_argument("--channel", default=None, help="channel name (default oshape:<name>)")
    a = p.parse_args(argv)
    day = a.day_dir.name; chan_file = a.day_dir / a.tag / "channels.npz"
    z = np.load(chan_file, allow_pickle=True); keys = [str(k) for k in z["keys"]]; ch = {k: z[k] for k in z.files if k != "keys"}
    ts = json.loads((a.day_dir / "timestamps.json").read_text())
    photos = json.loads(a.labels.read_text())["days"].get(day, {}).get("photos", {}) if a.labels else {}
    pdir = a.day_dir / f"part_landmarks_{a.name}"; report = {"day": day}
    preds, descs = {}, {}
    for part in PARTS:
        n_pred = n_mask = 0
        for k in keys:
            f = pdir / f"{k}__{part}.json"; npz = a.day_dir / "parts_xr" / f"{k}__{part}.npz"
            if not f.exists() or not npz.exists():
                continue
            rec = json.loads(f.read_text()); preds[(k, part)] = rec; n_pred += 1
            mask = refined_mask(npz, part)
            if mask is None:
                mask = np.load(npz)["whale"]
            n_mask += 1
            d = descriptor(rec, mask, part, a.min_conf, a.max_snap, a.allow_hidden, a.top_fraction)
            if d is not None:
                descs[(k, part)] = d
        D = distance_matrix(descs, keys, part, a.align)
        rep = {"predictions": n_pred, "descriptors": sum(1 for (k, pp) in descs if pp == part)}
        if photos:
            rep.update(evaluate(D, keys, photos, ts))
        report[part] = rep; print(f"{day} {part}: {json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rep.items()})}", flush=True)
        ch[f"{a.channel or 'oshape:' + a.name}|{part}"] = zscore_similarity(D)
        if a.sheet and rep["descriptors"] >= 2:
            sheet(descs, D, keys, part, a.day_dir, preds, photos, ts, a.sheet.with_name(f"{a.sheet.stem}_{part}{a.sheet.suffix}"))
    if a.write:
        tmp = chan_file.with_suffix(".tmp.npz"); np.savez_compressed(tmp, keys=np.array(keys), **ch); os.replace(tmp, chan_file); print(f"wrote oshape:{a.name} channels into {chan_file}")
    (a.day_dir / a.tag / f"{(a.channel or 'oshape:' + a.name).replace(':', '_')}.json").write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
