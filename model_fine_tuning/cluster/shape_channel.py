"""Fin / fluke SHAPE channel for the day clustering, from the landmark detector's predictions.

The detector (landmarks/parts.py predict) writes, per native part chart, the tip / junction landmarks and the
trailing- / leading-edge chains with a confidence and a visibility state. Here each part becomes a shape descriptor
in a landmark-defined frame, so it is invariant to the chart's residual scale and rotation:

  fin:   origin = midpoint of the back and front junctions, x along back -> front, scale = that chord;
         points = the 12 trailing-edge + 12 leading-edge chain points (tip first, junction last)
  fluke: origin = midpoint of the two tips, x along left tip -> right tip, scale = the span;
         points = the 24 trailing-edge chain points (left tip -> right tip); a fluke seen from below is the mirror
         image of the same fluke seen from above, so the mirrored descriptor is also tried and the better match kept

Pair distance = RMS distance between corresponding chain points that are visible and confident in both (at least
--min-points of them); the channel is the z-scored negative distance, written into the day's channels.npz under
"shape:<name>|fin" and "shape:<name>|fluke" (NaN where either side has no usable prediction), next to the identity
channels, so cluster_day.py can use it with --use "...,shape:<name>".

With --labels it also reports the pair AUC of the shape distance on cross-burst pairs (all, and within the catalog
group, which is where the identity embeddings fail) and writes a sheet of the closest pairs per part for a look.

    python -m model_fine_tuning.cluster.shape_channel --day-dir experiments/days_all/2025-02-27 --name parts_v2 \
        --labels model_fine_tuning/splits/labels_all_days.json --sheet docs/shape_pairs_2025-02-27.png
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

PARTS = ("fin", "fluke")
FIN_FRAME = ("dorsal_fin_back", "dorsal_fin_front")
FIN_CHAIN = [f"fin_te_{i:02d}" for i in range(12)] + [f"fin_le_{i:02d}" for i in range(12)]
FLUKE_FRAME = ("fluke_tip_left", "fluke_tip_right")
FLUKE_CHAIN = [f"fluke_te_{i:02d}" for i in range(24)]


def load_predictions(day_dir: Path, name: str, keys: list[str]) -> dict:
    """{(key, part): record} for the instances that have a prediction file."""
    d = day_dir / f"part_landmarks_{name}"
    out = {}
    for k in keys:
        for part in PARTS:
            f = d / f"{k}__{part}.json"
            if f.exists():
                out[(k, part)] = json.loads(f.read_text())
    return out


def descriptor(rec: dict, part: str, min_conf: float, frame_min_conf: float) -> dict | None:
    """Normalised chain points (n, 2) with a validity mask; None when the frame landmarks are not usable."""
    lm = rec["landmarks"]
    frame_names, chain = (FIN_FRAME, FIN_CHAIN) if part == "fin" else (FLUKE_FRAME, FLUKE_CHAIN)
    for n in frame_names:
        if n not in lm or lm[n]["state"] != "visible" or lm[n]["conf"] < frame_min_conf:
            return None
    a = np.array(lm[frame_names[0]]["uv"], float); b = np.array(lm[frame_names[1]]["uv"], float)
    chord = np.linalg.norm(b - a)
    if chord < 1e-3:
        return None
    ex = (b - a) / chord; ey = np.array([-ex[1], ex[0]])
    origin = (a + b) / 2.0
    pts = np.array([lm[n]["uv"] if n in lm else [np.nan, np.nan] for n in chain], float)
    valid = np.array([n in lm and lm[n]["state"] == "visible" and lm[n]["conf"] >= min_conf for n in chain])
    rel = pts - origin
    local = np.stack([rel @ ex, rel @ ey], 1) / chord
    # orient y so that the part points "up": the tip (fin) / the notch side (fluke) has positive y
    ref = lm.get("dorsal_fin_tip" if part == "fin" else "fluke_notch")
    if ref is not None:
        y_ref = ((np.array(ref["uv"], float) - origin) @ ey) / chord
        if y_ref < 0:
            local[:, 1] *= -1.0
    return {"pts": local, "valid": valid, "chord": float(chord), "n_valid": int(valid.sum())}


def mirror(desc: dict) -> dict:
    """The same part seen from the other side: x -> -x and the chain order reversed (fluke only)."""
    pts = desc["pts"][::-1].copy(); pts[:, 0] *= -1.0
    return {**desc, "pts": pts, "valid": desc["valid"][::-1].copy()}


def pair_distance(da: dict, db: dict, min_points: int) -> float:
    both = da["valid"] & db["valid"]
    if both.sum() < min_points:
        return np.nan
    d = da["pts"][both] - db["pts"][both]
    return float(np.sqrt(np.mean(np.sum(d * d, 1))))


def shape_matrix(descs: dict, keys: list[str], part: str, min_points: int, try_mirror: bool) -> tuple[np.ndarray, np.ndarray]:
    """(n, n) RMS shape distance (NaN where unavailable) and the index of instances with a descriptor."""
    pos = {k: i for i, k in enumerate(keys)}
    idx = np.array([pos[k] for (k, p) in descs if p == part], int)
    n = len(keys); D = np.full((n, n), np.nan, np.float32)
    items = [(pos[k], d) for (k, p), d in descs.items() if p == part]
    mirrored = {i: mirror(d) for i, d in items} if try_mirror else {}
    for a in range(len(items)):
        ia, da = items[a]
        for b in range(a + 1, len(items)):
            ib, db = items[b]
            v = pair_distance(da, db, min_points)
            if try_mirror:
                vm = pair_distance(da, mirrored[ib], min_points)
                v = np.nanmin([v, vm]) if np.isfinite(vm) or np.isfinite(v) else np.nan
            D[ia, ib] = D[ib, ia] = v
    return D, idx


def zscore_similarity(D: np.ndarray) -> np.ndarray:
    S = -D
    v = S[np.isfinite(S)]
    return ((S - v.mean()) / (v.std() + 1e-9)).astype(np.float32) if len(v) > 1 else S.astype(np.float32)


# ------------------------------------------------------------------ evaluation against the catalog
def bursts(ts: dict, gap: float = 1.5) -> dict:
    by_cam: dict = {}
    for stem, t in ts.items():
        if t.get("time") is not None:
            by_cam.setdefault((t.get("folder", "?"), t.get("camera", "?")), []).append((float(t["time"]), stem))
    burst_of, k = {}, 0
    for items in by_cam.values():
        items.sort(); prev = None
        for t, stem in items:
            if prev is not None and t - prev > gap:
                k += 1
            burst_of[stem] = k; prev = t
        k += 1
    return burst_of


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def evaluate(D: np.ndarray, keys: list[str], day: str, labels_file: Path, ts: dict) -> dict:
    lab = json.loads(labels_file.read_text())["days"].get(day, {}); photos = lab.get("photos", {})
    burst_of = bursts(ts)
    ident, burst = [], []
    for k in keys:
        stem = k.rsplit("~", 1)[0]
        ident.append(photos.get(stem.split("__")[-1], "?") if not k.endswith(("~1", "~2", "~3")) else "?")
        burst.append(burst_of.get(stem, -1))
    ident = np.array(ident); burst = np.array(burst)
    n = len(keys); iu = np.triu_indices(n, 1)
    d = D[iu]; ok = np.isfinite(d) & (ident[iu[0]] != "?") & (ident[iu[1]] != "?") & (burst[iu[0]] != burst[iu[1]]) & (burst[iu[0]] >= 0)
    ok &= ~np.char.startswith(np.char.partition(ident[iu[0]], "_")[:, 2], "O") & ~np.char.startswith(np.char.partition(ident[iu[1]], "_")[:, 2], "O")
    same = ident[iu[0]] == ident[iu[1]]
    group = np.array([i.split("_")[1][:3] if "_" in i else "?" for i in ident])
    same_group = group[iu[0]] == group[iu[1]]
    s = -d
    return {"pairs": int(ok.sum()), "positives": int((ok & same).sum()),
            "auc_all": auc(s[ok & same], s[ok & ~same]),
            "auc_within_group": auc(s[ok & same], s[ok & same_group & ~same]),
            "negatives_within_group": int((ok & same_group & ~same).sum())}


def pair_sheet(descs: dict, D: np.ndarray, keys: list[str], part: str, day_dir: Path, out: Path, ident: dict, count: int = 12, tile: int = 300) -> None:
    """The closest pairs by shape distance: chart crops side by side with the chain drawn, labelled same / different."""
    import cv2
    pos = {k: i for i, k in enumerate(keys)}
    items = [(k, d) for (k, p), d in descs.items() if p == part]
    cand = []
    for a in range(len(items)):
        for b in range(a + 1, len(items)):
            v = D[pos[items[a][0]], pos[items[b][0]]]
            if np.isfinite(v):
                cand.append((v, items[a][0], items[b][0]))
    cand.sort()
    rows = []
    for v, ka, kb in cand[:count]:
        tiles = []
        for k in (ka, kb):
            img = cv2.imread(str(day_dir / "parts_xr" / "charts" / f"{k}__{part}.jpg"))
            if img is None:
                continue
            sc = tile / max(img.shape[:2]); img = cv2.resize(img, (int(img.shape[1] * sc), int(img.shape[0] * sc)))
            d = descs[(k, part)]
            # the chain in chart pixels (reverse the normalisation) for the eye
            tiles.append(img)
        if len(tiles) == 2:
            lab = "SAME" if ident.get(ka, "?") == ident.get(kb, "?") != "?" else ("diff" if "?" not in (ident.get(ka, "?"), ident.get(kb, "?")) else "?")
            row = np.concatenate(tiles, 1)
            cv2.putText(row, f"d={v:.3f} {lab}  {ident.get(ka, '?')[-6:]} vs {ident.get(kb, '?')[-6:]}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            rows.append(row)
    if rows:
        w = max(r.shape[1] for r in rows); rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
        cv2.imwrite(str(out), np.concatenate(rows, 0))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--day-dir", type=Path, required=True); p.add_argument("--name", required=True, help="detector run name: predictions in part_landmarks_<name>/")
    p.add_argument("--tag", default="cluster_v3"); p.add_argument("--labels", type=Path, default=None); p.add_argument("--sheet", type=Path, default=None)
    p.add_argument("--min-conf", type=float, default=0.05); p.add_argument("--frame-min-conf", type=float, default=0.05); p.add_argument("--min-points", type=int, default=8)
    p.add_argument("--no-write", action="store_true"); p.add_argument("--channel-name", default=None, help="default shape:<name>")
    a = p.parse_args(argv)
    day = a.day_dir.name; chan_file = a.day_dir / a.tag / "channels.npz"
    z = np.load(chan_file, allow_pickle=True); keys = [str(k) for k in z["keys"]]; ch = {k: z[k] for k in z.files if k != "keys"}
    preds = load_predictions(a.day_dir, a.name, keys)
    descs = {kp: d for kp, r in preds.items() if (d := descriptor(r, kp[1], a.min_conf, a.frame_min_conf)) is not None}
    ts = json.loads((a.day_dir / "timestamps.json").read_text())
    name = a.channel_name or f"shape:{a.name}"
    report = {"day": day, "name": name}
    ident = {}
    if a.labels:
        photos = json.loads(a.labels.read_text())["days"].get(day, {}).get("photos", {})
        ident = {k: photos.get(k.rsplit("~", 1)[0].split("__")[-1], "?") for k in keys}
    for part in PARTS:
        n_pred = sum(1 for (k, pp) in preds if pp == part); n_desc = sum(1 for (k, pp) in descs if pp == part)
        D, idx = shape_matrix(descs, keys, part, a.min_points, try_mirror=(part == "fluke"))
        finite = int(np.isfinite(D[np.triu_indices(len(keys), 1)]).sum())
        report[part] = {"predictions": n_pred, "descriptors": n_desc, "pairs_scored": finite}
        if a.labels:
            report[part].update(evaluate(D, keys, day, a.labels, ts))
        ch[f"{name}|{part}"] = zscore_similarity(D)
        if a.sheet and n_desc >= 2:
            pair_sheet(descs, D, keys, part, a.day_dir, a.sheet.with_name(f"{a.sheet.stem}_{part}{a.sheet.suffix}"), ident)
        print(f"{day} {part}: {json.dumps(report[part])}", flush=True)
    if not a.no_write:
        tmp = chan_file.with_suffix(".tmp.npz"); np.savez_compressed(tmp, keys=np.array(keys), **ch); os.replace(tmp, chan_file)
        print(f"wrote {name}|fin, {name}|fluke into {chan_file}")
    (a.day_dir / a.tag / f"shape_{a.name}.json").write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
