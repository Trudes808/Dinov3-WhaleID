"""Apply the identity embedding to one day and score it: part-wise pair AUC, then photo-pair
scores in match.py's layout so cluster.py / explain.py run unchanged.

    python -m model_fine_tuning.cluster.eval_id --checkpoint experiments/id_v1/checkpoints/best.pt --day-dir experiments/days/2024-12-18 \
        --labels splits/labels_all_days.json --out experiments/days/2024-12-18/cluster_id_v1 [--profile-min 0.6]

Photo-pair appearance score = the best over the part types both photos have of the cosine
between their embeddings (fin pairs only when both are near-profile; fin and body charts are
also compared against the other photo's *mirrored* chart, the "flip" hypothesis, because the
resolved flank can be wrong), z-scored over the day. Explanations: for a linked pair, the two
charts side by side with the part and cosine, and a patch-level similarity map from the
backbone (position-aligned cosine minus the misaligned mean), the same reading as before.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .match import time_term
from .train_id import PARTS, WhaleIdModel


@torch.no_grad()
def embed_charts(model, paths_: list[Path], parts: list[int], device, size: int, mirror: bool = False, batch: int = 32) -> np.ndarray:
    out = []
    for i in range(0, len(paths_), batch):
        imgs = []
        for pth in paths_[i: i + batch]:
            img = cv2.cvtColor(cv2.imread(str(pth), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            if mirror:
                img = img[:, ::-1]
            imgs.append(np.ascontiguousarray(img.transpose(2, 0, 1)) / 255.0)
        x = torch.from_numpy(np.stack(imgs)).float().to(device); part = torch.tensor(parts[i: i + batch], device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            z = model.embed(x, part)
        out.append(F.normalize(z.float(), dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, model.config["dim"]))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--day-dir", type=Path, required=True)
    p.add_argument("--labels", type=Path, default=None, help="labels_all_days.json, for the AUC report only")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--size", type=int, default=448)
    p.add_argument("--profile-min", type=float, default=0.6)
    args = p.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = WhaleIdModel.from_checkpoint(args.checkpoint, device)
    day = args.day_dir.name
    ts = json.loads((args.day_dir / "timestamps.json").read_text())
    stems = sorted(ts)
    idx = {s: i for i, s in enumerate(stems)}
    n = len(stems)
    # charts per photo
    charts = {part: {} for part in PARTS}
    meta = {}
    for npz in (args.day_dir / "parts").glob("*__*.npz"):
        stem, part = npz.name[: -4].rsplit("__", 1)
        if stem in idx:
            charts[part][stem] = args.day_dir / "parts" / "charts" / f"{stem}__{part}.jpg"
            z = np.load(npz); meta[(stem, part)] = {"profile": float(z["profile"]), "view_side": int(z["view_side"])}
    t = np.array([ts[s]["time"] for s in stems], float); cam = np.array([ts[s]["camera"] for s in stems])
    dt = np.abs(t[:, None] - t[None, :])
    out = {"dt_s": dt, "time_term": time_term(dt), "same_camera": cam[:, None] == cam[None, :]}
    best_same = np.full((n, n), np.nan); best_flip = np.full((n, n), np.nan)
    emb_store = {}
    for k, part in enumerate(PARTS):
        items = sorted(charts[part])
        if len(items) < 2:
            continue
        paths_ = [charts[part][s] for s in items]
        Z = embed_charts(model, paths_, [k] * len(items), device, args.size)
        Zm = embed_charts(model, paths_, [k] * len(items), device, args.size, mirror=True) if part != "fluke" else None
        ii = np.array([idx[s] for s in items])
        S = Z @ Z.T
        if part == "fin":
            prof = np.array([meta[(s, part)]["profile"] for s in items])
            ok = (prof >= args.profile_min)[:, None] & (prof >= args.profile_min)[None, :]
            S = np.where(ok, S, np.nan)
        m = np.full((n, n), np.nan); m[np.ix_(ii, ii)] = S
        out[f"id_{part}__same"] = m; out[f"id_{part}__n_same"] = np.isfinite(m).astype(np.int32)
        if Zm is not None:
            Sf = Z @ Zm.T
            if part == "fin":
                Sf = np.where(ok, Sf, np.nan)
            mf = np.full((n, n), np.nan); mf[np.ix_(ii, ii)] = Sf
        else:
            mf = np.full((n, n), np.nan)
        out[f"id_{part}__flip"] = mf; out[f"id_{part}__n_flip"] = np.isfinite(mf).astype(np.int32)
        best_same = np.fmax(best_same, m); best_flip = np.fmax(best_flip, mf)
        emb_store[part] = {"stems": items, "Z": Z}
        print(f"{part}: {len(items)} charts", flush=True)
    for name, m in (("same", best_same), ("flip", best_flip)):
        off = ~np.eye(n, dtype=bool) & np.isfinite(m)
        z = (m - m[off].mean()) / (m[off].std() + 1e-9) if off.sum() > 10 else m
        out[f"id__{name}"] = z; out[f"id__n_{name}"] = np.isfinite(z).astype(np.int32)
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "scores.npz", stems=np.array(stems), **out)
    np.savez_compressed(args.out / "embeddings.npz", **{f"{p_}_stems": np.array(v["stems"]) for p_, v in emb_store.items()}, **{f"{p_}_Z": v["Z"] for p_, v in emb_store.items()})
    sources = ["id"] + [f"id_{p_}" for p_ in PARTS if f"id_{p_}__same" in out]
    json.dump({"sources": sources, "calib": {}, "profile_min": args.profile_min, "checkpoint": str(args.checkpoint)}, open(args.out / "match_meta.json", "w"))
    if args.labels:
        from sklearn.metrics import roc_auc_score

        L = json.loads(args.labels.read_text())["days"].get(day, {}).get("photos", {})
        gt = np.array([L.get(s.split("__")[-1], "?") for s in stems]); grp = np.array([g.split("_")[-1][:3] for g in gt])
        iu = np.triu_indices(n, 1); same = (gt[:, None] == gt[None, :])[iu]; sg = (grp[:, None] == grp[None, :])[iu]; known = ((gt != "?")[:, None] & (gt != "?")[None, :])[iu]
        print(f"{'source':12s} {'pairs':>7s} {'AUC all':>8s} {'same group':>11s} {'same group 10s-1h':>18s}")
        for src in sources:
            a = out[f"{src}__same"][iu]; f = out[f"{src}__flip"][iu]; a = np.where(np.isnan(a), f, np.where(np.isnan(f), a, np.maximum(a, f))); ok = np.isfinite(a) & known
            def auc(m):
                mm = ok & m; return roc_auc_score(same[mm], a[mm]) if mm.sum() > 30 and same[mm].any() and (~same[mm]).any() else float("nan")
            print(f"{src:12s} {ok.sum():7d} {auc(np.ones_like(ok)):8.3f} {auc(sg):11.3f} {auc(sg & (dt[iu] > 10) & (dt[iu] < 3600)):18.3f}")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
