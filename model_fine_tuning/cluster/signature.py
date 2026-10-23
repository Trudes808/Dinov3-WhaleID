"""Signatures: the most distinctive regions of each whale, judged against all the other whales of the day, and
tracklet-to-tracklet linking through those regions only ("less is more").

    python -m model_fine_tuning.cluster.signature --day-dir DAY [--features DAY/dino_hr/features.npz] [--labels labels.json] --out DAY/signature

Input: the hi-res DINO tokens of every instance chart (cluster/dino_hr.py extract: PCA-64 unit descriptors on the 64 x 64
token grid of the 1024-px pose-normalised chart, whale tokens only) and the tracklets of the day (cluster/tracklets.py,
appearance = the hi-res global body embedding). For every whale token t of a chart (part p, tracklet A):
  commonness(t)    how ordinary this look is at this body location across the day's whales: for every OTHER tracklet,
                   the best cosine to its tokens of the same part within ALIGN_R of t's chart position; commonness =
                   the q-th best of those per-tracklet maxima (q = 5: a few re-sightings of the same whale in other
                   tracklets cannot make a mark look common)
  repeatability(t) the best cosine to the tokens of the OTHER charts of the same tracklet (same part, same position
                   window): a real mark recurs frame to frame, water glare does not (1.0 when the tracklet has no other
                   chart of that part: single shots rely on rarity alone)
  distinctiveness  d(t) = repeatability(t) - commonness(t)
A tracklet's signature per part = its top-M tokens by d(t) with a minimum spacing on the chart (the dorsal notch, the
barnacle patch, the scar), each with its descriptor, chart and position. Linking: for tracklets A and B, every
signature token of A looks for its best match in B's charts of the same part near the same position; the pair score
is the mean of the best K such matches, symmetrised. Scored against the catalog when labels are given (tracklet-pair
AUC next to the hi-res global embedding), but the point of this module is the qualitative view: distinctiveness maps
and matched points for the review web app (summary.json + heat/ PNGs).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .bursts import make_bursts
from .local_hr import ALIGN_R
from .tracklets import load_instances, make_tracklets

PARTS = ("body", "fin", "fluke")


def load_features(path: Path, center: bool = False, glob_path: Path | None = None, pca_dim: int = 64) -> dict:
    """center: subtract every chart's mean whale token before re-normalising, so a token describes how a spot differs
    from the rest of that whale (a mark) rather than the whale's overall look (which the fine-tuned tokens carry
    everywhere and which makes the whole animal look 'unique').
    Two layouts: dino_hr extract (dense (n, 4096, 64) tokens on a 64-grid) and local_feats features_v2 (ragged whale
    tokens at native resolution with chart-pixel positions; re-expressed on a nominal 64-grid of positions = chart
    fractions x 64 so the radius logic and the web app stay unchanged; raw backbone tokens get a per-day PCA)."""
    z = np.load(path)
    if "tok_offsets" in z.files:
        return load_features_v2(z, center, glob_path, pca_dim)
    local = z["local"]
    if center:
        local = local.astype(np.float32).copy(); mask = z["mask"]
        for i in range(len(local)):
            m = mask[i]
            if m.any():
                local[i] -= local[i][m].mean(0, keepdims=True)
        local /= np.maximum(np.linalg.norm(local, axis=-1, keepdims=True), 1e-6); local = local.astype(np.float16)
    return {"stems": [str(s) for s in z["stems"]], "parts": [str(p) for p in z["parts"]], "glob": z["glob"], "local": local, "mask": z["mask"], "grid": int(z["grid"])}


def load_features_v2(z, center: bool, glob_path: Path | None, pca_dim: int) -> dict:
    stems = [str(s) for s in z["stems"]]; parts = [str(p) for p in z["parts"]]; offs = z["tok_offsets"]; sizes = z["chart_size"]
    desc = z["tok_desc"].astype(np.float32); xy = z["tok_xy"]
    if str(z["kind"]) != "local" and desc.shape[1] > pca_dim:                              # raw backbone tokens -> PCA
        rng = np.random.default_rng(0); sample = desc[rng.choice(len(desc), min(len(desc), 60000), replace=False)]
        mean = sample.mean(0); _, _, Vt = np.linalg.svd(sample - mean, full_matrices=False); desc = (desc - mean) @ Vt[:pca_dim].T
    per_chart_local, per_chart_pos, per_chart_mask = [], [], []
    for i in range(len(stems)):
        d = desc[offs[i]:offs[i + 1]]
        if center and len(d):
            d = d - d.mean(0, keepdims=True)
        d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)
        per_chart_local.append(d.astype(np.float16)); per_chart_pos.append((xy[offs[i]:offs[i + 1]] / float(sizes[i]) * 64.0)[:, ::-1].astype(np.float32))    # (row, col) nominal grid
    F = {"stems": stems, "parts": parts, "grid": 64, "ragged": True, "local": per_chart_local, "rpos": per_chart_pos,
         "mask": [np.ones(len(d), bool) for d in per_chart_local], "token_px": int(z["token_px"]), "chart_size": sizes}
    if glob_path is not None and Path(glob_path).exists():
        g = np.load(glob_path); row = {(str(s_), str(p_)): i for i, (s_, p_) in enumerate(zip(g["stems"], g["parts"]))}
        F["glob"] = np.stack([g["glob"][row[(s_, p_)]] if (s_, p_) in row else np.zeros(g["glob"].shape[1], np.float32) for s_, p_ in zip(stems, parts)])
    else:
        F["glob"] = np.stack([d.astype(np.float32).mean(0) if len(d) else np.zeros(desc.shape[1], np.float32) for d in per_chart_local])
        F["glob"] /= np.maximum(np.linalg.norm(F["glob"], axis=1, keepdims=True), 1e-6)
    return F


def grid_positions(g: int) -> np.ndarray:
    return np.stack(np.divmod(np.arange(g * g), g), 1).astype(np.float32)                  # (row, col) per token


def token_canon_coords(day_dir: Path, parts_dir: Path, key: str, part: str, grid: int, chart_size: int = 1536) -> np.ndarray:
    """Canonical body coordinates (metres: body axis, lateral, height) of every token centre of a chart, NaN off the whale:
    chart px -> full-resolution px (inverse of the chart warp M) -> 2560-px image -> segmentation crop -> the CSE map."""
    z = np.load(parts_dir / f"{key}__{part}.npz")
    M = z["M"]; x0, y0, sc = z["x0y0s"]; scale_full = float(z["scale_full"])
    stem = key.rsplit("~", 1)[0] if "~" in key else key
    c = np.load(day_dir / "cse" / f"{stem}_canon.npz"); signed = c["signed_mm"].astype(np.float32) / 1000.0; use = c["use"]; C = use.shape[0]
    pos = grid_positions(grid); cx = (pos[:, 1] + 0.5) * chart_size / grid; cy = (pos[:, 0] + 0.5) * chart_size / grid
    if bool(z["mirrored"]):                                                                      # charts are mirrored / flipped AFTER the warp
        cx = chart_size - cx
    if part == "fluke" and bool(z.get("flipped_v", False)):
        cy = chart_size - cy
    A = M[:, :2]; t = M[:, 2]
    full = np.linalg.solve(A, np.stack([cx - t[0], cy - t[1]], 0)).T                           # (N, 2) full-res px
    crop = (full / scale_full - np.array([x0, y0])) * sc
    out = np.full((grid * grid, 3), np.nan, np.float32)
    ix, iy = np.round(crop[:, 0]).astype(int), np.round(crop[:, 1]).astype(int)
    ok = (ix >= 0) & (ix < C) & (iy >= 0) & (iy < C)
    ok[ok] &= use[iy[ok], ix[ok]]
    out[ok] = signed[iy[ok], ix[ok]]
    return out


def chart_tokens(F: dict, i: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit descriptors of the whale tokens and their matching positions: the canonical coordinates (metres) when the
    features carry them (F["cpos"], tokens without one dropped), else the chart grid position."""
    if F.get("ragged"):
        return torch.from_numpy(F["local"][i].astype(np.float32)).to(device), torch.from_numpy(np.ascontiguousarray(F["rpos"][i])).to(device)
    m = F["mask"][i].copy()
    if "cpos" in F:
        m &= np.isfinite(F["cpos"][i][:, 0])
        P = F["cpos"][i][m]
    else:
        P = grid_positions(F["grid"])[m]
    return torch.from_numpy(F["local"][i][m].astype(np.float32)).to(device), torch.from_numpy(np.ascontiguousarray(P)).to(device)


def token_grid_pos(F: dict, i: int) -> tuple[np.ndarray, np.ndarray]:
    """(token indices into F["local"][i], nominal-grid (row, col) positions) in chart_tokens order."""
    if F.get("ragged"):
        return np.arange(len(F["local"][i])), F["rpos"][i]
    m = F["mask"][i].copy()
    if "cpos" in F:
        m &= np.isfinite(F["cpos"][i][:, 0])
    idx = np.nonzero(m)[0]
    return idx, grid_positions(F["grid"])[idx]


def match_radius(F: dict, part: str) -> float:
    return F["canon_radius"] if "cpos" in F else ALIGN_R[part] * F.get("align_scale", 1.0) * F["grid"]


@torch.no_grad()
def dense_scores(F: dict, tracks: list[dict], part_rows: dict, device, max_anchors: int = 2000, topk_frac: float = 1.0) -> np.ndarray:
    """(T, T) dense link score, the training metric of cluster/train_local.py: for a chart pair of one part, the mean
    over A's whale tokens of the best cosine within the match radius in B; tracklet pair = best chart pair (symmetrised)."""
    T = len(tracks); S = np.full((T, T), np.nan, np.float32)
    bank: dict = {}
    g = torch.Generator(device="cpu").manual_seed(0)
    for t in range(T):
        for part in PARTS:
            rows = part_rows.get((t, part), [])
            if rows:
                lst = []
                for i in rows:
                    L, P = chart_tokens(F, i, device)
                    if len(L) > max_anchors:
                        sel = torch.randperm(len(L), generator=g)[:max_anchors].to(device); L, P = L[sel], P[sel]
                    if len(L):
                        lst.append((L, P))
                if lst:
                    bank[(t, part)] = lst
    for a in range(T):
        for b in range(a + 1, T):
            best = -2.0
            for part in PARTS:
                if (a, part) not in bank or (b, part) not in bank:
                    continue
                r = match_radius(F, part)
                for La, Pa in bank[(a, part)]:
                    for Lb, Pb in bank[(b, part)]:
                        sim = La @ Lb.T; near = torch.cdist(Pa, Pb) <= r
                        va = torch.where(near, sim, torch.full_like(sim, -1.0)).max(1).values; va = va[va > -1.0]
                        vb = torch.where(near, sim, torch.full_like(sim, -1.0)).max(0).values; vb = vb[vb > -1.0]
                        if len(va) and len(vb):
                            if topk_frac < 1.0:
                                va = torch.topk(va, min(len(va), max(10, int(topk_frac * len(va))))).values; vb = torch.topk(vb, min(len(vb), max(10, int(topk_frac * len(vb))))).values
                            best = max(best, 0.5 * (float(va.mean()) + float(vb.mean())))
            if best > -2.0:
                S[a, b] = S[b, a] = best
    return S


@torch.no_grad()
def distinctiveness(F: dict, tracklet_of: dict, device, q: int = 5, chunk_tokens: int = 24000) -> dict:
    """Per chart index: (commonness, repeatability, d) per whale token, in the order of chart_tokens()."""
    out = {}
    by_part = {p: [i for i, (s, pt) in enumerate(zip(F["stems"], F["parts"])) if pt == p and s in tracklet_of] for p in PARTS}
    for part, idx in by_part.items():
        if len(idx) < 3:
            continue
        toks = [chart_tokens(F, i, device) for i in idx]
        tr = np.array([tracklet_of[F["stems"][i]] for i in idx]); n_tr = int(tr.max()) + 1
        # the part's tokens grouped by tracklet: descriptors, positions, chart row (one matmul per chart x tracklet)
        by_tr: dict[int, tuple] = {}
        for t_id in np.unique(tr):
            rows = [r for r in range(len(toks)) if tr[r] == t_id]
            by_tr[int(t_id)] = (torch.cat([toks[r][0] for r in rows]), torch.cat([toks[r][1] for r in rows]), torch.cat([torch.full((len(toks[r][0]),), r, device=device) for r in rows]))
        radius = match_radius(F, part)
        for r, (L, P) in enumerate(toks):
            n = len(L)
            if n == 0:
                out[idx[r]] = (np.zeros(0), np.zeros(0), np.zeros(0)); continue
            best_tr = torch.full((n, n_tr), -1.0, device=device)                 # best cosine per other tracklet
            rep = torch.full((n,), -1.0, device=device)
            for t_id, (Lb, Pb, rb) in by_tr.items():
                for s0 in range(0, len(Lb), chunk_tokens):
                    S = L @ Lb[s0:s0 + chunk_tokens].T                             # (n, chunk)
                    near = torch.cdist(P, Pb[s0:s0 + chunk_tokens]) <= radius
                    S = torch.where(near & (rb[s0:s0 + chunk_tokens] != r)[None, :], S, torch.full_like(S, -1.0))
                    m = S.max(1).values
                    if t_id == tr[r]:
                        rep = torch.maximum(rep, m)
                    else:
                        best_tr[:, t_id] = torch.maximum(best_tr[:, t_id], m)
                    del S, near
            kth = torch.topk(best_tr, min(q, n_tr), dim=1).values[:, -1]           # q-th best other tracklet
            com = kth.clamp(min=0.0)
            out[idx[r]] = (com.cpu().numpy(), rep.cpu().numpy(), None)
    # single shots (no other chart of that part in the tracklet) get the day's typical repeatability, not 1.0, so that
    # they sit on the same scale as the multi-shot tracklets
    have = np.concatenate([v[1][v[1] >= 0] for v in out.values() if len(v[1])]) if out else np.zeros(0)
    typical = float(np.median(have)) if len(have) else 0.7
    for i, (com, rep, _) in out.items():
        rep_v = np.where(rep < 0, typical, rep).astype(np.float32)
        out[i] = (com, rep_v, rep_v - com)
    return out


def pick_signature(F: dict, dist: dict, chart_rows: list[int], M: int = 24, min_sep: float = 2.0) -> list[dict]:
    """Top-M tokens by distinctiveness over a tracklet's charts of one part, greedy with a minimum chart spacing."""
    cand = []
    g = F["grid"]; pos_all = grid_positions(g)
    for i in chart_rows:
        if i not in dist:
            continue
        com, rep, d = dist[i]
        tok_idx, P = token_grid_pos(F, i)                                        # token index per entry of d (chart_tokens order)
        CP = F["cpos"][i][tok_idx] if "cpos" in F else P
        for k in np.argsort(-d)[: 4 * M]:
            cand.append({"row": i, "pos": P[k], "cpos": CP[k], "d": float(d[k]), "com": float(com[k]), "rep": float(rep[k]), "k": int(tok_idx[k])})
    cand.sort(key=lambda c: -c["d"])
    chosen = []
    for c in cand:
        if all(not (o["row"] == c["row"] and np.linalg.norm(o["pos"] - c["pos"]) < min_sep) for o in chosen):
            chosen.append(c)
        if len(chosen) >= M:
            break
    return chosen


@torch.no_grad()
def link_scores(F: dict, sigs: dict, tracks: list[dict], part_rows: dict, device, K: int = 8) -> np.ndarray:
    """(T, T) signature link score: mean of the best K aligned matches of A's signature tokens in B's charts (symmetrised)."""
    T = len(tracks); S = np.full((T, T), np.nan, np.float32)
    g = F["grid"]; pos_all = np.stack(np.divmod(np.arange(g * g), g), 1).astype(np.float32)
    bank = {}                                                                     # (tracklet, part) -> (L, P)
    for t in range(T):
        for part in PARTS:
            rows = part_rows.get((t, part), [])
            if rows:
                Ls, Ps = zip(*[chart_tokens(F, i, device) for i in rows]); bank[(t, part)] = (torch.cat(Ls), torch.cat(Ps))
    sig_t = {}
    for (t, part), pts in sigs.items():
        if pts:
            L = torch.from_numpy(np.stack([F["local"][p["row"]][p["k"]].astype(np.float32) for p in pts])).to(device)
            P = torch.from_numpy(np.stack([p["cpos"] for p in pts]).astype(np.float32)).to(device); w = torch.tensor([max(p["d"], 1e-3) for p in pts], device=device)
            sig_t[(t, part)] = (L, P, w)
    for a in range(T):
        for b in range(T):
            if a == b:
                continue
            vals = []
            for part in PARTS:
                if (a, part) not in sig_t or (b, part) not in bank:
                    continue
                L, P, w = sig_t[(a, part)]; Lb, Pb = bank[(b, part)]
                sim = L @ Lb.T; near = torch.cdist(P, Pb) <= match_radius(F, part)
                best = torch.where(near, sim, torch.full_like(sim, -1.0)).max(1).values
                vals.append(best)
            if vals:
                v = torch.cat(vals); S[a, b] = float(torch.topk(v, min(K, len(v))).values.mean())
    return 0.5 * (S + S.T)


def heat_png(F: dict, i: int, dist: dict, chart_jpg: Path, out: Path, size: int = 512, scale: tuple[float, float] = (0.0, 0.6)) -> None:
    """Chart with the distinctiveness map blended on the whale (red = unique, blue = common); one colour scale for the
    whole day (scale = day-wide 5th / 99th percentile of d), so a chart without distinctive regions stays blue."""
    img = cv2.imread(str(chart_jpg)); img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    g = F["grid"]; heat = np.full(g * g, np.nan, np.float32)
    if i in dist:
        if F.get("ragged"):                                                      # rasterise the native tokens on the nominal grid (mean per cell)
            idx, P = token_grid_pos(F, i); cell = np.clip(P[:, 0].astype(int), 0, g - 1) * g + np.clip(P[:, 1].astype(int), 0, g - 1)
            acc = np.zeros(g * g, np.float64); cnt = np.zeros(g * g); np.add.at(acc, cell, dist[i][2]); np.add.at(cnt, cell, 1)
            heat[cnt > 0] = (acc[cnt > 0] / cnt[cnt > 0]).astype(np.float32)
        else:
            m = F["mask"][i].copy()
            if "cpos" in F:
                m &= np.isfinite(F["cpos"][i][:, 0])
            heat[m] = dist[i][2]
    h = heat.reshape(g, g); ok = np.isfinite(h)
    if ok.any():
        lo, hi = scale
        hn = np.clip((np.nan_to_num(h, nan=lo) - lo) / max(hi - lo, 1e-6), 0, 1)
        col = cv2.applyColorMap((hn * 255).astype(np.uint8), cv2.COLORMAP_JET)
        col = cv2.resize(col, (size, size), interpolation=cv2.INTER_LINEAR); m = cv2.resize(ok.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST) > 0
        img[m] = (0.45 * img[m] + 0.55 * col[m]).astype(np.uint8)
    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 85])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--day-dir", type=Path, required=True); p.add_argument("--features", type=Path, default=None); p.add_argument("--parts-tag", default="parts_hr")
    p.add_argument("--glob-features", type=Path, default=None, help="dino_hr features.npz whose global embeddings rank the links (default DAY/dino_hr/features.npz)")
    p.add_argument("--labels", type=Path, default=None); p.add_argument("--out", type=Path, default=None); p.add_argument("--gap", type=float, default=1.5)
    p.add_argument("--max-link-cost", type=float, default=1e9, help="tracklet link cost; default = plain bursts (the same-whale prior for repeatability)"); p.add_argument("--M", type=int, default=24); p.add_argument("--q", type=int, default=5); p.add_argument("--K", type=int, default=8)
    p.add_argument("--top", type=int, default=5); p.add_argument("--heat", action="store_true", help="write the heat PNGs for the web app"); p.add_argument("--device", default="cuda")
    p.add_argument("--center", action="store_true", help="per-chart mean-centred tokens (marks rather than the whale's overall look)")
    p.add_argument("--shape", action="store_true", help="fin outline + fluke trailing-edge signatures (cluster/shape_sig.py)")
    p.add_argument("--canon", action="store_true", help="match tokens by canonical body coordinates from the CSE map (same body spot across poses) instead of chart position")
    p.add_argument("--canon-radius", type=float, default=0.5, help="metres on the body within which tokens are compared (--canon)")
    p.add_argument("--sig-weight", type=float, default=0.25, help="weight of the signature z-score next to the global z-score in the link ranking")
    p.add_argument("--align-scale", type=float, default=1.0, help="multiplies ALIGN_R (body 0.5, fin / fluke 0.3 of the chart): 0.1 = the 3 %% radius the local descriptor was trained with")
    p.add_argument("--dense", action="store_true", help="also compute the dense link score (all whale tokens, the train_local validation metric)")
    p.add_argument("--dense-topk", type=float, default=1.0, help="dense score = mean of this top fraction of anchor matches (1.0 = all)")
    p.add_argument("--rank", default="combined", choices=["combined", "dense"], help="which score ranks the links in the summary")
    a = p.parse_args(argv)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    day = a.day_dir.name; out = a.out or (a.day_dir / "signature"); out.mkdir(parents=True, exist_ok=True)
    F = load_features(a.features or (a.day_dir / "dino_hr" / "features.npz"), center=a.center, glob_path=a.glob_features or (a.day_dir / "dino_hr" / "features.npz"))
    if F.get("ragged") and a.canon:
        raise SystemExit("--canon is not implemented for native-resolution (ragged) features")
    if F.get("ragged"):
        print(f"  native-resolution tokens: {sum(len(d) for d in F['local'])} whale tokens over {len(F['stems'])} charts", flush=True)
    ts = json.loads((a.day_dir / "timestamps.json").read_text()); inst = load_instances(a.day_dir, a.parts_tag)
    F["align_scale"] = a.align_scale
    if a.canon:
        F["cpos"] = np.stack([token_canon_coords(a.day_dir, a.day_dir / a.parts_tag, s_, pt, F["grid"]) for s_, pt in zip(F["stems"], F["parts"])])
        F["canon_radius"] = a.canon_radius
        print(f"  canonical token positions: {np.isfinite(F['cpos'][..., 0])[F['mask']].mean():.2f} of the whale tokens mapped", flush=True)
    row = {(s, pt): i for i, (s, pt) in enumerate(zip(F["stems"], F["parts"]))}
    app = {s: F["glob"][i] for (s, pt), i in row.items() if pt == "body"}
    tracks = make_tracklets(ts, inst, a.gap, app, a.max_link_cost)
    tracks = [t for t in tracks if any((k, pt) in row for k in t["keys"] for pt in PARTS)]
    tracklet_of = {k: t["id"] for t in tracks for k in t["keys"]}
    for n, t in enumerate(tracks):                                               # dense ids
        for k in t["keys"]:
            tracklet_of[k] = n
        t["id"] = n
    print(f"{day}: {len(F['stems'])} charts, {len(tracks)} tracklets", flush=True)
    dist = distinctiveness(F, tracklet_of, device, q=a.q)
    part_rows = {}
    for t in tracks:
        for pt in PARTS:
            rows = [row[(k, pt)] for k in t["keys"] if (k, pt) in row]
            if rows:
                part_rows[(t["id"], pt)] = rows
    sigs = {key: pick_signature(F, dist, rows, a.M) for key, rows in part_rows.items()}
    S = link_scores(F, sigs, tracks, part_rows, device, a.K)
    Dn = dense_scores(F, tracks, part_rows, device, topk_frac=a.dense_topk) if a.dense else None
    shape = shape_sc = None
    if a.shape:
        from . import shape_sig

        tracklet_of_row = {i: tracklet_of[s_] for i, s_ in enumerate(F["stems"]) if s_ in tracklet_of}
        shape = shape_sig.compute(a.day_dir / a.parts_tag, [(i, F["stems"][i], F["parts"][i]) for i in range(len(F["stems"])) if F["parts"][i] in ("fin", "fluke")], tracklet_of_row)
        shape_sc = shape_sig.tracklet_scores(shape, part_rows, len(tracks))
        print(f"  shapes: {len(shape['rows'])} outlines ({', '.join(f'{k}: {len(v[chr(114)+chr(111)+chr(119)+chr(115)])}' for k, v in shape['banks'].items())})", flush=True)
    # global hi-res reference: best over parts of the max chart-pair cosine between the tracklets
    G = np.full_like(S, np.nan)
    for (ta, pa), ra in part_rows.items():
        for (tb, pb), rb in part_rows.items():
            if ta < tb and pa == pb:
                v = float((F["glob"][ra] @ F["glob"][rb].T).max()); G[ta, tb] = G[tb, ta] = np.fmax(np.nan_to_num(G[ta, tb], nan=-9), v)
    # combined ranking: z-scored hi-res global + z-scored signature (the global finds the candidates, the signature points explain them)
    def zs(M_):
        v = M_[np.isfinite(M_)]; return (M_ - v.mean()) / (v.std() + 1e-9) if len(v) > 1 else M_
    C = np.where(np.isfinite(S) & np.isfinite(G), a.sig_weight * zs(S) + zs(G), np.where(np.isfinite(G), zs(G), zs(S)))
    if a.dense and a.rank == "dense":
        C = np.where(np.isfinite(Dn), zs(Dn), C)
    L = json.loads(a.labels.read_text())["days"].get(day, {}).get("photos", {}) if a.labels else {}
    lab = []
    for t in tracks:
        ids = [L.get(k.rsplit("~", 1)[0].split("__")[-1], "?") for k in t["keys"] if k.endswith("~0")]
        ids = [i for i in ids if i != "?" and not i.split("_")[1].startswith("O")]
        lab.append(max(set(ids), key=ids.count) if ids else "?")
    metrics = {}
    if any(l != "?" for l in lab):
        from sklearn.metrics import roc_auc_score
        lab_a = np.array(lab); iu = np.triu_indices(len(tracks), 1); known = (lab_a[iu[0]] != "?") & (lab_a[iu[1]] != "?")
        y = (lab_a[iu[0]] == lab_a[iu[1]])[known]
        cands = [("signature", S), ("hr_global", G), ("combined", C)] + ([("dense", Dn)] if Dn is not None else []) + ([(f"shape_{k}", v) for k, v in shape_sc.items()] if shape_sc else [])
        for name, M_ in cands:
            s = M_[iu][known]; ok = np.isfinite(s)
            metrics[f"tracklet_pair_auc_{name}"] = float(roc_auc_score(y[ok], s[ok])) if y[ok].any() and (~y[ok]).any() else float("nan")
            metrics[f"pairs_{name}"] = int(ok.sum())
        print("  tracklet-pair AUC: " + "  ".join(f"{n} {metrics[f'tracklet_pair_auc_{n}']:.3f} ({metrics[f'pairs_{n}']})" for n, _ in cands), flush=True)
    # summary for the web app
    heat_dir = out / "heat"; heat_dir.mkdir(exist_ok=True)
    charts_dir = a.day_dir / a.parts_tag / "charts"
    all_d = np.concatenate([v[2] for v in dist.values() if len(v[2])]) if dist else np.zeros(1)
    scale = (float(np.percentile(all_d, 5)), float(np.percentile(all_d, 99)))
    summary = {"day": day, "grid": F["grid"], "metrics": metrics, "features": str(a.features or "dino_hr"), "heat_scale": scale, "canon": a.canon, "align_scale": a.align_scale, "rank": a.rank, "tracklets": []}
    for t in tracks:
        tid = t["id"]; charts = []
        for pt in PARTS:
            for i in part_rows.get((tid, pt), []):
                key = F["stems"][i]
                if a.heat:
                    heat_png(F, i, dist, charts_dir / f"{key}__{pt}.jpg", heat_dir / f"{key}__{pt}.jpg", scale=scale)
                com, rep, d = dist.get(i, (np.zeros(0),) * 3)
                rec = {"key": key, "part": pt, "row": int(i), "d_p90": float(np.percentile(d, 90)) if len(d) else None, "d_max": float(d.max()) if len(d) else None}
                if shape and i in shape["rows"]:
                    sh = shape["rows"][i]; rec["outline"] = [[round(float(x), 1), round(float(y), 1), round(float(u), 2)] for (x, y), u in zip(sh["pts"], sh["u"])]
                    rec["outline_u_max"] = round(float(sh["u"].max()), 2)
                charts.append(rec)
        sig_pts = {pt: [{"key": F["stems"][s["row"]], "x": float(s["pos"][1]), "y": float(s["pos"][0]), "d": round(s["d"], 3), "com": round(s["com"], 3), "rep": round(s["rep"], 3)}
                       for s in sigs.get((tid, pt), [])] for pt in PARTS}
        def correspondences(b: int) -> list[dict]:
            """A's signature tokens matched in tracklet b's charts (for drawing)."""
            corr = []
            for pt in PARTS:
                if (tid, pt) not in sigs or (int(b), pt) not in part_rows:
                    continue
                Lb_rows = part_rows[(int(b), pt)]
                for s in sigs[(tid, pt)][:12]:
                    best = (-1.0, None)
                    for j in Lb_rows:
                        Lj, Pj = chart_tokens(F, j, device)
                        if len(Lj) == 0:
                            continue
                        sim = torch.from_numpy(F["local"][s["row"]][s["k"]].astype(np.float32)).to(device) @ Lj.T
                        near = torch.cdist(torch.from_numpy(np.asarray(s["cpos"], np.float32)[None]).to(device), Pj)[0] <= match_radius(F, pt)
                        sim = torch.where(near, sim, torch.full_like(sim, -1.0)); kbest = int(sim.argmax()); v = float(sim[kbest])
                        if v > best[0]:
                            gp = token_grid_pos(F, j)[1][kbest]
                            best = (v, {"key_b": F["stems"][j], "xb": float(gp[1]), "yb": float(gp[0])})
                    if best[1]:
                        corr.append({"part": pt, "key_a": F["stems"][s["row"]], "xa": float(s["pos"][1]), "ya": float(s["pos"][0]), "sim": round(best[0], 3), **best[1]})
            return corr

        def top_matches(score_row: np.ndarray) -> list[dict]:
            out_m = []
            for b in np.argsort(-np.nan_to_num(score_row, nan=-9))[: a.top + 1]:
                if b == tid or not np.isfinite(score_row[b]):
                    continue
                out_m.append({"tracklet": int(b), "combined": round(float(C[tid, b]), 2) if np.isfinite(C[tid, b]) else None, "score": round(float(S[tid, b]), 3) if np.isfinite(S[tid, b]) else None,
                              "hr_global": round(float(G[tid, b]), 3) if np.isfinite(G[tid, b]) else None, "label": lab[b], "corr": correspondences(int(b))})
                if len(out_m) >= a.top:
                    break
            return out_m
        matches = top_matches(C[tid]); matches_sig = top_matches(S[tid])
        shape_matches = []
        if shape_sc and "shape" in shape_sc:
            Ssh = shape_sc["shape"][tid]
            for b in np.argsort(-np.nan_to_num(Ssh, nan=-9))[: a.top + 1]:
                if b == tid or not np.isfinite(Ssh[b]):
                    continue
                shape_matches.append({"tracklet": int(b), "score": round(float(Ssh[b]), 3), "label": lab[b],
                                      **{f"{k}": (round(float(v[tid, b]), 3) if np.isfinite(v[tid, b]) else None) for k, v in shape_sc.items() if k != "shape"}})
                if len(shape_matches) >= a.top:
                    break
        summary["tracklets"].append({"id": tid, "label": lab[tid], "burst": t["burst"], "camera": t["camera"], "start": t["start"], "end": t["end"], "keys": t["keys"], "photos": t["photos"],
                                     "charts": charts, "signature": sig_pts, "matches": matches, "matches_sig": matches_sig, "shape_matches": shape_matches})
    (out / "summary.json").write_text(json.dumps(summary))
    np.savez_compressed(out / "scores.npz", link=S, hr_global=G, labels=np.array(lab))
    print(f"  -> {out}/summary.json ({len(tracks)} tracklets, heat PNGs: {a.heat})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
