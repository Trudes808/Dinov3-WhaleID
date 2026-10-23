"""Day clustering v3: whale instances, tracklets, several part-aligned scorers, learned fusion, cannot-links.

    python -m model_fine_tuning.cluster.cluster_day channels --day-dir DAY --out DAY/cluster_v3 \
        --sources id:experiments/id_v3/checkpoints/best.pt,md,hr:dino_hr --local lightglue,hrlocal --candidates 30
    python -m model_fine_tuning.cluster.cluster_day cluster --day-dir DAY --out DAY/cluster_v3 --labels labels.json [--fusion experiments/fusion_v3.json]

channels: every instance chart (cluster/parts.py --instances, parts_hr) is scored against the others of the same part
type by each source: global embeddings ("id:<ckpt>" = cluster/train_id.py model, "md" = MegaDescriptor-L-384,
"hr:<tag>" = cluster/dino_hr.py features in DAY/<tag>/features.npz) give a cosine matrix per part, z-scored over the
day's pairs of that part; the local scorers (cluster/local_hr.py: "lightglue", "hrlocal") run on the candidate pairs
(top-K neighbours of every chart by the mean global z-score) and are z-scored over those. All matrices (instances x
instances, NaN where a channel does not apply) are saved to channels.npz; adding a source later only computes the
missing channels.

cluster: the instance-pair score is the fusion (cluster/fuse.py model fitted on the val days: per-channel calibration
then a logistic combination with missing-channel indicators) or, without one, the best over parts of the mean global
z-score. Nodes are tracklets (cluster/tracklets.py; appearance = the first id source's body embedding), the node pair
score the mean of the top-3 instance-pair scores, plus time_weight x the step time term (0 for two tracklets of the same
burst: they were split on purpose). Instances of one photo that show the same body region cannot be one whale: the
constraint is propagated through the average linkage (cluster/linkage.py). The threshold is self-calibrated (same-
tracklet consecutive pairs vs pairs > 90 min apart) or fixed with --threshold. Photos are scored on their main instance
(~0) against the catalog; groups by interleaving as before. Labels are read for scoring only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .cluster_bursts import embed_charts, evaluate, groups_from_animals
from .linkage import constrained_average_linkage
from .match import time_term
from .tracklets import cannot_link_instances, instance_coverage, load_instances, make_tracklets

PARTS = ("body", "fin", "fluke")


# ------------------------------------------------------------------ day layout
def day_layout(day_dir: Path, parts_tag: str) -> dict:
    ts = json.loads((day_dir / "timestamps.json").read_text())
    inst = load_instances(day_dir, parts_tag)
    keys = sorted(i["key"] for s, v in inst.items() for i in v if ts.get(s, {}).get("time") is not None)
    pos = {k: i for i, k in enumerate(keys)}
    photo = {k: k.rsplit("~", 1)[0] if "~" in k else k for k in keys}
    times = np.array([ts[photo[k]]["time"] for k in keys], float)
    cam = np.array([f"{ts[photo[k]].get('folder', '?')}|{ts[photo[k]].get('camera', '?')}" for k in keys])
    parts_dir = day_dir / parts_tag
    charts = {p: [(k, parts_dir / "charts" / f"{k}__{p}.jpg", parts_dir / f"{k}__{p}.npz") for k in keys if (parts_dir / f"{k}__{p}.npz").exists()] for p in PARTS}
    return {"ts": ts, "instances": inst, "keys": keys, "pos": pos, "photo": photo, "times": times, "cam": cam, "charts": charts, "parts_dir": parts_dir}


def channel_source(src: str) -> str:
    """Channel name of a source: md, hr:<tag>, id:<experiment name of the checkpoint>, idxr:<experiment> (the same
    identity model applied to the native-resolution parts_xr charts, central 1536-px window)."""
    for pre in ("id:", "idxr:", "idsil:"):
        if src.startswith(pre):
            parts = Path(src[len(pre):]).parts
            return pre + (parts[-3] if len(parts) >= 3 and parts[-2] == "checkpoints" else Path(src[len(pre):]).stem)
    return src


def zscore(S: np.ndarray) -> np.ndarray:
    v = S[np.isfinite(S)]
    return (S - v.mean()) / (v.std() + 1e-9) if len(v) > 1 else S


# ------------------------------------------------------------------ channels
def global_embeddings(lay: dict, source: str, part: str, device, day_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    files = lay["charts"][part]
    if len(files) < 3:
        return None
    idx = np.array([lay["pos"][k] for k, _, _ in files])
    if source.startswith("hr:"):
        z = np.load(day_dir / source[3:] / "features.npz")
        row = {(str(s), str(p)): i for i, (s, p) in enumerate(zip(z["stems"], z["parts"]))}
        sel = [(i, row[(k, part)]) for i, (k, _, _) in zip(idx, files) if (k, part) in row]
        if len(sel) < 3:
            return None
        return np.array([i for i, _ in sel]), z["glob"][[r for _, r in sel]]
    if source.startswith("idsil:"):                                             # shape-only charts (cluster/sil_charts.py)
        if part == "body":
            return None
        sd = day_dir / "parts_sil" / "charts"
        files_s = [(k, sd / f"{k}__{part}.jpg", nz) for k, _, nz in files if (sd / f"{k}__{part}.jpg").exists()]
        if len(files_s) < 3:
            return None
        idx = np.array([lay["pos"][k] for k, _, _ in files_s])
        return idx, embed_charts([f for _, f, _ in files_s], "id:" + source[6:], device, part=part)
    if source.startswith("idxr:"):                                              # native-resolution charts, central window
        if part == "body":
            return None
        xr = day_dir / "parts_xr" / "charts"
        files_xr = [(k, xr / f"{k}__{part}.jpg", nz) for k, _, nz in files if (xr / f"{k}__{part}.jpg").exists()]
        if len(files_xr) < 3:
            return None
        idx = np.array([lay["pos"][k] for k, _, _ in files_xr])
        E = embed_charts([f for _, f, _ in files_xr], "id:" + source[5:], device, part=part, center_crop=1536)
        return idx, E
    E = embed_charts([f for _, f, _ in files], "megadescriptor" if source == "md" else source, device, part=part)
    return idx, E


def compute_channels(lay: dict, day_dir: Path, sources: list[str], local: list[str], candidates: int, device, existing: dict | None = None, candidate_sources: list[str] | None = None,
                     local_parts: tuple = PARTS) -> tuple[dict, dict]:
    """candidate_sources: channel sources whose mean z-score ranks the local scorers' candidate pairs (default: all global channels);
    local_parts: part types the local scorers run on."""
    n = len(lay["keys"]); ch = dict(existing or {}); emb = {}; a_use = candidate_sources or []
    for src in sources:
        for part in PARTS:
            name = f"{channel_source(src)}|{part}"
            if name in ch:
                continue
            r = global_embeddings(lay, src, part, device, day_dir)
            if r is None:
                continue
            idx, E = r
            emb[(src, part)] = (idx, E)
            S = E @ E.T; np.fill_diagonal(S, np.nan)
            M = np.full((n, n), np.nan, np.float32); M[np.ix_(idx, idx)] = zscore(S)
            ch[name] = M
            print(f"  {name}: {len(idx)} charts", flush=True)
    if local:
        for part in local_parts:
            glob_names = [k for k in ch if k.endswith(f"|{part}") and k.split("|")[0] not in ("lightglue", "hrlocal") and (not a_use or k.split("|")[0] in a_use)]
            need = [l for l in local if f"{l}|{part}" not in ch]
            if not glob_names or not need or len(lay["charts"][part]) < 3:
                continue
            G = np.nanmean(np.stack([ch[k] for k in glob_names]), 0)
            files = lay["charts"][part]; idx = np.array([lay["pos"][k] for k, _, _ in files]); sub = G[np.ix_(idx, idx)]
            pairs = set()
            for a in range(len(idx)):
                row = sub[a]; ok = np.nonzero(np.isfinite(row))[0]
                for b in ok[np.argsort(-row[ok])[:candidates]]:
                    pairs.add((min(a, b), max(a, b)))
            pairs = np.array(sorted(pairs)) if pairs else np.zeros((0, 2), int)
            print(f"  {part}: {len(pairs)} candidate pairs for {need}", flush=True)
            for l in need:
                if l == "lightglue":
                    from .local_hr import AlikedLightGlue, lightglue_pair_values

                    vals = lightglue_pair_values(AlikedLightGlue(device), [(f, nz) for _, f, nz in files], pairs, part)
                elif l == "hrlocal":
                    from .local_hr import dino_local_pair_values

                    hr = [s for s in sources if s.startswith("hr:")]
                    if not hr:
                        continue
                    z = np.load(day_dir / hr[0][3:] / "features.npz")
                    row = {(str(s), str(p)): i for i, (s, p) in enumerate(zip(z["stems"], z["parts"]))}
                    rows = np.array([row.get((k, part), -1) for k, _, _ in files])
                    ok = rows >= 0
                    if ok.sum() < 3:
                        continue
                    keep = ok[pairs[:, 0]] & ok[pairs[:, 1]]
                    vals = np.full(len(pairs), np.nan, np.float32)
                    vals[keep] = dino_local_pair_values(z["local"], z["mask"], int(z["grid"]), np.stack([rows[pairs[keep, 0]], rows[pairs[keep, 1]]], 1), part, device)
                else:
                    raise SystemExit(f"unknown local scorer {l}")
                M = np.full((n, n), np.nan, np.float32)
                if len(pairs):
                    v = vals.astype(np.float32); fin = np.isfinite(v)
                    v[fin] = (v[fin] - v[fin].mean()) / (v[fin].std() + 1e-9)
                    M[idx[pairs[:, 0]], idx[pairs[:, 1]]] = v; M[idx[pairs[:, 1]], idx[pairs[:, 0]]] = v
                ch[f"{l}|{part}"] = M
    return ch, emb


# ------------------------------------------------------------------ fusion
def fuse_scores(ch: dict, fusion: dict | None) -> np.ndarray:
    """(n, n) instance-pair score. Without a fusion model: best over parts of the mean global z-score."""
    names = sorted(ch)
    if fusion is None:
        per_part = []
        for part in PARTS:
            g = [ch[k] for k in names if k.endswith(f"|{part}") and k.split("|")[0] not in ("lightglue", "hrlocal")]
            if g:
                per_part.append(np.nanmean(np.stack(g), 0))
        return np.fmax.reduce(per_part) if per_part else np.full(next(iter(ch.values())).shape, np.nan)
    from .fuse import apply_fusion

    return apply_fusion(ch, fusion)


# ------------------------------------------------------------------ clustering
def node_scores(tracks: list[dict], score: np.ndarray, times: np.ndarray, pos: dict, topk: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = len(tracks); app = np.full((m, m), np.nan); dt = np.full((m, m), np.inf); same_burst = np.zeros((m, m), bool)
    idx = [np.array([pos[k] for k in t["keys"] if k in pos]) for t in tracks]
    for i in range(m):
        if len(idx[i]) == 0:
            continue
        ti = times[idx[i]]
        for j in range(i + 1, m):
            if len(idx[j]) == 0:
                continue
            block = score[np.ix_(idx[i], idx[j])].ravel(); block = block[np.isfinite(block)]
            if len(block):
                app[i, j] = app[j, i] = float(np.mean(np.sort(block)[-topk:]))
            dt[i, j] = dt[j, i] = float(np.min(np.abs(ti[:, None] - times[idx[j]][None, :])))
            same_burst[i, j] = same_burst[j, i] = tracks[i]["burst"] == tracks[j]["burst"]
    return app, dt, same_burst


def self_threshold(score: np.ndarray, tracks: list[dict], pos: dict, times: np.ndarray, time_weight: float, thr_offset: float = 0.5, cam: np.ndarray | None = None, photo: list | None = None) -> dict:
    """Midpoint between the median score of burst pairs (instances <= 2 s apart on one camera, different photos: almost
    surely one whale, independent of the tracklet linking) and that of pairs > 90 min apart (almost surely different)."""
    idx = np.array(sorted(pos.values()))
    iu = np.triu_indices(len(idx), 1); a, b = idx[iu[0]], idx[iu[1]]
    s = score[a, b]; dts = np.abs(times[a] - times[b]); ok = np.isfinite(s)
    same_cam = (cam[a] == cam[b]) if cam is not None else np.ones(len(a), bool)
    diff_photo = (np.array(photo)[a] != np.array(photo)[b]) if photo is not None else np.ones(len(a), bool)
    near = s[ok & (dts <= 2.0) & same_cam & diff_photo]
    far = s[ok & (dts > 5400)]
    if len(near) < 10 or len(far) < 10:
        return {"threshold": 0.0, "near_pairs": len(near), "far_pairs": int(len(far))}
    nm, fm = float(np.median(near)), float(np.median(far))
    return {"threshold": 0.5 * (nm + fm) + time_weight * thr_offset, "near_median": nm, "far_median": fm, "near_pairs": len(near), "far_pairs": int(len(far))}


def run_cluster(lay: dict, ch: dict, fusion: dict | None, app_emb: dict | None, a) -> tuple[dict, dict]:
    keys, pos, times = lay["keys"], lay["pos"], lay["times"]
    score = fuse_scores(ch, fusion)
    if a.main_only:                                                       # ablation: secondary instances dropped from the graph
        pos = {k: i for k, i in pos.items() if "~" not in k or k.endswith("~0")}
    elif a.secondary_min_frac > 0:                                        # small secondary instances (distant or partial whales) stay out
        area = {i["key"]: i.get("area") or 0 for v in lay["instances"].values() for i in v}
        main_area = {s: (v[0].get("area") or 1) for s, v in lay["instances"].items() if v}
        pos = {k: i for k, i in pos.items() if "~" not in k or k.endswith("~0") or area[k] >= a.secondary_min_frac * main_area.get(lay["photo"][k], 1)}
    tracks = make_tracklets(lay["ts"], lay["instances"], a.gap, app_emb, a.max_link_cost)
    tracks = [t for t in tracks if any(k in pos for k in t["keys"])]
    multi = {s: v for s, v in lay["instances"].items() if len(v) > 1}
    if multi and a.cannot_link != "none":
        cl_inst = cannot_link_instances(multi, instance_coverage(a.day_dir, list(multi)) if a.cannot_link == "coverage" else None, rule=a.cannot_link)
    else:
        cl_inst = set()
    track_of = {k: i for i, t in enumerate(tracks) for k in t["keys"]}
    CL = np.zeros((len(tracks), len(tracks)), bool)
    for x, y in cl_inst:                                                  # only between graph nodes: a gated-out blob must not constrain through its tracklet
        if x in pos and y in pos and x in track_of and y in track_of and track_of[x] != track_of[y]:
            CL[track_of[x], track_of[y]] = CL[track_of[y], track_of[x]] = True
    app, dt, same_burst = node_scores(tracks, score, times, pos)
    tt = time_term(dt); tt[same_burst] = 0.0
    total = np.where(np.isfinite(app), app, 0.0) + a.time_weight * tt; np.fill_diagonal(total, 0.0)
    cal = self_threshold(score, tracks, pos, times, a.time_weight, a.thr_offset, lay["cam"], [lay["photo"][k] for k in keys])
    thr = cal["threshold"] if a.threshold is None else a.threshold
    lab_t = constrained_average_linkage(total, thr, CL)
    pred = np.full(len(keys), -1)
    for t, c in zip(tracks, lab_t):
        for k in t["keys"]:
            if k in pos:
                pred[pos[k]] = c
    pred[pred < 0] = pred.max() + 1 + np.arange((pred < 0).sum())
    return {"score": score, "tracks": tracks, "CL": CL, "app": app, "dt": dt, "same_burst": same_burst, "total": total, "pred": pred, "threshold": thr, "cal": cal, "cannot_link_pairs": len(cl_inst)}, {}


def photo_level(lay: dict, pred: np.ndarray, labels: dict | None) -> tuple[list[str], np.ndarray, list[str], list[str]]:
    """Main instances only: photo stems, their predicted cluster, the catalog animal and group ('?' when unknown)."""
    stems, pc, gt_a = [], [], []
    for k, i in lay["pos"].items():
        if "~" in k and not k.endswith("~0"):
            continue
        stems.append(lay["photo"][k]); pc.append(pred[i]); gt_a.append((labels or {}).get(lay["photo"][k].split("__")[-1], "?"))
    gt_g = [g.split("_")[-1][:3] if g != "?" else "?" for g in gt_a]
    return stems, np.array(pc), gt_a, gt_g


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["channels", "cluster", "both"])
    p.add_argument("--day-dir", type=Path, required=True); p.add_argument("--out", type=Path, required=True); p.add_argument("--parts-tag", default="parts_hr")
    p.add_argument("--sources", default="md"); p.add_argument("--local", default=""); p.add_argument("--candidates", type=int, default=30)
    p.add_argument("--candidate-sources", default=None, help="sources (as given to --sources) that rank the candidate pairs of the local scorers")
    p.add_argument("--local-parts", default=None, help="comma list of part types for the local scorers (default all)")
    p.add_argument("--labels", type=Path, default=None); p.add_argument("--fusion", type=Path, default=None); p.add_argument("--device", default="cuda")
    p.add_argument("--gap", type=float, default=1.5); p.add_argument("--max-link-cost", type=float, default=1.5); p.add_argument("--time-weight", type=float, default=1.5)
    p.add_argument("--threshold", type=float, default=None); p.add_argument("--thr-offset", type=float, default=0.0, help="self-calibrated threshold = midpoint + time_weight x offset")
    p.add_argument("--group-window", type=float, default=120.0); p.add_argument("--group-min-pairs", type=int, default=10)
    p.add_argument("--tag", default="", help="suffix for the cluster outputs (metrics<tag>.json, scores<tag>.npz)")
    p.add_argument("--use", default=None, help="cluster: comma list of channel sources to use (default all saved channels)")
    p.add_argument("--main-only", action="store_true", help="ablation: cluster the main instances only")
    p.add_argument("--cannot-link", default="fin", choices=["fin", "coverage", "none"], help="which instance pairs of one photo are two whales for sure (tracklets.py)")
    p.add_argument("--secondary-min-frac", type=float, default=0.25, help="a secondary instance joins the graph only if its area is this share of the photo's main instance")
    a = p.parse_args(argv)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    day = a.day_dir.name; lay = day_layout(a.day_dir, a.parts_tag); a.out.mkdir(parents=True, exist_ok=True)
    n = len(lay["keys"]); print(f"{day}: {n} instances from {len(set(lay['photo'].values()))} photos; charts {{{', '.join(f'{p}: {len(v)}' for p, v in lay['charts'].items())}}}", flush=True)
    chan_file = a.out / "channels.npz"
    ch = {}
    if chan_file.exists():
        z = np.load(chan_file, allow_pickle=True)
        if list(z["keys"]) == lay["keys"]:
            ch = {k: z[k] for k in z.files if "|" in k}
        else:
            print("  channels.npz has a different instance list: recomputing", flush=True)
    if a.cmd in ("channels", "both"):
        sources = [s for s in a.sources.split(",") if s]; local = [l for l in a.local.split(",") if l]
        ch, emb = compute_channels(lay, a.day_dir, sources, local, a.candidates, device, ch, [channel_source(s) for s in a.candidate_sources.split(",")] if a.candidate_sources else None,
                                   local_parts=tuple(a.local_parts.split(",")) if a.local_parts else PARTS)
        tmp = chan_file.with_suffix(".tmp.npz"); np.savez_compressed(tmp, keys=np.array(lay["keys"]), **ch); tmp.replace(chan_file)      # atomic: readers see a complete file
        if emb:
            np.savez_compressed(a.out / "embeddings.npz", **{f"{s}|{p}|idx": v[0] for (s, p), v in emb.items()}, **{f"{s}|{p}|E": v[1] for (s, p), v in emb.items()})
        print(f"  channels: {sorted(ch)} -> {chan_file}", flush=True)
    if a.cmd in ("cluster", "both"):
        if not ch:
            raise SystemExit("no channels: run channels first")
        fusion = json.loads(a.fusion.read_text()) if a.fusion else None
        if a.use:
            use = a.use.split(","); ch = {k: v for k, v in ch.items() if k.split("|")[0] in use}
            print(f"  using channels {sorted(ch)}", flush=True)
        app_emb = None
        if (a.out / "embeddings.npz").exists():
            z = np.load(a.out / "embeddings.npz", allow_pickle=True)
            ids = sorted({k.rsplit("|", 2)[0] for k in z.files if k.startswith("id:") and k.endswith("|body|idx")})
            if ids:
                idx, E = z[f"{ids[0]}|body|idx"], z[f"{ids[0]}|body|E"]; app_emb = {lay["keys"][i]: E[j] for j, i in enumerate(idx)}
        res, _ = run_cluster(lay, ch, fusion, app_emb, a)
        L = json.loads(a.labels.read_text())["days"].get(day, {}).get("photos", {}) if a.labels else {}
        stems, pc, gt_a, gt_g = photo_level(lay, res["pred"], L)
        ptimes = np.array([lay["ts"][s]["time"] for s in stems], float)
        groups = groups_from_animals(pc, ptimes, a.group_window, a.group_min_pairs)
        animals, grp = evaluate(pc, gt_a), evaluate(groups, gt_g)
        sweep = []
        for thr in np.linspace(np.nanpercentile(res["total"], 50), np.nanpercentile(res["total"], 99.5), 25):
            lab_t = constrained_average_linkage(res["total"], float(thr), res["CL"]); pp = np.full(n, -1)
            for t, c in zip(res["tracks"], lab_t):
                for k in t["keys"]:
                    if k in lay["pos"]:
                        pp[lay["pos"][k]] = c
            pp[pp < 0] = pp.max() + 1 + np.arange((pp < 0).sum())
            m = evaluate(photo_level(lay, pp, L)[1], gt_a)
            if m:
                sweep.append({"threshold": round(float(thr), 3), "ari": round(m["ari"], 4), "pair_f1": round(m["pair_f1"], 4), "clusters": m["clusters"]})
        out = {"day": day, "instances": n, "photos": len(stems), "tracklets": len(res["tracks"]), "cannot_link_pairs": res["cannot_link_pairs"], "channels": sorted(ch),
               "fusion": str(a.fusion) if a.fusion else None, "time_weight": a.time_weight, "threshold": res["threshold"], "calibration": res["cal"],
               "animals": animals, "groups": grp, "oracle_sweep_animals": sweep}
        (a.out / f"metrics{a.tag}.json").write_text(json.dumps(out, indent=1))
        np.savez_compressed(a.out / f"scores{a.tag}.npz", keys=np.array(lay["keys"]), inst_score=res["score"].astype(np.float32), node_app=res["app"], node_dt=res["dt"], node_total=res["total"],
                            same_burst=res["same_burst"], cannot_link=res["CL"], pred_instance=res["pred"], photo_stems=np.array(stems), pred_photo=pc, pred_group=groups,
                            near_median=res["cal"].get("near_median", np.nan), far_median=res["cal"].get("far_median", np.nan))
        (a.out / f"tracklets{a.tag}.json").write_text(json.dumps(res["tracks"], indent=1))
        print(f"  calibration: {res['cal']}  tracklets {len(res['tracks'])}  cannot-link pairs {res['cannot_link_pairs']}", flush=True)
        if animals:
            print(f"  animals: clusters {animals['clusters']} (truth {animals['individuals']})  ARI {animals['ari']:.3f}  pair P/R/F1 {animals['pair_precision']:.3f}/{animals['pair_recall']:.3f}/{animals['pair_f1']:.3f}  purity {animals['cluster_purity']:.3f}  thr {res['threshold']:.2f}  oracle ARI {max(s['ari'] for s in sweep):.3f}")
            print(f"  groups:  clusters {grp['clusters']} (truth {grp['individuals']})  ARI {grp['ari']:.3f}  F1 {grp['pair_f1']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
