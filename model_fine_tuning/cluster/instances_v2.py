"""Whales first: per-photo whale instances straight from the segmentation's RAW mask, judged one by one by the existing
not-a-whale filter, with the two blades of one fluke joined by a fixed rule. An OFFLINE prototype next to the live
pipeline (canon/predict.py keep_main_components -> one crop -> body map -> parts.split_instances): nothing live reads
it, nothing of a day's existing folders is changed, the body map is not used. LOG 2026-11-17.

    python -m model_fine_tuning.cluster.instances_v2 candidates --day-dir DAY [--collect 100] [--workers 4]
        GPU, inference only: seg_r04_ft4 exactly as canon/predict.py runs it (segment(), long side 1280, threshold 0.5)
        but WITHOUT the 25 % clean-up; every 8-connected blob of at least --collect px (2560-px frame) is a candidate.
        -> DAY/instances_v2/blobs/<stem>.png   uint8, blob id (1 = largest) per pixel at the mask's resolution, 0 = none
           DAY/instances_v2/blobs.jsonl        per photo: blobs (id, area, box, centroid, long / short side, angle)
                                               and the blob pairs close enough to be one whale (gap = least pixel distance)
           DAY/instances_v2/feat.npz           per blob and per close pair (its union box): the not-a-whale features
                                               (link_scores.extract, square crop) and the view features (view_class.extract)
    python -m model_fine_tuning.cluster.instances_v2 assemble --day-dir DAY [rule options]
        CPU: the rule -> DAY/instances_v2/index.jsonl, masks/<stem>.png, mapping.json, summary.json

Rule (defaults = the choices of LOG 2026-11-17; the evidence is in experiments/instances_v2/):
  1. candidates: blobs of at least --floor px (1000, 2560-px frame), however small against the photo's largest blob;
     the photo's largest blob is a candidate from --main-floor px (200, canon/predict.py's --min-pixels).
  2. fluke blades: two candidates become one instance when the union box looks like a fluke to the view classifier
     (P >= --join-pfluke 0.5), they lie side by side (centroid height difference <= --join-dy 0.6 x the taller blob's
     height, horizontal overlap of the boxes <= --join-xover 0.25 x the narrower width), the least pixel distance is
     <= --join-gap 0.25 x the union's width, and the larger area is at most --join-ratio 6 x the smaller. Each blob
     joins at most once (closest pair first). Nothing else is joined: a later stage can link two pieces of one whale,
     but it cannot split a mixed one (a gap of 0.25-0.35 adds true blade pairs but also two-whale pairs).
  3. not a whale: an instance whose score (link_scores not_whale.pkl on its own box, threshold --nw-threshold 2.0) is
     above the threshold is dropped; an instance WITHOUT the photo's largest blob must also pass the blob gate
     (experiments/instances_v2/blob_gate.pkl, `fit-gate`: a logistic regression on the view features fitted on the
     AI's by-eye labels of 189 blobs the 25 % rule deletes; it tells whale from splash, wake, water, shore, which the
     not-a-whale filter was never trained on and keeps). --secondary-frac 0.25 limits the gate to the instances the
     25 % rule would have deleted.
  4. keys: every kept instance gets a NEW key "<stem>~n<j>" (j = 0 the largest kept instance). The letter makes the
     key impossible to confuse with an old "<stem>~<k>" and makes any code that int()s the suffix fail loudly.
     mapping.json maps every old key (DAY/parts_hr/index.jsonl, old masks rebuilt from DAY/cse exactly as
     parts.split_instances made them) to the new instance it overlaps most (pixels of the 2560-px frame), with the
     overlap, all overlapped new instances, and whether the old key was dropped by safe/not_whale.json; new
     instances that no old key overlaps are listed under "new_only".
  5. crop: per kept instance its tight box and a square crop (x0, y0, s) as canon/predict.py makes it for the whole
     mask (square_crop_params(box, 512, 1.6)), so a later per-whale stage can cut each whale on its own.

index.jsonl, one row per photo: {"stem", "W", "H", "raw_px", "instances": [{"key", "id" (value in masks/<stem>.png),
"blobs" (ids in blobs/<stem>.png), "area", "bbox", "crop_x0y0s", "nw", "p_fluke", "p_dorsal", "joined", "secondary", "gate",
"old"}], "rejected": [{..., "why": not_whale | gate}]}. Old keys without overlap get "nearest" (a hint only) in mapping.json.
    python -m model_fine_tuning.cluster.instances_v2 fit-gate     -> experiments/instances_v2/blob_gate.pkl (see 3.)
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
SEG = REPO / "experiments" / "seg_r04_ft4" / "checkpoints" / "best.pt"
OUT = "instances_v2"
MAX_BLOBS = 250
PAIR_TOP = 8               # pairs are looked at among the largest blobs of a photo only
PAIR_GAP = 1.0             # a pair is stored when its gap is at most this x the longer of the two long sides
GATE = REPO / "experiments" / "instances_v2" / "blob_gate.pkl"
DEFAULTS = {"floor": 1000, "main_floor": 200, "nw_threshold": 2.0, "secondary_frac": 1e9, "join_pfluke": 0.5, "join_dy": 0.6, "join_gap": 0.25, "join_ratio": 6.0, "join_xover": 0.25}


# ------------------------------------------------------------------------------------------------------ candidates
def blob_table(mask: np.ndarray, collect: int):
    """Connected blobs (8-connected) of a raw mask, largest first: (label image with ids 1..n, records)."""
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros(mask.shape, np.uint8), []
    area = st[1:, cv2.CC_STAT_AREA]
    order = [k for k in np.lexsort((st[1:, cv2.CC_STAT_LEFT], st[1:, cv2.CC_STAT_TOP], -area)) if area[k] >= collect][:MAX_BLOBS]
    lut = np.zeros(n, np.uint8)
    recs = []
    for i, k in enumerate(order):
        lut[k + 1] = i + 1
        x, y, w, h = (int(v) for v in st[k + 1, :4])
        sub = lab[y:y + h, x:x + w] == k + 1
        ys, xs = np.nonzero(sub)
        pts = np.stack([xs, ys], 1).astype(np.float64); mu = pts.mean(0)
        if len(pts) > 2:
            ev, evec = np.linalg.eigh(np.cov((pts - mu).T)); ang = float(np.degrees(np.arctan2(evec[1, 1], evec[0, 1])))
        else:
            ang = 0.0
        cs, _ = cv2.findContours(sub.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cpts = np.concatenate([c.reshape(-1, 2) for c in cs]).astype(np.float32) + [x, y]
        recs.append({"b": i + 1, "area": int(area[k]), "bbox": [x, y, x + w, y + h], "c": [round(float(cen[k + 1][0]), 1), round(float(cen[k + 1][1]), 1)],
                     "long": int(max(w, h)), "angle": round(((ang + 90) % 180) - 90, 1), "_contour": cpts[:: max(1, len(cpts) // 600)]})
    return lut[lab], recs


def close_pairs(recs: list) -> list:
    """Blob pairs among the PAIR_TOP largest whose least contour distance is at most PAIR_GAP x the longer long side."""
    from scipy.spatial.distance import cdist
    out = []
    top = recs[:PAIR_TOP]
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            a, b = top[i], top[j]; lim = PAIR_GAP * max(a["long"], b["long"])
            ba, bb = a["bbox"], b["bbox"]
            dx = max(ba[0] - bb[2], bb[0] - ba[2], 0); dy = max(ba[1] - bb[3], bb[1] - ba[3], 0)
            if np.hypot(dx, dy) > lim:
                continue
            g = float(cdist(a["_contour"], b["_contour"]).min())
            if g <= lim:
                out.append({"i": a["b"], "j": b["b"], "gap": round(g, 1)})
    return out


def union_box(recs_by_id: dict, ids) -> list:
    bs = np.array([recs_by_id[i]["bbox"] for i in ids], float)
    return [float(bs[:, 0].min()), float(bs[:, 1].min()), float(bs[:, 2].max()), float(bs[:, 3].max())]


def nw_crop(box) -> tuple:
    """link_scores.instance_crops: the square crop of an instance box (x 1.15, at least 64 px)."""
    x0, y0, x1, y1 = box
    return ((x0 + x1) / 2, (y0 + y1) / 2, max(max(x1 - x0, y1 - y0) * 1.15, 64.0))


def cmd_candidates(a) -> None:
    import torch
    from ..segmentation.model import WhaleSegModel
    from ..segmentation.predict import segment
    from . import link_scores, view_class
    day = Path(a.day_dir); out = day / OUT; (out / "blobs").mkdir(parents=True, exist_ok=True); t = time.time()
    stems = sorted(p.stem for p in (day / "images").glob("*.jpg"))
    dev = torch.device("cuda"); model, _ = WhaleSegModel.from_checkpoint(Path(a.seg_checkpoint), dev)

    class Photos(torch.utils.data.Dataset):
        def __len__(self):
            return len(stems)

        def __getitem__(self, j):
            return j, cv2.cvtColor(cv2.imread(str(day / "images" / f"{stems[j]}.jpg"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    rows = []
    dl = torch.utils.data.DataLoader(Photos(), batch_size=None, num_workers=a.workers)
    for j, img in dl:
        img = img.numpy() if hasattr(img, "numpy") else img
        _, mask = segment(model, img, dev, 1280, 0.5)
        lab, recs = blob_table(mask, a.collect)
        cv2.imwrite(str(out / "blobs" / f"{stems[j]}.png"), lab)
        pairs = close_pairs(recs)
        for r in recs:
            r.pop("_contour")
        rows.append({"stem": stems[j], "W": int(img.shape[1]), "H": int(img.shape[0]), "raw_px": int(mask.sum()), "blobs": recs, "pairs": pairs})
    del model; torch.cuda.empty_cache()
    print(f"{day.name}: {len(rows)} photos segmented, {sum(len(r['blobs']) for r in rows)} blobs >= {a.collect} px, {sum(len(r['pairs']) for r in rows)} close pairs ({time.time() - t:.0f} s)", flush=True)
    keys, boxes = [], []
    for r in rows:
        by = {b["b"]: b for b in r["blobs"]}
        for b in r["blobs"]:
            keys.append(f"{r['stem']}|{b['b']}"); boxes.append(b["bbox"])
        for p in r["pairs"]:
            keys.append(f"{r['stem']}|{p['i']}+{p['j']}"); boxes.append(union_box(by, (p["i"], p["j"])))
    stem_of = [k.split("|")[0] for k in keys]
    F = link_scores.extract(day, [(s, *nw_crop(b)) for s, b in zip(stem_of, boxes)], workers=a.workers, log=0) if keys else {"cls": np.zeros((0, 768), np.float16), "mean": np.zeros((0, 768), np.float16)}
    V = view_class.extract(lambda s: day / "images" / f"{s}.jpg", list(zip(stem_of, boxes)), workers=a.workers) if keys else np.zeros((0, 4 * 768), np.float16)
    with open(out / "blobs.jsonl.tmp", "w") as h:
        for r in rows:
            h.write(json.dumps(r) + "\n")
    (out / "blobs.jsonl.tmp").replace(out / "blobs.jsonl")
    tmp = out / "feat.tmp.npz"
    np.savez(tmp, keys=np.array(keys), box=np.array(boxes, np.float32).reshape(-1, 4), nw_cls=F["cls"], nw_mean=F["mean"], view=V); tmp.replace(out / "feat.npz")
    (out / "candidates.json").write_text(json.dumps({"seg": str(SEG if a.seg_checkpoint == str(SEG) else a.seg_checkpoint), "max_side": 1280, "threshold": 0.5, "collect_px": a.collect,
                                                    "pair_top": PAIR_TOP, "pair_gap": PAIR_GAP, "photos": len(rows), "made": time.strftime("%Y-%m-%d %H:%M")}, indent=1))
    print(f"{day.name}: {len(keys)} boxes with features ({time.time() - t:.0f} s)", flush=True)


# ---------------------------------------------------------------------------------------------------------- scores
def load_scores(day: Path) -> dict:
    """Per feature key ("<stem>|<blob>" or "<stem>|<i>+<j>"): the not-a-whale score and P(fluke), P(dorsal fin)."""
    from . import view_class
    z = np.load(day / OUT / "feat.npz"); keys = [str(k) for k in z["keys"]]
    m = pickle.load(open(REPO / "experiments" / "safe_clusters" / "not_whale.pkl", "rb"))
    unit = lambda x: x.astype(np.float32) / np.maximum(np.linalg.norm(x.astype(np.float32), axis=1, keepdims=True), 1e-6)
    nw = ((np.hstack([unit(z["nw_cls"]), unit(z["nw_mean"])]) - m["mean"]) / m["scale"]) @ m["coef"] + m["intercept"] if keys else np.zeros(0)
    P = view_class.probability(view_class.load_model(), view_class.unit_blocks(z["view"])) if keys else np.zeros((0, 2))
    G = gate_score(z["view"]) if keys else np.zeros(0)
    return {k: (float(s), float(p[0]), float(p[1]), float(g)) for k, s, p, g in zip(keys, nw, P, G)}


# ------------------------------------------------------------------------------------------------------- blob gate
def gate_score(view: np.ndarray, path: Path = GATE) -> np.ndarray:
    """The blob gate (fit-gate): decision value whale (> 0 side) against splash / water / gear for a blob the 25 % rule
    would have deleted; NaN when the gate file does not exist."""
    from . import view_class
    if not path.exists():
        return np.full(len(view), np.nan)
    g = pickle.load(open(path, "rb"))
    return g["model"].decision_function(view_class.unit_blocks(view)) - g["threshold"]


def cmd_fit_gate(a) -> None:
    """Fit the blob gate on the AI's by-eye labels of blobs the 25 % rule deletes (experiments/instances_v2/
    blob_sample.json + blob_labels_ai.json): standardise, PCA 32, logistic regression (C 0.1, balanced) on the view
    features of the blob's own box. The threshold keeps --recall of the whale blobs >= --floor px in day-grouped
    cross-validation. The original not_whale.pkl is not touched."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from . import view_class
    here = REPO / "experiments" / "instances_v2"; sample = json.loads((here / a.sample).read_text()); L = json.loads((here / a.labels).read_text())["labels"]
    X, y, D, A = [], [], [], []; cache = {}
    for smp in sample:
        lab = L.get(str(smp["tile"]))
        if lab is None or lab == "unclear":
            continue
        if smp["day"] not in cache:
            z = np.load(Path(a.root) / smp["day"] / OUT / "feat.npz"); cache[smp["day"]] = ({str(k): i for i, k in enumerate(z["keys"])}, z["view"])
        kid, V = cache[smp["day"]]; X.append(V[kid[f"{smp['stem']}|{smp['b']}"]]); y.append(lab in ("whale_part", "second_whale")); D.append(smp["day"]); A.append(smp["area"])
    X = view_class.unit_blocks(np.array(X)); y = np.array(y); D = np.array(D); A = np.array(A)
    make = lambda: make_pipeline(StandardScaler(), PCA(32, random_state=0), LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000))
    ud = np.array(sorted(set(D))); np.random.default_rng(0).shuffle(ud); S = np.zeros(len(y))
    for f in np.array_split(ud, 6):
        te = np.isin(D, f); S[te] = make().fit(X[~te], y[~te]).decision_function(X[te])
    sel = A >= a.floor; ws = np.sort(S[sel & y]); thr = float(ws[int(np.floor((1 - a.recall) * len(ws)))])
    k = sel & (S >= thr)
    rep = (f"{len(y)} labelled blobs ({int(y.sum())} whale) on {len(ud)} days; day-grouped CV, blobs >= {a.floor} px: threshold {thr:.3f} keeps "
           f"{int((k & y).sum())} / {int((sel & y).sum())} whales and {int((k & ~y).sum())} / {int((sel & ~y).sum())} non-whales (precision {(k & y).sum() / max(k.sum(), 1):.2f})")
    print(rep)
    g = {"name": "blob_gate_v1", "model": make().fit(X, y), "threshold": thr, "features": "view_class features of the blob box (unit blocks)", "labels": f"{a.sample} + {a.labels} (AI by-eye labels, not the researcher's)",
         "cv": rep, "fitted": time.strftime("%Y-%m-%d")}
    pickle.dump(g, open(Path(a.out), "wb")); print("saved", a.out)



# ----------------------------------------------------------------------------------------------------- fluke join
def pair_geometry(a: dict, b: dict, gap: float) -> dict:
    """Side by side? dy = centroid height difference / the taller box height; xover = horizontal overlap of the boxes /
    the narrower width (> 0 overlap); gap_w = least pixel distance / the union width; ratio = larger / smaller area."""
    ba, bb = a["bbox"], b["bbox"]; ha, hb = ba[3] - ba[1], bb[3] - bb[1]; wa, wb = ba[2] - ba[0], bb[2] - bb[0]
    uw = max(ba[2], bb[2]) - min(ba[0], bb[0])
    return {"dy": abs(a["c"][1] - b["c"][1]) / max(ha, hb, 1), "xover": (min(ba[2], bb[2]) - max(ba[0], bb[0])) / max(min(wa, wb), 1),
            "gap_w": gap / max(uw, 1), "ratio": max(a["area"], b["area"]) / max(min(a["area"], b["area"]), 1)}


def is_fluke_pair(g: dict, p_union: float, R: dict) -> bool:
    return p_union >= R["join_pfluke"] and g["dy"] <= R["join_dy"] and g["xover"] <= R["join_xover"] and g["gap_w"] <= R["join_gap"] and g["ratio"] <= R["join_ratio"]


# --------------------------------------------------------------------------------------------------- old instances
def old_instances(day: Path, stem: str, W: int, H: int, table: dict) -> tuple[list, list]:
    """The old instance masks of a photo in the 2560-px frame (DAY/cse/<stem>_canon.npz split as parts.py --instances
    did: split_instances(use, 0.1, u=u)), with their keys "<stem>~k"."""
    from .parts import split_instances
    f = day / "cse" / f"{stem}_canon.npz"
    if not f.exists():
        return [], []
    z = np.load(f); use = z["use"]
    if use.sum() < 400:
        return [], []
    x0, y0, s = (float(v) for v in z["x0y0s"]); u = (z["signed_mm"][..., 0].astype(np.float32) / 1000.0 - table["x_min_m"]) / table["length_m"]
    masks = split_instances(use, 0.1, u=u)
    Minv = np.array([[1 / s, 0, x0], [0, 1 / s, y0]], np.float64)            # crop px -> 2560-px frame
    full = [cv2.warpAffine(m.astype(np.uint8), Minv, (W, H), flags=cv2.INTER_NEAREST) > 0 for m in masks]
    return [f"{stem}~{k}" for k in range(len(masks))], full


def cmd_assemble(a) -> None:
    from ..common import geometry, paths
    from ..common.data import square_crop_params
    day = Path(a.day_dir); out = day / OUT; (out / "masks").mkdir(exist_ok=True); t = time.time()
    R = {k: getattr(a, k) for k in DEFAULTS}
    S = load_scores(day); table = geometry.load_folded_table(paths.FOLDED_TABLE)
    drop_old = set(json.loads((day / "safe" / "not_whale.json").read_text()).get("drop", [])) if (day / "safe" / "not_whale.json").exists() else set()
    old_rows = {json.loads(l)["stem"]: json.loads(l) for l in open(day / "parts_hr" / "index.jsonl")} if (day / "parts_hr" / "index.jsonl").exists() else {}
    mapping, new_only, n_old_mismatch = {}, [], 0
    tot = {"photos": 0, "candidates": 0, "kept": 0, "joined": 0, "rejected_nw": 0, "rejected_gate": 0, "old_instances": 0, "old_kept": 0}
    with open(out / "index.jsonl.tmp", "w") as h:
        for line in open(out / "blobs.jsonl"):
            r = json.loads(line); stem = r["stem"]; by = {b["b"]: b for b in r["blobs"]}
            big = r["blobs"][0]["area"] if r["blobs"] else 0
            cand = [b for b in r["blobs"] if b["area"] >= R["floor"] or (b["b"] == 1 and b["area"] >= R["main_floor"])]; cid = {b["b"] for b in cand}
            # fluke blades: closest qualifying pair first, each blob once
            used, groups = set(), []
            for p in sorted(r["pairs"], key=lambda p: p["gap"]):
                i, j = p["i"], p["j"]
                if i in used or j in used or i not in cid or j not in cid:
                    continue
                g = pair_geometry(by[i], by[j], p["gap"]); sc = S.get(f"{stem}|{i}+{j}")
                if sc and is_fluke_pair(g, sc[1], R):
                    groups.append((i, j)); used |= {i, j}
            groups += [(b["b"],) for b in cand if b["b"] not in used]
            lab = cv2.imread(str(out / "blobs" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
            kept, rejected = [], []
            for gr in groups:
                fk = f"{stem}|{gr[0]}" if len(gr) == 1 else f"{stem}|{gr[0]}+{gr[1]}"
                nw, pf, pd, gt = S[fk]; box = union_box(by, gr); area = int(sum(by[i]["area"] for i in gr))
                secondary = 1 not in gr and area < R["secondary_frac"] * big          # default: every instance without the photo's largest blob
                rec = {"blobs": list(gr), "area": area, "bbox": [round(v, 1) for v in box], "nw": round(nw, 3), "p_fluke": round(pf, 3), "p_dorsal": round(pd, 3), "joined": len(gr) > 1,
                       "secondary": secondary, "gate": None if not secondary or np.isnan(gt) else round(gt, 3)}
                if nw > R["nw_threshold"]:
                    rec["why"] = "not_whale"; rejected.append(rec)
                elif secondary and not np.isnan(gt) and gt < 0:
                    rec["why"] = "gate"; rejected.append(rec)
                else:
                    kept.append(rec)
            kept.sort(key=lambda x: -x["area"])
            inst_map = np.zeros(lab.shape, np.uint8); lut = np.zeros(256, np.uint8)
            for n, rec in enumerate(kept):
                rec["key"] = f"{stem}~n{n}"; rec["id"] = n + 1
                x0, y0, s = square_crop_params(rec["bbox"], 512, 1.6, 0.0, None); rec["crop_x0y0s"] = [round(float(x0), 1), round(float(y0), 1), round(float(s), 5)]
                lut[rec["blobs"]] = n + 1
            inst_map = lut[lab]
            cv2.imwrite(str(out / "masks" / f"{stem}.png"), inst_map)
            # old keys -> new instance of the largest overlap
            okeys, omasks = old_instances(day, stem, r["W"], r["H"], table)
            ok_index = [i["key"] for i in (old_rows.get(stem) or {}).get("instances", [])]
            if ok_index and okeys != ok_index:
                n_old_mismatch += 1
            hit = {rec["key"]: 0 for rec in kept}
            for k, m in zip(okeys, omasks):
                ids, cnt = np.unique(inst_map[m], return_counts=True); ov = {int(i): int(c) for i, c in zip(ids, cnt) if i > 0}
                best = max(ov, key=ov.get) if ov else None
                mapping[k] = {"new": kept[best - 1]["key"] if best else None, "overlap_px": ov.get(best, 0) if best else 0, "old_px": int(m.sum()),
                              "all": {kept[i - 1]["key"]: c for i, c in sorted(ov.items())}, "old_dropped_not_whale": k in drop_old}
                if best:
                    hit[kept[best - 1]["key"]] += 1
                elif kept and m.any():                        # no overlap: the nearest kept instance, as a hint only (distance in px)
                    ys, xs = np.nonzero(m); c = np.array([xs.mean(), ys.mean()])
                    d = [float(np.hypot(*(np.array([(x["bbox"][0] + x["bbox"][2]) / 2, (x["bbox"][1] + x["bbox"][3]) / 2]) - c))) for x in kept]
                    j = int(np.argmin(d)); mapping[k]["nearest"] = {"key": kept[j]["key"], "box_centre_px": round(d[j], 1)}
                for i in ov:
                    kept[i - 1].setdefault("old", []).append(k)
                tot["old_instances"] += 1; tot["old_kept"] += k not in drop_old
            new_only += [rec["key"] for rec in kept if "old" not in rec]
            h.write(json.dumps({"stem": stem, "W": r["W"], "H": r["H"], "raw_px": r["raw_px"], "instances": kept, "rejected": rejected}) + "\n")
            tot["photos"] += 1; tot["candidates"] += len(cand); tot["kept"] += len(kept); tot["joined"] += sum(x["joined"] for x in kept); tot["rejected_nw"] += sum(x["why"] == "not_whale" for x in rejected); tot["rejected_gate"] += sum(x["why"] == "gate" for x in rejected)
    (out / "index.jsonl.tmp").replace(out / "index.jsonl")
    (out / "mapping.json").write_text(json.dumps({"rule": R, "old_to_new": mapping, "new_only": new_only}, indent=0))
    summ = {"rule": R, **tot, "new_only": len(new_only), "old_unmapped": sum(v["new"] is None for v in mapping.values()), "old_key_list_mismatch_photos": n_old_mismatch, "made": time.strftime("%Y-%m-%d %H:%M")}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(f"{day.name}: {json.dumps(summ)} ({time.time() - t:.0f} s)", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("candidates"); p.add_argument("--day-dir", required=True); p.add_argument("--collect", type=int, default=100)
    p.add_argument("--workers", type=int, default=4); p.add_argument("--seg-checkpoint", default=str(SEG)); p.set_defaults(fn=cmd_candidates)
    p = sub.add_parser("assemble"); p.add_argument("--day-dir", required=True)
    for k, v in DEFAULTS.items():
        p.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    p.set_defaults(fn=cmd_assemble)
    p = sub.add_parser("fit-gate"); p.add_argument("--sample", default="blob_sample.json"); p.add_argument("--labels", default="blob_labels_ai.json")
    p.add_argument("--root", default=str(REPO / "experiments" / "days_all")); p.add_argument("--floor", type=int, default=1000); p.add_argument("--recall", type=float, default=0.95)
    p.add_argument("--out", default=str(GATE)); p.set_defaults(fn=cmd_fit_gate)
    a = ap.parse_args(argv); a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
