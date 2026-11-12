"""Link scores between the whale instances of a day from frozen DINOv3 window features: the second source of evidence
(next to the identity embedding) for ordering candidate links between safe clusters for the reviewer.

A researcher links two clusters through one small area of the photo (a dorsal fin, a scar, barnacles on a fluke tip).
So every instance is covered with square windows at several sizes, each window gets a frozen DINOv3 ViT-B/16 CLS
feature (whitened to 128 dimensions), a small "where to look" model (fitted on the boxes the researcher drew on
2025-01-02) picks the windows a researcher would use, and two instances are compared by the best cosine between
their picked windows of similar size. Nothing here is fine-tuned and no catalog label is read by this file's
feature and score steps.

    python -m model_fine_tuning.cluster.link_scores windows --day-dir DAY [--fit-whitening]
        -> DAY/safe/windows.npz    Z (windows x 128, float16, unit rows), inst (index into keys), geo (cx, cy, side,
                                   mask fraction; px of the 2560-px frame), keys (instance keys with a window)
           DAY/safe/inst_feat.npz  keys (every instance), cls / mean (768, float16) of the square crop on its box
        --fit-whitening fits the whitening on this day (done once, on 2025-01-02) -> experiments/safe_clusters/window_whitening.npz
    python -m model_fine_tuning.cluster.link_scores instances --day-dir DAY     -> DAY/safe/inst_feat.npz only (quick)
    python -m model_fine_tuning.cluster.link_scores not-whale --day-dir DAY     -> DAY/safe/not_whale.json: the instances that are
        not a whale (people on the bow, deck, gear, drones; experiments/safe_clusters/not_whale.pkl), read by safe_tracklets --filter
    python -m model_fine_tuning.cluster.link_scores look --ref-dir DAY --boxes B.pkl
        -> experiments/safe_clusters/where_to_look.pkl   (the where-to-look model from the researcher's region boxes)
    python -m model_fine_tuning.cluster.link_scores scores --day-dir DAY [--saliency S.npy]
        -> DAY/safe/window_scores.npz   keys, D_sal / D_all (instances x instances, float16; NaN = no comparable windows)
    python -m model_fine_tuning.cluster.link_scores pairs --day-dir DAY
        -> DAY/safe/link_scores.json    the candidate links between the day's safe clusters for the reviewer, scored (rm = the
                                        rank mean, lp = the short-gap linker's P); needs the day's clusters.json and embeddings

Validated in experiments/stage2_labels (LOG 2026-11-11): on the val days the rank mean of the student's fin and body
scores and D_sal finds more catalog-same cluster pairs at precision 0.90 / 0.95 than the student alone.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

SIZES = (100, 150, 225, 340, 510, 765, 1150)      # window sides, px of the 2560-px frame
DIM = 128                                          # whitened feature size
TOPK = 10                                          # windows per instance the where-to-look model keeps
MODELS = Path(__file__).resolve().parents[2] / "experiments" / "safe_clusters"
WHITENING = MODELS / "window_whitening.npz"
LOOK = MODELS / "where_to_look.pkl"
NOT_WHALE = MODELS / "not_whale.pkl"
LINKER = MODELS / "link_gbt.pkl"


# ------------------------------------------------------------------------------------------------- window proposals
def _integral(m: np.ndarray) -> np.ndarray:
    return np.pad(m.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))


def proposals(day_dir: Path, min_frac: float = 0.2, sizes=SIZES):
    """Square windows on every instance: at each size a grid with half-window stride over the instance box, kept where
    at least min_frac of the window is whale mask. Returns keys (sorted instance keys that got a window), inst (per
    window, index into keys) and geo (per window: cx, cy, side, mask fraction)."""
    from PIL import Image
    P = json.loads((day_dir / "safe" / "photos.json").read_text())
    rows = []
    for stem, p in P.items():
        if not p["instances"]:
            continue
        m = np.asarray(Image.open(day_dir / "canon" / f"{stem}_mask.png")) > 0
        H, W = m.shape; I = _integral(m)
        for it in p["instances"]:
            x0, y0, x1, y1 = it["bbox"]; bw, bh = x1 - x0, y1 - y0
            for s in sizes:
                st = s / 2
                xs = np.arange(x0, x1 + 1e-6, st) if bw > st else np.array([(x0 + x1) / 2])
                ys = np.arange(y0, y1 + 1e-6, st) if bh > st else np.array([(y0 + y1) / 2])
                for cy in ys:
                    for cx in xs:
                        a0 = int(np.clip(round(cx - s / 2), 0, W)); a1 = int(np.clip(round(cx + s / 2), 0, W))
                        b0 = int(np.clip(round(cy - s / 2), 0, H)); b1 = int(np.clip(round(cy + s / 2), 0, H))
                        if a1 <= a0 or b1 <= b0:
                            continue
                        fr = (I[b1, a1] - I[b0, a1] - I[b1, a0] + I[b0, a0]) / float(s * s)
                        if fr >= min_frac:
                            rows.append((it["key"], float(cx), float(cy), float(s), float(fr)))
    keys = sorted({r[0] for r in rows}); kid = {k: i for i, k in enumerate(keys)}
    inst = np.array([kid[r[0]] for r in rows], np.int32); geo = np.array([r[1:] for r in rows], np.float32).reshape(-1, 4)
    return keys, inst, geo


# ------------------------------------------------------------------------------------------------ frozen features
def _cut(im, cx, cy, side, R):
    from PIL import Image
    W, H = im.size; s = float(side); x0 = cx - s / 2; y0 = cy - s / 2
    if s <= W:
        x0 = min(max(x0, 0), W - s)
    if s <= H:
        y0 = min(max(y0, 0), H - s)
    c = im.crop((int(round(x0)), int(round(y0)), int(round(x0 + s)), int(round(y0 + s))))
    return c.resize((R, R), Image.BICUBIC if s < R else Image.LANCZOS)


def extract(day_dir: Path, crops: list, R: int = 224, bs: int = 256, workers: int = 4, log: int = 20000) -> dict:
    """Frozen DINOv3 ViT-B/16 features (last block, final norm) of square crops (stem, cx, cy, side) cut from the day's
    2560-px images: cls and the mean patch token, float16, aligned with crops."""
    import torch
    from PIL import Image
    from ..common.backbone import IMAGENET_MEAN, IMAGENET_STD, load_backbone
    mean = np.array(IMAGENET_MEAN, np.float32) * 255; std = np.array(IMAGENET_STD, np.float32) * 255
    by = defaultdict(list)
    for i, (stem, cx, cy, side) in enumerate(crops):
        by[stem].append((i, cx, cy, side))
    stems = sorted(by)

    class Photos(torch.utils.data.Dataset):
        def __len__(self):
            return len(stems)

        def __getitem__(self, j):
            im = Image.open(day_dir / "images" / f"{stems[j]}.jpg").convert("RGB"); idx, arr = [], []
            for i, cx, cy, side in by[stems[j]]:
                arr.append(((np.asarray(_cut(im, cx, cy, side, R), np.float32) - mean) / std).transpose(2, 0, 1)); idx.append(i)
            return torch.tensor(np.array(idx)), torch.from_numpy(np.stack(arr))

    model = load_backbone("vitb16").cuda().eval().half(); n = len(crops)
    out = {"cls": np.zeros((n, 768), np.float16), "mean": np.zeros((n, 768), np.float16)}
    # two photos per loader batch and few workers: a photo's crops are ~150 MB of float32 and every queued batch sits in /dev/shm
    dl = torch.utils.data.DataLoader(Photos(), batch_size=2, num_workers=workers, collate_fn=lambda b: (torch.cat([x[0] for x in b]), torch.cat([x[1] for x in b])))
    done = 0
    with torch.no_grad():
        for idx, x in dl:
            for k in range(0, len(idx), bs):
                xb = x[k:k + bs].cuda(non_blocking=True).half(); ib = idx[k:k + bs].numpy()
                pt, cl = model.get_intermediate_layers(xb, n=[11], reshape=False, return_class_token=True, norm=True)[0]
                out["cls"][ib] = cl.float().cpu().numpy(); out["mean"][ib] = pt.float().mean(1).cpu().numpy()
            done += len(idx)
            if log and done // log != (done - len(idx)) // log:
                print(f"  crops {done} / {n}", flush=True)
    return out


def whiten_fit(X: np.ndarray, dim: int = DIM, sample: int = 60000, seed: int = 0):
    rng = np.random.default_rng(seed); idx = rng.choice(len(X), min(sample, len(X)), replace=False)
    Xs = X[idx].astype(np.float32); Xs /= np.linalg.norm(Xs, axis=1, keepdims=True)
    mu = Xs.mean(0); _, s, Vt = np.linalg.svd(Xs - mu, full_matrices=False)
    return mu, (Vt[:dim].T / (s[:dim] / np.sqrt(len(Xs)))).astype(np.float32)


def transform(X: np.ndarray, mu: np.ndarray, Pm: np.ndarray, bs: int = 65536):
    """Unit rows -> centred, whitened, unit rows again (a half tensor on the GPU)."""
    import torch
    out = []
    for k in range(0, len(X), bs):
        x = torch.nn.functional.normalize(torch.tensor(X[k:k + bs].astype(np.float32)).cuda(), dim=1)
        x = (x - torch.tensor(mu).cuda()) @ torch.tensor(Pm).cuda()
        out.append(torch.nn.functional.normalize(x, dim=1).half())
    return torch.cat(out)


def instance_crops(day_dir: Path):
    """One square crop per instance (its box plus a margin): keys and (cx, cy, side). For the instance filter."""
    P = json.loads((day_dir / "safe" / "photos.json").read_text()); ikeys, ibox = [], []
    for stem, p in P.items():
        for it in p["instances"]:
            x0, y0, x1, y1 = it["bbox"]; ikeys.append(it["key"]); ibox.append(((x0 + x1) / 2, (y0 + y1) / 2, max(max(x1 - x0, y1 - y0) * 1.15, 64.0)))
    return ikeys, ibox


def _savez(path: Path, **arrays) -> None:
    """np.savez through a temporary file: a reader (or a killed run) never sees half a file."""
    tmp = path.with_name(path.stem + ".tmp.npz"); np.savez(tmp, **arrays); tmp.replace(path)


def cmd_instances(a) -> None:
    day_dir = Path(a.day_dir); ikeys, ibox = instance_crops(day_dir)
    F = extract(day_dir, [(k.split("~")[0], *b) for k, b in zip(ikeys, ibox)], log=0)
    _savez(day_dir / "safe" / "inst_feat.npz", keys=np.array(ikeys), cls=F["cls"], mean=F["mean"], box=np.array(ibox, np.float32))
    print(f"{day_dir.name}: {len(ikeys)} instance crops", flush=True)


def cmd_not_whale(a) -> None:
    """Score every instance of the day with the not-a-whale probe (a logistic regression on the frozen crop features,
    fitted on by-eye labels of the train days: people on the bow, the deck, gear, poles, drones) and list the ones
    above the threshold. safe_tracklets --filter leaves those out of the clustering."""
    day_dir = Path(a.day_dir); m = pickle.load(open(NOT_WHALE, "rb")); z = np.load(day_dir / "safe" / "inst_feat.npz")
    unit = lambda x: x.astype(np.float32) / np.maximum(np.linalg.norm(x.astype(np.float32), axis=1, keepdims=True), 1e-6)
    sc = ((np.hstack([unit(z["cls"]), unit(z["mean"])]) - m["mean"]) / m["scale"]) @ m["coef"] + m["intercept"]
    keys = [str(k) for k in z["keys"]]; drop = [k for k, v in zip(keys, sc) if v > m["threshold"]]
    (day_dir / "safe" / "not_whale.json").write_text(json.dumps({"model": m["name"], "threshold": m["threshold"], "drop": drop, "scores": {k: round(float(v), 3) for k, v in zip(keys, sc)}}, indent=1))
    print(f"{day_dir.name}: {len(drop)} of {len(keys)} instances are not a whale (score > {m['threshold']})", flush=True)


def cmd_windows(a) -> None:
    day_dir = Path(a.day_dir); t = time.time()
    keys, inst, geo = proposals(day_dir); n = len(inst)
    crops = [(keys[i].split("~")[0], float(g[0]), float(g[1]), float(g[2])) for i, g in zip(inst, geo)]
    ikeys, ibox = instance_crops(day_dir)
    crops += [(k.split("~")[0], *b) for k, b in zip(ikeys, ibox)]
    print(f"{day_dir.name}: {n} windows on {len(keys)} instances, {len(ikeys)} instance crops ({time.time() - t:.0f} s)", flush=True)
    F = extract(day_dir, crops)
    if a.fit_whitening:
        mu, Pm = whiten_fit(F["cls"][:n]); np.savez(WHITENING, mu=mu, Pm=Pm, day=day_dir.name, windows=n)
        print(f"whitening fitted on {day_dir.name} -> {WHITENING}", flush=True)
    w = np.load(WHITENING); Z = transform(F["cls"][:n], w["mu"], w["Pm"]).cpu().numpy()
    _savez(day_dir / "safe" / "inst_feat.npz", keys=np.array(ikeys), cls=F["cls"][n:], mean=F["mean"][n:], box=np.array(ibox, np.float32))
    _savez(day_dir / "safe" / "windows.npz", Z=Z, inst=inst, geo=geo, keys=np.array(keys), sizes=np.array(SIZES))      # last: its presence says the day is done
    print(f"{day_dir.name}: done in {time.time() - t:.0f} s", flush=True)


# ------------------------------------------------------------------------- where to look, instance-to-instance scores
def _sq(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, max(b[2] - b[0], b[3] - b[1]) * 1.15)


def _iou(geo: np.ndarray, c, ids: np.ndarray) -> np.ndarray:
    g = geo[ids]; x0 = np.maximum(c[0] - c[2] / 2, g[:, 0] - g[:, 2] / 2); x1 = np.minimum(c[0] + c[2] / 2, g[:, 0] + g[:, 2] / 2)
    y0 = np.maximum(c[1] - c[2] / 2, g[:, 1] - g[:, 2] / 2); y1 = np.minimum(c[1] + c[2] / 2, g[:, 1] + g[:, 2] / 2)
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    return inter / (c[2] ** 2 + g[:, 2] ** 2 - inter)


def load_windows(day_dir: Path) -> dict:
    z = np.load(day_dir / "safe" / "windows.npz", allow_pickle=False)
    keys = [str(k) for k in z["keys"]]; geo = z["geo"]; sizes = [int(v) for v in z["sizes"]]
    return {"Z": z["Z"], "inst": z["inst"], "geo": geo, "keys": keys, "kid": {k: i for i, k in enumerate(keys)}, "size": np.array([sizes.index(int(v)) for v in geo[:, 2]])}


def fit_look(d: dict, boxes: dict) -> dict:
    """The where-to-look model: logistic regression on the whitened window features, windows that overlap a box the
    researcher drew (IoU >= 0.3 with the squared box) against the windows of the same instances that miss every box
    (IoU < 0.05). boxes: instance key -> [(x0, y0, x1, y1) px of the 2560-px frame]."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    Z = d["Z"].astype(np.float32); by = defaultdict(list)
    for i, k in enumerate(d["inst"]):
        by[int(k)].append(i)
    lab = np.full(len(Z), -1, np.int8); n_inst = 0
    for key, bl in boxes.items():
        i = d["kid"].get(key)
        if i is None:
            continue
        w = np.array(by[i]); m = np.max(np.stack([_iou(d["geo"], _sq(b), w) for b in bl]), 0); lab[w[m >= 0.3]] = 1; lab[w[m < 0.05]] = 0; n_inst += 1
    tr = lab >= 0; sc = StandardScaler().fit(Z[tr]); clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=400).fit(sc.transform(Z[tr]), lab[tr])
    return {"mean": sc.mean_.astype(np.float32), "scale": sc.scale_.astype(np.float32), "coef": clf.coef_[0].astype(np.float32), "intercept": float(clf.intercept_[0]),
            "instances": n_inst, "positives": int((lab == 1).sum()), "negatives": int((lab == 0).sum())}


def saliency(look: dict, Z: np.ndarray) -> np.ndarray:
    return (((Z.astype(np.float32) - look["mean"]) / look["scale"]) @ look["coef"] + look["intercept"]).astype(np.float32)


def topk_scores(Z, sal: np.ndarray, inst: np.ndarray, size: np.ndarray, n: int, topk: int = TOPK) -> np.ndarray:
    """D[i, j] = the best cosine between the topk most salient windows of instance i and those of instance j, window
    sizes at most one step apart (NaN when no two windows compare). Z: unit float tensor on the GPU."""
    import torch
    by = defaultdict(list)
    for i, k in enumerate(inst):
        by[int(k)].append(i)
    idx = np.full((n, topk), -1, np.int64)
    for i, w in by.items():
        w = np.array(w); t = w[np.argsort(-sal[w])[:topk]]; idx[i, :len(t)] = t
    with torch.no_grad():
        valid = torch.tensor(idx >= 0).cuda(); ti = torch.tensor(np.where(idx >= 0, idx, 0)).cuda()
        Zs = Z[ti]; sz = torch.tensor(size).cuda()[ti]
        D = torch.full((n, n), float("nan"), device="cuda")
        for a in range(0, n, 64):
            S = torch.einsum("ikd,jld->ijkl", Zs[a:a + 64], Zs)
            ok = ((sz[a:a + 64][:, None, :, None] - sz[None, :, None, :]).abs() <= 1) & valid[a:a + 64][:, None, :, None] & valid[None, :, None, :]
            S = S.masked_fill(~ok, -9.0).flatten(2).amax(2); D[a:a + 64] = torch.where(S > -8, S, torch.full_like(S, float("nan")))
    return D.cpu().numpy()


def best_match(Z, inst: np.ndarray, size: np.ndarray, n_inst: int, step: int = 1, chunk: int = 4096) -> np.ndarray:
    """M[u, j] = the best cosine of window u to any window of instance j whose size is at most `step` steps away
    (-1 where there is none). Z: unit half tensor on the GPU."""
    import torch
    M = np.full((len(Z), n_inst), -1.0, np.float16)
    with torch.no_grad():
        for k in sorted(set(size.tolist())):
            ui = np.where(size == k)[0]; vi = np.where(np.abs(size - k) <= step)[0]
            vi = vi[np.argsort(inst[vi], kind="stable")]; js, starts = np.unique(inst[vi], return_index=True); ends = np.r_[starts[1:], len(vi)]
            Zv = Z[torch.tensor(vi).cuda()]
            for c in range(0, len(ui), chunk):
                u = ui[c:c + chunk]; S = Z[torch.tensor(u).cuda()] @ Zv.T
                mx = torch.empty((len(u), len(js)), dtype=torch.float16, device="cuda")
                for q, (a, b) in enumerate(zip(starts, ends)):
                    mx[:, q] = S[:, a:b].max(1)[0]
                M[np.ix_(u, js)] = mx.cpu().numpy()
    return M


def instance_scores(M: np.ndarray, inst: np.ndarray, n_inst: int) -> np.ndarray:
    """D[i, j] = the best match of any window of instance i in instance j."""
    D = np.full((n_inst, n_inst), np.nan, np.float32)
    order = np.argsort(inst, kind="stable"); js, starts = np.unique(inst[order], return_index=True); ends = np.r_[starts[1:], len(order)]
    for i, a, b in zip(js, starts, ends):
        blk = M[order[a:b]].astype(np.float32); blk[blk <= -1] = -1e9; D[i] = blk.max(0)
    D[D <= -1e8] = np.nan
    return D


def cmd_look(a) -> None:
    d = load_windows(Path(a.ref_dir)); regions = pickle.load(open(a.boxes, "rb"))["regions"]; boxes = defaultdict(list)
    for g in regions:                                      # experiments/stage2_labels/dataset.py: one dict per region pair, a box and its instance per side
        for s in "ab":
            boxes[g["ikey_" + s]].append(g["box_" + s])
    look = fit_look(d, boxes); look.update(day=Path(a.ref_dir).name, boxes=sum(map(len, boxes.values())), source=str(a.boxes))
    pickle.dump(look, open(LOOK, "wb"))
    print(f"where-to-look model: {look['boxes']} boxes on {look['instances']} instances, {look['positives']} windows on a box, {look['negatives']} off -> {LOOK}")


def cmd_scores(a) -> None:
    import torch
    day_dir = Path(a.day_dir); t = time.time(); d = load_windows(day_dir); n = len(d["keys"]); look = pickle.load(open(LOOK, "rb"))
    sal = np.load(a.saliency).astype(np.float32) if a.saliency else saliency(look, d["Z"])       # --saliency: out-of-fold values for the day the model was fitted on
    assert len(sal) == len(d["Z"]), "saliency does not match the windows"
    Z = torch.tensor(d["Z"]).cuda()
    D_sal = topk_scores(Z.float(), sal, d["inst"], d["size"], n); D_sal = np.fmax(D_sal, D_sal.T)
    D_all = instance_scores(best_match(Z, d["inst"], d["size"], n), d["inst"], n); D_all = np.fmax(D_all, D_all.T)
    _savez(day_dir / "safe" / "window_scores.npz", keys=np.array(d["keys"]), D_sal=D_sal.astype(np.float16), D_all=D_all.astype(np.float16), sal=sal.astype(np.float16))
    print(f"{day_dir.name}: {n} instances, D_sal finite {np.isfinite(D_sal).mean():.3f}, D_all finite {np.isfinite(D_all).mean():.3f} ({time.time() - t:.0f} s)", flush=True)


# ------------------------------------------------------------------------------------------ cluster-pair table
CONT_KEYS = ("d_pos", "d_size", "log_area_min", "d_focal", "d_ev", "d_range", "d_rot", "d_nose", "side_flip", "d_lab_whale", "d_lab_water", "d_asym", "d_contrast",
             "d_lum_std", "n_inst_max", "parts_shared", "parts_union")
PAIR_WINDOW = 900.0       # s: pairs of clusters further apart are not scored (one interaction)
BURST_GAP = 1.5           # s: a gap up to this is inside one burst


def _major(c: dict):
    e = sorted((c.get("catalog") or {}).items(), key=lambda kv: (-kv[1], kv[0]))
    return e[0][0] if e else None


def pair_table(day_dir: Path, window: float = PAIR_WINDOW) -> list[dict]:
    """One row per pair of safe clusters whose gap is at most `window` s. Per row: the time (gap, spans, sizes), the
    continuity between the earlier cluster's last frame and the later one's first (safe_clusters.pair_features:
    position, size, exposure, range, rotation, colour), the identity-embedding cosines over all instance pairs per
    part (app_*), the window scores (win_sal_* from the where-to-look windows, win_all_* from all windows; _max = the
    best instance pair, _rm = the mean of the row and column bests), and key_a / key_b, the instance pair to show.
    cat_a / cat_b / cat (same | diff | ?) come from the catalog when the clustering was made with labels: for
    training and evaluation only, no score reads them."""
    import math
    from .safe_clusters import PARTS, load_embeddings, pair_features
    z = json.loads((day_dir / "safe" / "clusters.json").read_text()); C = {c["id"]: c for c in z["clusters"]}
    P = json.loads((day_dir / "safe" / "photos.json").read_text())
    if z.get("report", {}).get("filter"):                  # the clustering left the non-whale instances out: so do the frame counts here
        drop = set(json.loads((day_dir / "safe" / "not_whale.json").read_text())["drop"])
        for p in P.values():
            p["instances"] = [i for i in p["instances"] if i["key"] not in drop]
    I = {it["key"]: it for p in P.values() for it in p["instances"]}; T = {k: P[k.split("~")[0]]["time"] for k in I}
    emb = load_embeddings(day_dir)
    ws = np.load(day_dir / "safe" / "window_scores.npz"); kid = {str(k): i for i, k in enumerate(ws["keys"])}
    D = {"sal": ws["D_sal"].astype(np.float32), "all": ws["D_all"].astype(np.float32)}
    mem = {c: [k for k in cl["keys"] if k in kid] for c, cl in C.items()}
    Em = {part: {c: ([k for k in cl["keys"] if k in emb.get(part, {})], np.asarray([emb[part][k] for k in cl["keys"] if k in emb.get(part, {})], np.float32)) for c, cl in C.items()} for part in PARTS}
    lab = {c: [I[k]["lab_whale"] for k in cl["keys"] if I[k].get("lab_whale")] for c, cl in C.items()}
    ids = sorted(C); rows = []
    for x, a in enumerate(ids):
        for b in ids[x + 1:]:
            A, B = C[a], C[b]
            if A["start"] > B["start"]:
                A, B = B, A
            g = max(0.0, B["start"] - A["end"])
            if g > window:
                continue
            f = {"a": a, "b": b, "gap": g, "log_gap": math.log1p(g), "same_cam": float(A["camera"] == B["camera"]), "n_a": len(A["keys"]), "n_b": len(B["keys"]),
                 "n_min": min(len(A["keys"]), len(B["keys"])), "span_a": A["end"] - A["start"], "span_b": B["end"] - B["start"], "overlap": float(B["start"] < A["end"])}
            ka = max(A["keys"], key=lambda k: T[k]); kb = min(B["keys"], key=lambda k: T[k])
            pf = pair_features(P[ka.split("~")[0]], I[ka], P[kb.split("~")[0]], I[kb], 1, {})
            for k in CONT_KEYS:
                if pf.get(k) is not None:
                    f[k] = pf[k]
            best, n, show = None, 0, None
            for part in PARTS:
                (ka_, Ea), (kb_, Eb) = Em[part][A["id"]], Em[part][B["id"]]
                if len(ka_) and len(kb_):
                    S = Ea @ Eb.T; v = np.sort(S.ravel())[::-1]
                    f[f"app_{part}_max"] = float(v[0]); f[f"app_{part}_mean"] = float(v.mean()); f[f"app_{part}_top3"] = float(v[:3].mean()); f[f"app_{part}_med"] = float(np.median(v))
                    f[f"app_{part}_n"] = float(len(v)); f[f"app_{part}_rowmax"] = float(0.5 * (S.max(1).mean() + S.max(0).mean())); n += 1
                    if best is None or v[0] > best:
                        best = float(v[0]); j = int(np.argmax(S)); show = (ka_[j // S.shape[1]], kb_[j % S.shape[1]], part)
            if best is not None:
                f["app_best"] = best
            f["app_n"] = float(n)
            if lab[A["id"]] and lab[B["id"]]:
                f["lab_mean_d"] = float(np.linalg.norm(np.mean(lab[A["id"]], 0) - np.mean(lab[B["id"]], 0)))
            ma, mb = mem[A["id"]], mem[B["id"]]
            for nm in ("sal", "all"):
                if ma and mb:
                    blk = D[nm][np.ix_([kid[k] for k in ma], [kid[k] for k in mb])]; fin = np.isfinite(blk)
                    if fin.any():
                        r = np.where(fin, blk, -np.inf); r1 = r.max(1); r0 = r.max(0)
                        f[f"win_{nm}_max"] = float(r.max()); f[f"win_{nm}_rm"] = float(0.5 * (r1[np.isfinite(r1)].mean() + r0[np.isfinite(r0)].mean()))
                        if nm == "sal":
                            j = int(np.argmax(r)); f["win_keys"] = (ma[j // r.shape[1]], mb[j % r.shape[1]])
            # the instance pair to show: the best window match, else the best identity cosine, else the two frames closest in time
            ka2, kb2 = f.pop("win_keys", None) or (show[:2] if show else (ka, kb))
            f["key_a"], f["key_b"] = (ka2, kb2) if A["id"] == a else (kb2, ka2); f["app_part"] = show[2] if show else None
            ca, cb = _major(A), _major(B); f["cat_a"], f["cat_b"] = (ca, cb) if A["id"] == a else (cb, ca); f["cat"] = "?" if ca is None or cb is None else "same" if ca == cb else "diff"
            rows.append(f)
    return rows


def nrank(v: np.ndarray) -> np.ndarray:
    """Ranks scaled to (0, 1], NaN stays NaN."""
    from scipy.stats import rankdata
    out = np.full(len(v), np.nan); f = np.isfinite(v)
    if f.any():
        out[f] = rankdata(v[f]) / f.sum()
    return out


def rank_mean(rows: list[dict], names=("app_fin_max", "app_body_max", "win_sal_max")) -> np.ndarray:
    """The untrained combination validated on the val days: the mean of the within-day ranks of the student's fin and
    body cosines and the where-to-look window score, over the scores a pair has (0 when it has none)."""
    M = np.stack([nrank(np.array([r.get(nm, np.nan) if r.get(nm) is not None else np.nan for r in rows], float)) for nm in names], 1)
    n = np.isfinite(M).sum(1); s = np.nansum(M, 1) / np.maximum(n, 1); s[n == 0] = 0.0
    return s


def linker_p(model: dict, rows: list[dict]) -> np.ndarray:
    """P(same whale) of the short-gap linker (experiments/stage2_linker/linker.py: gradient-boosted trees on the pair
    table, fitted on the catalog labels of the train days) for the rows inside its band; NaN outside it."""
    lo, hi = model["band"]; g = np.array([r["gap"] for r in rows], float)
    M = np.stack([np.array([r.get(nm, np.nan) if r.get(nm) is not None else np.nan for r in rows], float) for nm in model["names"]], 1)
    if model.get("rk"):                                    # the appearance and window columns as ranks among the day's pairs beyond a burst
        sel = g > lo
        for j, nm in enumerate(model["names"]):
            if nm.startswith(("app_", "win_")) and not nm.endswith("_n"):
                v = np.full(len(rows), np.nan); v[sel] = nrank(M[sel, j]); M[:, j] = v
    p = np.mean([c.predict_proba(M)[:, 1] for c in model["models"]], 0)
    p[(g <= lo) | (g > hi)] = np.nan
    return p


def revision(report: dict) -> str:
    """The name of a clustering, as the review pages write it into their records (Whale-Review-Web pairs._revision)."""
    base = f"{Path(str(report.get('model', 'model'))).stem}|P{report.get('threshold')}|cost{report.get('cost')}|{'strict' if report.get('strict') else 'plain'}"
    return base + (f"|{report['filter'].get('name')}" if report.get("filter") else "")


def cmd_pairs(a) -> None:
    """The candidate links of the day for the reviewer, scored: every pair of clusters inside one burst (gap <= 1.5 s),
    each cluster's best partners up to a minute away by the short-gap linker (when its model exists) and its best
    partners up to 15 min away by the rank mean. The review page orders its queue with this file."""
    day_dir = Path(a.day_dir); t = time.time(); rows = pair_table(day_dir); z = json.loads((day_dir / "safe" / "clusters.json").read_text())
    g = np.array([r["gap"] for r in rows], float); far = g > BURST_GAP
    rm = np.full(len(rows), np.nan); rm[far] = rank_mean([r for r, f in zip(rows, far) if f])
    loo = LINKER.with_name("link_gbt_loo") / f"{day_dir.name}.pkl"       # a day the linker was fitted on gets the fit that left it out
    model = None if a.no_linker else pickle.load(open(loo, "rb")) if loo.exists() else pickle.load(open(LINKER, "rb")) if LINKER.exists() else None
    lp = linker_p(model, rows) if model else np.full(len(rows), np.nan); hi = model["band"][1] if model else 60.0
    keep = set(np.where(~far)[0].tolist())
    for tier, sel, score, top in (("short", far & (g <= hi), np.where(np.isfinite(lp), lp, rm), a.top_short), ("far", g > hi, rm, a.top_far)):
        per = defaultdict(list)
        for i in np.where(sel)[0]:
            per[rows[i]["a"]].append((score[i], i)); per[rows[i]["b"]].append((score[i], i))
        for lst in per.values():
            keep.update(i for _, i in sorted(lst, key=lambda t: -t[0])[:top])
    out = []
    for i in sorted(keep):
        r = rows[i]; f = lambda v, n=4: None if v is None or not np.isfinite(v) else round(float(v), n)
        out.append({"a": r["a"], "b": r["b"], "gap": round(r["gap"], 1), "tier": "within a burst" if not far[i] else "short gap" if g[i] <= hi else "across a gap",
                    "rm": f(rm[i]), "lp": f(lp[i]), "app": f(r.get("app_best")), "app_part": r.get("app_part"), "win": f(r.get("win_sal_max")), "key_a": r["key_a"], "key_b": r["key_b"]})
    doc = {"revision": revision(z.get("report", {})), "clusters": len(z["clusters"]), "burst_gap": BURST_GAP, "short_gap": hi, "window": PAIR_WINDOW,
           "rank": "rank mean of the student's fin and body cosines and the where-to-look window score (ranks among the day's pairs beyond a burst)",
           "linker": None if model is None else {"name": model.get("name"), "kind": model.get("kind"), "band": list(model["band"]), "held_out": model.get("held_out")}, "pairs": out}
    tmp = day_dir / "safe" / "link_scores.tmp.json"; tmp.write_text(json.dumps(doc)); tmp.replace(day_dir / "safe" / "link_scores.json")
    tiers = {k: sum(1 for o in out if o["tier"] == k) for k in ("within a burst", "short gap", "across a gap")}
    print(f"{day_dir.name}: {len(rows)} pairs within {PAIR_WINDOW:.0f} s -> {len(out)} candidate links {tiers}, linker {'on' if model else 'off'} ({time.time() - t:.0f} s)", flush=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("windows"); s.add_argument("--day-dir", required=True); s.add_argument("--fit-whitening", action="store_true"); s.set_defaults(fn=cmd_windows)
    s = sub.add_parser("instances"); s.add_argument("--day-dir", required=True); s.set_defaults(fn=cmd_instances)
    s = sub.add_parser("not-whale"); s.add_argument("--day-dir", required=True); s.set_defaults(fn=cmd_not_whale)
    s = sub.add_parser("look"); s.add_argument("--ref-dir", required=True); s.add_argument("--boxes", required=True); s.set_defaults(fn=cmd_look)
    s = sub.add_parser("scores"); s.add_argument("--day-dir", required=True); s.add_argument("--saliency", default=None); s.set_defaults(fn=cmd_scores)
    s = sub.add_parser("pairs"); s.add_argument("--day-dir", required=True); s.add_argument("--top-short", type=int, default=4); s.add_argument("--top-far", type=int, default=5)
    s.add_argument("--no-linker", action="store_true"); s.set_defaults(fn=cmd_pairs)
    a = p.parse_args(argv); a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
