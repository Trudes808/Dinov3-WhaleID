"""Safe clusters: the photographer's run of shots of one whale, linked with high precision from the capture sequence
and the frames themselves: capture times, camera settings, the whale's pose, range, position and size in the frame,
its colour and lighting, and the identity embedding of the parts both frames show. The clusters are meant to be
nearly pure; the later linking stage (fins, flukes, scars) joins them into animals.

    python -m model_fine_tuning.cluster.safe_clusters features --day-dir DAY                      -> DAY/safe/photos.json
    python -m model_fine_tuning.cluster.safe_clusters fit      --root ROOT --days days.json --labels L --out experiments/safe_clusters/M.pkl
                                                              [--split train] [--kind gbt|logit] [--feature-set meta|app]
    python -m model_fine_tuning.cluster.safe_clusters cluster  --day-dir DAY --model M.pkl [--labels L] [--threshold T]   -> DAY/safe/clusters.json
    python -m model_fine_tuning.cluster.safe_clusters report   --root ROOT --days days.json --labels L --model M.pkl [--threshold T] [--baselines]
    python -m model_fine_tuning.cluster.safe_clusters sweep    --root ROOT --days days.json --labels L --model M.pkl [--thresholds 0.5,0.7,0.9,0.95,0.98]

Nodes are whale instances (cluster/parts.py --instances: every mask component of a photo, ~0 the largest, the one the
catalog's single label describes). Candidate links: instance pairs across photos of one camera at most `skip` positions
apart in the capture order and at most `window` seconds apart. Pair features: the time gap, the position shift and size
ratio of the instance in the frame (and their rates), the change of focal length and metered exposure (EV from shutter /
aperture / ISO), of the whale's estimated range (focal length over pixels-per-metre), of its orientation (the canon
model's direct rotation, head side), of its colour and lighting (Lab of the whale pixels and of the water around it,
lit-side asymmetry), the parts both show and the cosine of their identity embeddings (cluster_v3/embeddings.npz).
A classifier (gradient-boosted trees or logistic regression) fitted on labelled main-instance pairs of the train days
gives P(same whale); links are accepted in descending P above the threshold, one instance per photo per cluster
(two instances of one photo are two whales), by union-find. A photo's cluster is its main instance's.
Evaluation against the catalog: purity (photos in impure clusters, pair precision), completeness (clusters per animal,
the sequence-only oracle) overall and on single-whale photos, plus the burst / tracklet baselines.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import pickle
import struct
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

from .bursts import make_bursts
from .tracklets import load_instances, make_tracklets

FRAME_W = 2560.0
EXIF_TAGS = ("ExposureTime", "FNumber", "ISOSpeedRatings", "ExposureBiasValue", "FocalLength", "LensModel", "Model",
             "BodySerialNumber", "DateTimeOriginal", "SubSecTimeOriginal", "ExposureProgram", "MeteringMode", "Orientation")
PARTS = ("body", "fin", "fluke")


# --------------------------------------------------------------------------------------------- per-photo features
def _num(s: str | None) -> float | None:
    if s is None:
        return None
    s = str(s).strip()
    try:
        return float(Fraction(s)) if "/" in s else float(s)
    except (ValueError, ZeroDivisionError):
        return None


def read_exif(src: str) -> dict:
    """The camera settings of one RAW file (CR2 / CR3 / others readable by exifread), by tag name."""
    import exifread

    tags = {}
    try:
        if src.lower().endswith(".cr3"):
            data = open(src, "rb").read(4_000_000)
            for box in (b"CMT1", b"CMT2"):
                i = data.find(box)
                if i >= 0:
                    size = struct.unpack(">I", data[i - 4:i])[0]
                    tags.update({k: str(v) for k, v in exifread.process_file(io.BytesIO(data[i + 4:i + size]), details=False).items()})
        else:
            with open(src, "rb") as fh:
                tags = {k: str(v) for k, v in exifread.process_file(fh, details=False).items()}
    except Exception:
        return {}
    out = {}
    for name in EXIF_TAGS:
        for k, v in tags.items():
            if k.endswith(" " + name):
                out[name] = v
                break
    return out


def exposure_value(ex: dict) -> float | None:
    """Scene brightness the camera metered: log2(N^2 / t) - log2(ISO / 100) (auto exposure keeps it in step with the scene)."""
    t, n, iso = _num(ex.get("ExposureTime")), _num(ex.get("FNumber")), _num(ex.get("ISOSpeedRatings"))
    if not t or not n or not iso:
        return None
    return math.log2(n * n / t) - math.log2(iso / 100.0)


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    c = rgb.astype(np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    return np.stack([116.0 * f[..., 1] - 16.0, 500.0 * (f[..., 0] - f[..., 1]), 200.0 * (f[..., 1] - f[..., 2])], -1)


def instance_colour(lab: np.ndarray, mask: np.ndarray, bbox: list | None, scale: int) -> dict:
    """Lab mean of the whale pixels inside the instance box, of the water ring around it, the lit-side asymmetry
    (left vs right half luminance), the luminance spread and the whale / water contrast."""
    from scipy import ndimage

    m = mask
    if bbox:
        sel = np.zeros_like(m); x0, y0, x1, y1 = [int(round(v / scale)) for v in bbox]
        pad = int(0.05 * max(x1 - x0, y1 - y0)) + 2
        sel[max(y0 - pad, 0): y1 + pad + 1, max(x0 - pad, 0): x1 + pad + 1] = True
        m = m & sel
    if m.sum() < 20:
        return {}
    ys, xs = np.nonzero(m)
    r = max(int(0.15 * max(np.ptp(xs), np.ptp(ys))), 4)
    ring = ndimage.binary_dilation(m, iterations=r) & ~ndimage.binary_dilation(mask, iterations=2)
    whale = lab[m]; water = lab[ring] if ring.sum() >= 20 else lab[~mask]
    xm = (xs.min() + xs.max()) / 2.0
    left, right = lab[ys[xs <= xm], xs[xs <= xm], 0], lab[ys[xs > xm], xs[xs > xm], 0]
    asym = (left.mean() - right.mean()) / max(left.mean() + right.mean(), 1e-6) if len(left) and len(right) else 0.0
    return {"lab_whale": [round(float(v), 2) for v in whale.mean(0)], "lab_water": [round(float(v), 2) for v in water.mean(0)],
            "lum_asym": round(float(asym), 4), "lum_std": round(float(whale[:, 0].std()), 2),
            "contrast": round(float(whale[:, 0].mean() - water[:, 0].mean()), 2)}


def photo_colour(image_path: Path, mask_path: Path, instances: list[dict], scale: int = 4) -> list[dict]:
    from PIL import Image

    im = Image.open(image_path).convert("RGB"); W, H = im.size
    lab = _srgb_to_lab(np.asarray(im.resize((W // scale, H // scale), Image.BILINEAR)))
    mask = np.asarray(Image.open(mask_path).convert("L").resize((W // scale, H // scale), Image.NEAREST)) > 0
    return [instance_colour(lab, mask, i.get("bbox_2560"), scale) for i in instances]


def photo_features(day_dir: Path, workers: int = 8) -> dict:
    """stem -> features; written to DAY/safe/photos.json. Sources: timestamps.json, images/index.jsonl (RAW -> EXIF,
    cached in safe/exif.json), canon/points.jsonl (pose, range, side), parts_hr/index.jsonl (instances), the canon masks
    and 2560-px JPEGs (colour per instance)."""
    from concurrent.futures import ThreadPoolExecutor

    ts = json.loads((day_dir / "timestamps.json").read_text())
    index = {Path(json.loads(l)["image"]).stem: json.loads(l) for l in open(day_dir / "images" / "index.jsonl")}
    points = {}
    pf = day_dir / "canon" / "points.jsonl"
    if pf.exists():
        for l in open(pf):
            r = json.loads(l); points[Path(r["image"]).stem] = r
    inst = load_instances(day_dir, "parts_hr") if (day_dir / "parts_hr" / "index.jsonl").exists() else {}
    stems = sorted(ts)
    (day_dir / "safe").mkdir(exist_ok=True)
    exif_cache = day_dir / "safe" / "exif.json"
    exif = json.loads(exif_cache.read_text()) if exif_cache.exists() else {}
    todo = [s for s in stems if s not in exif and s in index]
    if todo:
        with ThreadPoolExecutor(workers) as pool:
            for s, ex in zip(todo, pool.map(lambda s: read_exif(index[s]["source"]), todo)):
                exif[s] = ex
        exif_cache.write_text(json.dumps(exif))

    def one(stem: str) -> tuple[str, dict]:
        t = ts[stem]; ex = exif.get(stem, {}); p = points.get(stem, {}); ins = inst.get(stem, [])
        f = {"time": t.get("time"), "camera": f"{t.get('folder', '?')}|{t.get('camera', '?')}",
             "focal_mm": _num(ex.get("FocalLength")) or _num(t.get("focal_mm")), "ev": exposure_value(ex), "ev_bias": _num(ex.get("ExposureBiasValue")),
             "exposure_time": _num(ex.get("ExposureTime")), "fnumber": _num(ex.get("FNumber")), "iso": _num(ex.get("ISOSpeedRatings")),
             "lens": ex.get("LensModel"), "whale_pixels": p.get("whale_pixels"), "ppm": p.get("ppm_direct"), "head_side": p.get("head_side"),
             "cue_confidence": p.get("cue_confidence"), "rotation": p.get("rotation_direct"), "status": p.get("status"),
             "instances": [{"key": i["key"], "parts": i["parts"], "bbox": i["bbox_2560"], "area": i["area"]} for i in ins]}
        if f["focal_mm"] and f["ppm"]:
            f["log_range"] = round(math.log(f["focal_mm"] / max(float(f["ppm"]), 1e-6)), 4)      # range up to a constant per camera
        mask = day_dir / "canon" / f"{stem}_mask.png"; img = day_dir / "images" / f"{stem}.jpg"
        if ins and mask.exists() and img.exists():
            try:
                for i, col in zip(f["instances"], photo_colour(img, mask, ins)):
                    i.update(col)
            except Exception:
                pass
        return stem, f

    with ThreadPoolExecutor(workers) as pool:
        feats = dict(pool.map(one, stems))
    (day_dir / "safe" / "photos.json").write_text(json.dumps(feats))
    return feats


# ------------------------------------------------------------------------------------------------ candidate pairs
PAIR_FEATURES = ["dt", "log_dt", "skip", "d_pos", "rate_pos", "d_size", "rate_size", "log_area_min", "d_focal", "d_ev", "d_bias", "d_range",
                 "d_rot", "rate_rot", "d_nose", "side_flip", "d_lab_whale", "d_lab_water", "d_asym", "d_contrast", "d_lum_std",
                 "n_inst_max", "secondary", "parts_shared", "parts_union",
                 "app_body", "app_fin", "app_fluke", "app_best", "app_n"]
APP_FEATURES = ["app_body", "app_fin", "app_fluke", "app_best", "app_n"]


def load_embeddings(day_dir: Path, tag: str = "cluster_v3") -> dict:
    """part -> {instance key: unit embedding} from cluster_v3/embeddings.npz (indices into channels.npz's keys)."""
    f = day_dir / tag / "embeddings.npz"; c = day_dir / tag / "channels.npz"
    out = {}
    if not f.exists() or not c.exists():
        return out
    keys = [str(k) for k in np.load(c, allow_pickle=True)["keys"]]
    z = np.load(f, allow_pickle=True)
    for k in z.files:
        if k.endswith("|idx"):
            base = k[: -len("|idx")]; part = base.split("|")[-1]
            idx = z[k]; E = z[base + "|E"].astype(np.float32)
            E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-6)
            out.setdefault(part, {}).update({keys[int(i)]: E[j] for j, i in enumerate(idx) if int(i) < len(keys)})
    return out


def _de(a, b):
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float))) if a is not None and b is not None else None


def pair_features(fa: dict, ia: dict, fb: dict, ib: dict, skip: int, emb: dict) -> dict:
    """Features of the link between instance ia of photo fa and instance ib of photo fb."""
    dt = abs(float(fb["time"]) - float(fa["time"])); dts = max(dt, 0.2)
    f = {"dt": round(dt, 3), "log_dt": round(math.log1p(dt), 4), "skip": skip}
    if ia.get("bbox") and ib.get("bbox"):
        ca = np.array([(ia["bbox"][0] + ia["bbox"][2]) / 2, (ia["bbox"][1] + ia["bbox"][3]) / 2])
        cb = np.array([(ib["bbox"][0] + ib["bbox"][2]) / 2, (ib["bbox"][1] + ib["bbox"][3]) / 2])
        f["d_pos"] = round(float(np.linalg.norm(ca - cb) / FRAME_W), 4); f["rate_pos"] = round(f["d_pos"] / dts, 4)
    if ia.get("area") and ib.get("area"):
        f["d_size"] = round(abs(math.log(ia["area"] / ib["area"])), 4); f["rate_size"] = round(f["d_size"] / dts, 4)
        f["log_area_min"] = round(math.log(min(ia["area"], ib["area"])), 3)
    if fa.get("focal_mm") and fb.get("focal_mm"):
        f["d_focal"] = round(abs(math.log(fa["focal_mm"] / fb["focal_mm"])), 4)
    if fa.get("ev") is not None and fb.get("ev") is not None:
        f["d_ev"] = round(abs(fa["ev"] - fb["ev"]), 4)
    if fa.get("ev_bias") is not None and fb.get("ev_bias") is not None:
        f["d_bias"] = round(abs(fa["ev_bias"] - fb["ev_bias"]), 4)
    if fa.get("log_range") is not None and fb.get("log_range") is not None:
        f["d_range"] = round(abs(fa["log_range"] - fb["log_range"]), 4)
    if fa.get("rotation") is not None and fb.get("rotation") is not None:
        Ra, Rb = np.asarray(fa["rotation"], float), np.asarray(fb["rotation"], float)
        c = (np.trace(Ra @ Rb.T) - 1.0) / 2.0
        f["d_rot"] = round(float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))), 2); f["rate_rot"] = round(f["d_rot"] / dts, 3)
        f["d_nose"] = round(float(np.degrees(np.arccos(np.clip(np.dot(Ra[0], Rb[0]), -1.0, 1.0)))), 2)
    if fa.get("head_side") in (-1, 1) and fb.get("head_side") in (-1, 1):
        conf = min(fa.get("cue_confidence") or 0.0, fb.get("cue_confidence") or 0.0)
        f["side_flip"] = round(float(fa["head_side"] != fb["head_side"]) * conf, 3)
    f["d_lab_whale"] = _de(ia.get("lab_whale"), ib.get("lab_whale"))
    f["d_lab_water"] = _de(ia.get("lab_water"), ib.get("lab_water"))
    for name, key in (("d_asym", "lum_asym"), ("d_contrast", "contrast"), ("d_lum_std", "lum_std")):
        if ia.get(key) is not None and ib.get(key) is not None:
            f[name] = round(abs(ia[key] - ib[key]), 4)
    f["n_inst_max"] = float(max(len(fa.get("instances", [])), len(fb.get("instances", []))))
    f["secondary"] = float(not ia["key"].endswith("~0") or not ib["key"].endswith("~0"))
    pa, pb = set(ia.get("parts") or []), set(ib.get("parts") or [])
    f["parts_shared"] = float(len(pa & pb)); f["parts_union"] = float(len(pa | pb))
    best, n = None, 0
    for part in PARTS:
        ea, eb = emb.get(part, {}).get(ia["key"]), emb.get(part, {}).get(ib["key"])
        if ea is not None and eb is not None:
            v = round(float(np.dot(ea, eb)), 4); f[f"app_{part}"] = v; n += 1
            best = v if best is None else max(best, v)
    if best is not None:
        f["app_best"] = best
    f["app_n"] = float(n)
    return f


def candidate_pairs(feats: dict, window: float = 300.0, skip: int = 3, emb: dict | None = None, labels: dict | None = None) -> list[dict]:
    """[{a, b, ka, kb, main, features..., same}] over instance pairs of same-camera photos at most `skip` apart and
    `window` s apart; `same` is the catalog verdict for labelled main-instance pairs, else None."""
    emb = emb or {}
    by_cam = defaultdict(list)
    for stem, f in feats.items():
        if f.get("time") is not None and f.get("instances"):
            by_cam[f["camera"]].append((float(f["time"]), stem))
    out = []
    for cam, items in by_cam.items():
        items.sort()
        for i in range(len(items)):
            for k in range(1, skip + 1):
                if i + k >= len(items):
                    break
                ta, a = items[i]; tb, b = items[i + k]
                if tb - ta > window:
                    break
                for ia in feats[a]["instances"]:
                    for ib in feats[b]["instances"]:
                        row = {"a": a, "b": b, "ka": ia["key"], "kb": ib["key"], "camera": cam, "main": ia["key"].endswith("~0") and ib["key"].endswith("~0")}
                        row.update(pair_features(feats[a], ia, feats[b], ib, k, emb))
                        row["same"] = None
                        if labels is not None and row["main"]:
                            la, lb = labels.get(a.split("__")[-1]), labels.get(b.split("__")[-1])
                            row["same"] = (la == lb) if (la and lb) else None
                        out.append(row)
    return out


def day_labels(labels_path: Path | None, day: str) -> dict | None:
    if labels_path is None:
        return None
    return json.loads(labels_path.read_text())["days"].get(day, {}).get("photos")


# ------------------------------------------------------------------------------------------------------ the model
def _matrix(pairs: list[dict], names: list[str]) -> np.ndarray:
    X = np.full((len(pairs), len(names)), np.nan)
    for i, p in enumerate(pairs):
        for j, n in enumerate(names):
            v = p.get(n)
            if v is not None:
                X[i, j] = float(v)
    return X


def fit_model(pairs: list[dict], names: list[str], kind: str = "gbt", C: float = 1.0, seed: int = 0) -> dict:
    rows = [p for p in pairs if p.get("same") is not None]
    X = _matrix(rows, names); y = np.array([bool(p["same"]) for p in rows])
    model = {"kind": kind, "features": names, "n_pairs": int(len(rows)), "n_same": int(y.sum())}
    if kind == "gbt":
        from sklearn.ensemble import HistGradientBoostingClassifier

        clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=400, l2_regularization=1.0, min_samples_leaf=40,
                                             early_stopping=False, random_state=seed)
        clf.fit(X, y)
        model["clf"] = clf
    else:
        from sklearn.linear_model import LogisticRegression

        mu = np.nanmean(X, axis=0); sd = np.nanstd(X, axis=0); sd[~np.isfinite(sd) | (sd < 1e-6)] = 1.0; mu[~np.isfinite(mu)] = 0.0
        miss = np.isnan(X); D = np.concatenate([np.where(miss, 0.0, (X - mu) / sd), miss.astype(float)], axis=1)
        clf = LogisticRegression(C=C, max_iter=3000); clf.fit(D, y)
        model.update({"clf": clf, "mu": mu, "sd": sd})
    return model


def predict(model: dict, pairs: list[dict]) -> np.ndarray:
    if not pairs:
        return np.zeros(0)
    X = _matrix(pairs, model["features"])
    if model["kind"] == "gbt":
        return model["clf"].predict_proba(X)[:, 1]
    miss = np.isnan(X); D = np.concatenate([np.where(miss, 0.0, (X - model["mu"]) / model["sd"]), miss.astype(float)], axis=1)
    return model["clf"].predict_proba(D)[:, 1]


def importance(model: dict, pairs: list[dict], n_repeats: int = 3, seed: int = 0) -> list[tuple[str, float]]:
    """Permutation importance (drop in AUC) of each feature on labelled pairs."""
    from sklearn.metrics import roc_auc_score

    rows = [p for p in pairs if p.get("same") is not None]
    y = np.array([bool(p["same"]) for p in rows]); base = roc_auc_score(y, predict(model, rows))
    rng = np.random.default_rng(seed); out = []
    for name in model["features"]:
        drops = []
        for _ in range(n_repeats):
            vals = [p.get(name) for p in rows]; perm = rng.permutation(len(rows))
            shuffled = [dict(p, **{name: vals[j]}) for p, j in zip(rows, perm)]
            drops.append(base - roc_auc_score(y, predict(model, shuffled)))
        out.append((name, float(np.mean(drops))))
    return sorted(out, key=lambda t: -t[1])


# ------------------------------------------------------------------------------------------------------ clustering
class UnionFind:
    def __init__(self, items, photo_of):
        self.p = {i: i for i in items}; self.photos = {i: {photo_of[i]} for i in items}

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]; i = self.p[i]
        return i

    def union(self, a, b) -> bool:
        """Merge unless the two clusters already share a photo (two instances of one photo are two whales)."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        if self.photos[ra] & self.photos[rb]:
            return False
        if len(self.photos[ra]) < len(self.photos[rb]):
            ra, rb = rb, ra
        self.p[rb] = ra; self.photos[ra] |= self.photos.pop(rb)
        return True


def chain_clusters(feats: dict, pairs: list[dict], prob: np.ndarray, threshold: float) -> list[dict]:
    """Union-find over the links above the threshold, best first, one instance per photo per cluster."""
    photo_of = {i["key"]: s for s, f in feats.items() if f.get("time") is not None for i in f.get("instances", [])}
    uf = UnionFind(list(photo_of), photo_of)
    order = np.argsort(-prob)
    accepted = []
    for j in order:
        if prob[j] < threshold:
            break
        p = pairs[j]
        if uf.find(p["ka"]) != uf.find(p["kb"]) and uf.union(p["ka"], p["kb"]):
            accepted.append((p["ka"], p["kb"], round(float(prob[j]), 4)))
    groups = defaultdict(list)
    for k in photo_of:
        groups[uf.find(k)].append(k)
    links = defaultdict(list)
    for ka, kb, pr in accepted:
        links[uf.find(ka)].append((ka, kb, pr))
    out = []
    for root, keys in groups.items():
        keys.sort(key=lambda k: float(feats[photo_of[k]]["time"]))
        photos = [photo_of[k] for k in keys]
        out.append({"keys": keys, "photos": photos, "main_photos": [photo_of[k] for k in keys if k.endswith("~0")],
                    "camera": feats[photos[0]]["camera"], "start": float(feats[photos[0]]["time"]), "end": float(feats[photos[-1]]["time"]),
                    "links": links.get(root, [])})
    out.sort(key=lambda c: (c["start"], c["camera"]))
    for k, c in enumerate(out):
        c["id"] = k
    return out


def cluster_metrics(clusters: list[list[str]], labels: dict, feats: dict, window: float | None = None, single_only: bool = False) -> dict:
    """Purity and completeness of photo clusters (each photo listed once, by its main instance) against the catalog's
    one-animal-per-photo labels; labelled photos with a whale only, optionally only photos with a single instance."""
    def ok(s):
        f = feats.get(s, {})
        return bool(f.get("instances")) and (not single_only or len(f["instances"]) == 1)
    lab = {}
    for c in clusters:
        for s in c:
            l = labels.get(s.split("__")[-1])
            if l and ok(s):
                lab[s] = l
    n = len(lab)
    multi = [[s for s in c if s in lab] for c in clusters]; multi = [c for c in multi if len(c) >= 2]
    impure = [c for c in multi if len(set(lab[s] for s in c)) > 1]
    photos_impure = sum(len(c) - Counter(lab[s] for s in c).most_common(1)[0][1] for c in impure)
    tp = sum(sum(v * (v - 1) // 2 for v in Counter(lab[s] for s in c).values()) for c in multi)
    pp = sum(len(c) * (len(c) - 1) // 2 for c in multi)
    per_animal = Counter(lab.values())
    ap = sum(v * (v - 1) // 2 for v in per_animal.values())
    cl_of = {s: k for k, c in enumerate(clusters) for s in c}
    clusters_per_animal = {a: len({cl_of[s] for s in lab if lab[s] == a}) for a in per_animal}
    from sklearn.metrics import adjusted_rand_score
    stems = sorted(lab); ari = float(adjusted_rand_score([lab[s] for s in stems], [cl_of[s] for s in stems])) if stems else float("nan")
    out = {"photos": n, "animals": len(per_animal), "clusters": len({cl_of[s] for s in lab}), "multi_clusters": len(multi),
           "impure_clusters": len(impure), "photos_in_impure_clusters": int(photos_impure), "impure_photo_fraction": round(photos_impure / max(n, 1), 4),
           "pair_precision": round(tp / max(pp, 1), 4), "pair_recall": round(tp / max(ap, 1), 4),
           "clusters_per_animal": round(float(np.mean(list(clusters_per_animal.values()))) if clusters_per_animal else float("nan"), 2), "ari": round(ari, 4)}
    if window is not None:
        by_cam = defaultdict(list)
        for s in lab:
            by_cam[feats[s]["camera"]].append((float(feats[s]["time"]), s))
        parent = {s: s for s in lab}

        def find(s):
            while parent[s] != s:
                parent[s] = parent[parent[s]]; s = parent[s]
            return s
        for items in by_cam.values():
            items.sort(); by_animal = defaultdict(list)
            for t, s in items:
                by_animal[lab[s]].append((t, s))
            for seq in by_animal.values():
                for (t0, s0), (t1, s1) in zip(seq, seq[1:]):
                    if t1 - t0 <= window:
                        parent[find(s1)] = find(s0)
        roots = len({find(s) for s in lab})
        out["oracle_clusters"] = roots; out["oracle_clusters_per_animal"] = round(roots / max(len(per_animal), 1), 2)
    return out


def photo_clusters(clusters: list[dict], feats: dict) -> list[list[str]]:
    """Photo lists by main instance; photos without instances become singletons."""
    out = [c["main_photos"] for c in clusters if c["main_photos"]]
    seen = {s for c in out for s in c}
    out += [[s] for s, f in feats.items() if f.get("time") is not None and s not in seen]
    return out


def baselines(day_dir: Path, feats: dict, labels: dict, window: float) -> dict:
    ts = json.loads((day_dir / "timestamps.json").read_text())
    out = {}
    for gap in (1.0, 1.5, 2.0):
        b = make_bursts(ts, gap)
        out[f"bursts_{gap}s"] = cluster_metrics([x["photos"] for x in b], labels, feats, window)
    inst = load_instances(day_dir, "parts_hr")
    tr = make_tracklets(ts, inst, 1.5, None, 1.0)
    main = [[s for s, k in zip(t["photos"], t["keys"]) if k.endswith("~0")] for t in tr]
    out["tracklets_1.5s"] = cluster_metrics([m for m in main if m], labels, feats, window)
    return out


def load_model(path: Path) -> dict:
    return pickle.loads(path.read_bytes())


def run_day(day_dir: Path, model: dict, threshold: float, labels: dict | None, window: float, skip: int, write: bool = True, pairs: list | None = None) -> dict:
    feats = json.loads((day_dir / "safe" / "photos.json").read_text())
    if pairs is None:
        emb = load_embeddings(day_dir) if any(n in model["features"] for n in APP_FEATURES) else {}
        pairs = candidate_pairs(feats, window, skip, emb, labels)
    prob = predict(model, pairs)
    clusters = chain_clusters(feats, pairs, prob, threshold)
    rep = {"day": day_dir.name, "threshold": threshold, "window": window, "skip": skip, "pairs": len(pairs), "links": sum(len(c["links"]) for c in clusters)}
    if labels:
        pc = photo_clusters(clusters, feats)
        rep["metrics"] = cluster_metrics(pc, labels, feats, window)
        rep["metrics_single"] = cluster_metrics(pc, labels, feats, window, single_only=True)
        y = np.array([p.get("same") for p in pairs], dtype=object); known = np.array([v is not None for v in y])
        if known.sum() and len(set(y[known])) == 2:
            from sklearn.metrics import roc_auc_score
            rep["pair_auc"] = round(float(roc_auc_score(y[known].astype(bool), prob[known])), 4)
        acc = {(ka, kb) for c in clusters for ka, kb, _ in c["links"]}
        rep["wrong_links"] = [(p["a"], p["b"], round(float(pr), 3), {k: p.get(k) for k in ("dt", "skip", "d_pos", "d_size", "d_rot", "d_range", "app_best", "n_inst_max")})
                              for p, pr in zip(pairs, prob) if (p["ka"], p["kb"]) in acc and p.get("same") is False]
    if write:
        (day_dir / "safe").mkdir(exist_ok=True)
        (day_dir / "safe" / "clusters.json").write_text(json.dumps({"report": rep, "clusters": clusters}, indent=1))
    return rep


# ------------------------------------------------------------------------------------------------------------- CLI
def _common(s):
    s.add_argument("--window", type=float, default=300.0); s.add_argument("--skip", type=int, default=3)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("features"); s.add_argument("--day-dir", type=Path, required=True); s.add_argument("--workers", type=int, default=8)
    s = sub.add_parser("fit"); s.add_argument("--root", type=Path, required=True); s.add_argument("--days", type=Path, required=True)
    s.add_argument("--labels", type=Path, required=True); s.add_argument("--out", type=Path, required=True)
    s.add_argument("--split", default="train"); s.add_argument("--kind", default="gbt", choices=["gbt", "logit"])
    s.add_argument("--feature-set", default="app", choices=["meta", "app"]); s.add_argument("--C", type=float, default=1.0); _common(s)
    s.add_argument("--importance-split", default="val", help="split for the permutation importance print (none to skip)")
    s = sub.add_parser("cluster"); s.add_argument("--day-dir", type=Path, required=True); s.add_argument("--model", type=Path, required=True)
    s.add_argument("--labels", type=Path, default=None); s.add_argument("--threshold", type=float, default=0.9); _common(s)
    s = sub.add_parser("report"); s.add_argument("--root", type=Path, required=True); s.add_argument("--days", type=Path, required=True)
    s.add_argument("--labels", type=Path, required=True); s.add_argument("--model", type=Path, required=True)
    s.add_argument("--threshold", type=float, default=0.9); _common(s)
    s.add_argument("--splits", default="val,eval"); s.add_argument("--baselines", action="store_true"); s.add_argument("--out", type=Path, default=None)
    s = sub.add_parser("sweep"); s.add_argument("--root", type=Path, required=True); s.add_argument("--days", type=Path, required=True)
    s.add_argument("--labels", type=Path, required=True); s.add_argument("--model", type=Path, required=True); _common(s)
    s.add_argument("--thresholds", default="0.5,0.6,0.7,0.8,0.9,0.95,0.98,0.99"); s.add_argument("--splits", default="val,eval"); s.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)

    if a.cmd == "features":
        feats = photo_features(a.day_dir, a.workers)
        n_ex = sum(1 for f in feats.values() if f.get("ev") is not None); n_rot = sum(1 for f in feats.values() if f.get("rotation") is not None)
        n_col = sum(1 for f in feats.values() for i in f.get("instances", []) if i.get("lab_whale"))
        print(f"{a.day_dir.name}: {len(feats)} photos, exposure {n_ex}, rotation {n_rot}, instances with colour {n_col}")
        return 0

    def day_pairs(day: str, need_app: bool):
        d = a.root / day; feats = json.loads((d / "safe" / "photos.json").read_text())
        return candidate_pairs(feats, a.window, a.skip, load_embeddings(d) if need_app else {}, day_labels(a.labels, day))

    if a.cmd == "fit":
        days_js = json.loads(a.days.read_text())
        names = [n for n in PAIR_FEATURES if a.feature_set == "app" or n not in APP_FEATURES]
        pairs = []
        for day in days_js[a.split]:
            pairs += day_pairs(day, a.feature_set == "app")
        model = fit_model(pairs, names, a.kind, a.C)
        model.update({"split": a.split, "days": days_js[a.split], "window": a.window, "skip": a.skip, "feature_set": a.feature_set})
        a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_bytes(pickle.dumps(model))
        print(f"{a.kind} fitted on {model['n_pairs']} labelled main-instance pairs ({model['n_same']} same) of {a.split} -> {a.out}")
        if a.importance_split != "none":
            vp = []
            for day in days_js[a.importance_split]:
                vp += day_pairs(day, a.feature_set == "app")
            from sklearn.metrics import roc_auc_score
            rows = [q for q in vp if q.get("same") is not None]
            print(f"  {a.importance_split} pair AUC {roc_auc_score([q['same'] for q in rows], predict(model, rows)):.4f} on {len(rows)} pairs; permutation importance (AUC drop):")
            for name, drop in importance(model, vp)[:16]:
                print(f"    {name:14s} {drop:+.4f}")
        return 0
    if a.cmd == "cluster":
        rep = run_day(a.day_dir, load_model(a.model), a.threshold, day_labels(a.labels, a.day_dir.name), a.window, a.skip)
        print(json.dumps({k: v for k, v in rep.items() if k != "wrong_links"}, indent=1))
        for w in rep.get("wrong_links", [])[:20]:
            print("  wrong:", w)
        return 0
    cols = ("clusters", "impure_clusters", "photos_in_impure_clusters", "pair_precision", "clusters_per_animal", "oracle_clusters_per_animal", "ari")
    if a.cmd == "report":
        model = load_model(a.model); days_js = json.loads(a.days.read_text()); table = {}
        print(f"{'day':11s} {'split':5s} {'auc':6s} " + " ".join(f"{c[:14]:>14s}" for c in cols))
        for split in a.splits.split(","):
            for day in days_js[split]:
                d = a.root / day; labels = day_labels(a.labels, day)
                rep = run_day(d, model, a.threshold, labels, a.window, a.skip)
                row = {"split": split, "pair_auc": rep.get("pair_auc"), "safe": rep["metrics"], "safe_single": rep["metrics_single"], "wrong_links": len(rep.get("wrong_links", []))}
                m = rep["metrics"]; print(f"{day:11s} {split:5s} {str(rep.get('pair_auc')):6s} " + " ".join(f"{str(m.get(c)):>14s}" for c in cols))
                ms = rep["metrics_single"]; print(f"  {'single-whale photos':22s}       " + " ".join(f"{str(ms.get(c)):>14s}" for c in cols))
                if a.baselines:
                    feats = json.loads((d / "safe" / "photos.json").read_text()); row["baselines"] = baselines(d, feats, labels, a.window)
                    for name, b in row["baselines"].items():
                        print(f"  {name:22s}       " + " ".join(f"{str(b.get(c)):>14s}" for c in cols))
                table[day] = row
        if a.out:
            a.out.write_text(json.dumps(table, indent=1))
        return 0
    if a.cmd == "sweep":
        model = load_model(a.model); days_js = json.loads(a.days.read_text()); ths = [float(t) for t in a.thresholds.split(",")]
        need_app = any(n in model["features"] for n in APP_FEATURES); table = {}
        for split in a.splits.split(","):
            cache = {day: (json.loads((a.root / day / "safe" / "photos.json").read_text()), day_pairs(day, need_app), day_labels(a.labels, day)) for day in days_js[split]}
            print(f"== {split}: " + f"{'thr':5s} {'impure_cl':>9s} {'impure_ph':>9s} {'ph_frac':>8s} {'pair_prec':>9s} {'cl/animal':>9s} {'oracle':>7s} {'ari':>6s} | single-whale photos: {'impure_ph':>9s} {'pair_prec':>9s}")
            for th in ths:
                agg = Counter(); n_days = 0; cpa = []; ari = []; prec_tp = prec_pp = 0; s_imp = 0; s_tp = s_pp = 0; orc = []
                for day, (feats, pairs, labels) in cache.items():
                    rep = run_day(a.root / day, model, th, labels, a.window, a.skip, write=False, pairs=pairs)
                    m, ms = rep["metrics"], rep["metrics_single"]; n_days += 1
                    agg["impure_clusters"] += m["impure_clusters"]; agg["photos_in_impure"] += m["photos_in_impure_clusters"]; agg["photos"] += m["photos"]
                    cpa.append(m["clusters_per_animal"]); ari.append(m["ari"]); orc.append(m["oracle_clusters_per_animal"])
                    s_imp += ms["photos_in_impure_clusters"]
                    # pooled pair precision from the per-day values and pair counts is not recoverable; use the mean
                    prec_tp += m["pair_precision"]; s_tp += ms["pair_precision"]
                row = {"impure_clusters": agg["impure_clusters"], "photos_in_impure": agg["photos_in_impure"], "impure_fraction": round(agg["photos_in_impure"] / max(agg["photos"], 1), 4),
                       "pair_precision_mean": round(prec_tp / n_days, 4), "clusters_per_animal_mean": round(float(np.mean(cpa)), 2), "oracle_mean": round(float(np.mean(orc)), 2),
                       "ari_mean": round(float(np.mean(ari)), 4), "single_photos_in_impure": s_imp, "single_pair_precision_mean": round(s_tp / n_days, 4)}
                table[f"{split}@{th}"] = row
                print(f"   {th:5.2f} {row['impure_clusters']:9d} {row['photos_in_impure']:9d} {row['impure_fraction']:8.4f} {row['pair_precision_mean']:9.4f} {row['clusters_per_animal_mean']:9.2f} {row['oracle_mean']:7.2f} {row['ari_mean']:6.3f} | {s_imp:>29d} {row['single_pair_precision_mean']:9.4f}")
        if a.out:
            a.out.write_text(json.dumps(table, indent=1))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
