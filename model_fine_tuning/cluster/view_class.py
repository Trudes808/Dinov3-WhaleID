"""View class of a whale instance: does the photo show its FLUKE, its DORSAL FIN, or neither ("other")?

The researchers tell animals apart by the fluke first and the dorsal fin second, so the Photo Labeling page works
through a day's stacks in that order (flukes, dorsal fins, the rest). This module says which stack is which, straight
from the pixels: two small classifiers (fluke yes / no, dorsal fin yes / no) on frozen DINOv3 features of the instance.
No mesh, no canonical map, no keypoints; "other" = neither (head, blow, a back without the fin, a pectoral fin, a tail
stock, a breach, a whale under water).

    python -m model_fine_tuning.cluster.view_class features --day-dir DAY     -> DAY/safe/view_feat.npz
    python -m model_fine_tuning.cluster.view_class predict  --day-dir DAY     -> DAY/safe/view_class.json
    python -m model_fine_tuning.cluster.view_class fit      [--targets T.json] [--root experiments/days_all] [--out M.pkl]
    python -m model_fine_tuning.cluster.view_class report   [--split eval]    the saved model against the labels of a split

Features (frozen DINOv3 ViT-B/16, last block, final norm; nothing is fine-tuned): the cls token and the mean patch
token of two crops of the instance box (x 1.15) cut from the day's 2560-px image:
  sq  the square crop at 448 px
  ar  a crop that keeps the box's shape (1:1, 3:2, 2:1 or 3:1, either way up; long side up to 672 px), so a long back
      keeps three times the detail it has in a 224-px square
= 4 x 768 numbers per instance, each block scaled to unit length. Head: standardise, PCA to 128, RBF SVM (C 4, balanced
classes), one for the fluke and one for the fin; the decision value becomes a probability through a logistic fitted on
out-of-fold scores of a random sample of instances (so P follows the real mix of photos). An instance shows the part
when P >= 0.5; view_class.json keeps both probabilities per instance key, and the box each was computed for (a reader
skips an instance whose box has changed since: re-run features + predict after a day's instances are rebuilt).

Labels (experiments/view_class, see its README): by-eye labels of random and of hard instances, the keypoints the
researcher clicked on the Pairs / Bursts pages and the catalogue's fluke collection, all on the train + val days; the
eval days are only ever scored. Measured there (LOG 2026-11-12): three-way accuracy on a blind random sample 0.95,
against 0.84 for the chart flags of cluster/parts.py (the synthetic-trained canonical map) that the page used before.
Synthetic shots (exact labels from the render metadata) were tried as extra training data: they help when real labels
are few (+2-3 points at 50-100 labels) and not with the labels there are now, so the model is fitted on real photos.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
MODEL = REPO / "experiments" / "safe_clusters" / "view_class.pkl"
TARGETS = REPO / "experiments" / "view_class" / "targets.json"
SPLITS = REPO / "model_fine_tuning" / "splits" / "days.json"
BUCKETS = [(448, 448), (384, 576), (336, 672), (224, 672), (576, 384), (672, 336)]          # (H, W) of the shape-keeping crop
BLOCKS = ("sq_cls", "sq_mean", "ar_cls", "ar_mean")
PARTS = ("fluke", "dorsal")
THRESHOLD = 0.5


# ---------------------------------------------------------------------------------------------------------- features
def _cut(im, cx, cy, w, h, W, H, mean, std):
    from PIL import Image
    c = im.crop((int(round(cx - w / 2)), int(round(cy - h / 2)), int(round(cx + w / 2)), int(round(cy + h / 2))))        # outside the frame = black
    return ((np.asarray(c.resize((W, H), Image.BICUBIC if w < W else Image.LANCZOS), np.float32) - mean) / std).transpose(2, 0, 1)


def shaped(box) -> tuple[int, float, float]:
    """The shape-keeping crop of a box: bucket index and the crop's width / height in image pixels."""
    x0, y0, x1, y1 = box; w, h = max((x1 - x0) * 1.15, 64.0), max((y1 - y0) * 1.15, 64.0); a = np.log(w / h)
    k = int(np.argmin([abs(a - np.log(W / H)) for H, W in BUCKETS])); H, W = BUCKETS[k]
    if w / h < W / H:
        w = h * W / H
    else:
        h = w * H / W
    return k, w, h


def extract(image_of, boxes: list, workers: int = 8) -> np.ndarray:
    """(n, 4 x 768) float16 for boxes [(image id, (x0, y0, x1, y1))]; image_of(image id) -> path. Rows follow boxes."""
    import torch
    from PIL import Image
    from ..common.backbone import IMAGENET_MEAN, IMAGENET_STD, load_backbone
    mean = np.array(IMAGENET_MEAN, np.float32) * 255; std = np.array(IMAGENET_STD, np.float32) * 255
    by = defaultdict(list)
    for i, (img, b) in enumerate(boxes):
        by[img].append((i, b))
    imgs = sorted(by)

    class Photos(torch.utils.data.Dataset):
        def __len__(self):
            return len(imgs)

        def __getitem__(self, j):
            im = Image.open(image_of(imgs[j])).convert("RGB"); out = []
            for i, b in by[imgs[j]]:
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2; s = max(max(b[2] - b[0], b[3] - b[1]) * 1.15, 64.0); k, w, h = shaped(b)
                out.append((i, _cut(im, cx, cy, s, s, 448, 448, mean, std), k, _cut(im, cx, cy, w, h, BUCKETS[k][1], BUCKETS[k][0], mean, std)))
            return out

    model = load_backbone("vitb16").cuda().eval().half(); n = len(boxes); X = np.zeros((n, 4 * 768), np.float16); pend = defaultdict(list)

    def flush(k, force=False):
        while pend[k] and (force or len(pend[k]) >= 48):
            chunk, pend[k] = pend[k][:48], pend[k][48:]
            x = torch.from_numpy(np.stack([c[1] for c in chunk])).cuda().half(); idx = [c[0] for c in chunk]; o = 0 if k == "sq" else 2 * 768
            with torch.no_grad():
                pt, cl = model.get_intermediate_layers(x, n=[11], reshape=False, return_class_token=True, norm=True)[0]
            X[idx, o:o + 768] = cl.float().cpu().numpy(); X[idx, o + 768:o + 2 * 768] = pt.float().mean(1).cpu().numpy()

    dl = torch.utils.data.DataLoader(Photos(), batch_size=8, num_workers=workers, collate_fn=lambda b: [x for l in b for x in l])
    for batch in dl:
        for i, sq, k, ar in batch:
            pend["sq"].append((i, sq)); pend[k].append((i, ar))
        for k in list(pend):
            flush(k)
    for k in list(pend):
        flush(k, True)
    return X


def day_boxes(day_dir: Path) -> tuple[list, list]:
    P = json.loads((day_dir / "safe" / "photos.json").read_text()); keys, boxes = [], []
    for stem, p in P.items():
        for it in p["instances"]:
            keys.append(it["key"]); boxes.append((stem, it["bbox"]))
    return keys, boxes


def cmd_features(a) -> None:
    day_dir = Path(a.day_dir); t = time.time(); keys, boxes = day_boxes(day_dir)
    X = extract(lambda stem: day_dir / "images" / f"{stem}.jpg", boxes) if keys else np.zeros((0, 4 * 768), np.float16)
    tmp = day_dir / "safe" / "view_feat.tmp.npz"; np.savez(tmp, keys=np.array(keys), X=X, blocks=np.array(BLOCKS), box=np.array([b for _, b in boxes], np.float32).reshape(-1, 4))
    tmp.replace(day_dir / "safe" / "view_feat.npz")                                    # box = what the features were cut from: a reader can tell a stale file
    print(f"{day_dir.name}: {len(keys)} instances in {time.time() - t:.0f} s", flush=True)


def unit_blocks(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32).reshape(len(X), 4, 768)
    return (X / np.maximum(np.linalg.norm(X, axis=2, keepdims=True), 1e-6)).reshape(len(X), -1)


def load_feat(day_dir: Path) -> tuple[list, np.ndarray]:
    z = np.load(day_dir / "safe" / "view_feat.npz"); return [str(k) for k in z["keys"]], unit_blocks(z["X"])


# ------------------------------------------------------------------------------------------------------------- model
def _fit_one(X: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    sc = StandardScaler().fit(X); p = PCA(min(128, len(X) - 1), random_state=0).fit(sc.transform(X))
    return {"scaler": sc, "pca": p, "svm": SVC(C=4.0, kernel="rbf", gamma="scale", class_weight="balanced").fit(p.transform(sc.transform(X)), y)}


def _decide(m: dict, X: np.ndarray) -> np.ndarray:
    return m["svm"].decision_function(m["pca"].transform(m["scaler"].transform(X))) if len(X) else np.zeros(0)


def probability(model: dict, X: np.ndarray) -> np.ndarray:
    """(n, 2) P(shows the fluke), P(shows the dorsal fin)."""
    out = np.zeros((len(X), 2), np.float32)
    for j, part in enumerate(PARTS):
        a, b = model["platt"][part]; out[:, j] = 1.0 / (1.0 + np.exp(-(a * _decide(model[part], X) + b)))
    return out


def load_model(path: Path = MODEL) -> dict:
    return pickle.load(open(path, "rb"))


def _labelled(root: Path, targets: dict, days: list) -> tuple:
    """Features and targets of the labelled instances of these days: X, Y (n, 2; 1 / 0 / -1 marginal / -9 unknown), day, random flag."""
    X, Y, D, R = [], [], [], []
    for d in days:
        f = root / d / "safe" / "view_feat.npz"
        want = {k: v for k, v in targets.items() if v["day"] == d}
        if not want or not f.exists():
            continue
        keys, Xd = load_feat(root / d); kid = {k: i for i, k in enumerate(keys)}
        for k, v in want.items():
            if k in kid:
                X.append(Xd[kid[k]]); Y.append(v["y"]); D.append(d); R.append(v.get("src", {}).get("round") in ("r0", "ev"))
    return np.array(X, np.float32), np.array(Y, np.int8), np.array(D), np.array(R, bool)


def confusion(Y: np.ndarray, P: np.ndarray, thr: float = THRESHOLD) -> tuple[np.ndarray, int]:
    """Three classes over the instances with both parts clearly labelled: rows true fluke / dorsal / other, columns predicted."""
    m = (Y >= 0).all(1); true = np.where(Y[m, 0] == 1, 0, np.where(Y[m, 1] == 1, 1, 2)); pred = np.where(P[m, 0] >= thr, 0, np.where(P[m, 1] >= thr, 1, 2)); cm = np.zeros((3, 3), int)
    for t, p in zip(true, pred):
        cm[t, p] += 1
    return cm, int(m.sum())


def _lines(Y, P, title: str) -> list[str]:
    from sklearn.metrics import roc_auc_score
    out = [title]
    for j, part in enumerate(PARTS):
        m = Y[:, j] >= 0; y = Y[m, j]; s = P[m, j]
        if y.min() == y.max():
            continue
        row = f"   {part:6s} AUC {roc_auc_score(y, s):.4f}"
        for thr in (0.5, 0.9):
            hit = s >= thr; tp = int(((y == 1) & hit).sum()); row += f" | P >= {thr}: precision {tp / max(hit.sum(), 1):.3f} ({tp}/{int(hit.sum())}) recall {tp / max((y == 1).sum(), 1):.3f} ({tp}/{int((y == 1).sum())})"
        out.append(row)
    cm, n = confusion(Y, P)
    if n:
        out.append(f"   three classes ({n} clear cases; rows true fluke / dorsal / other, columns predicted): {cm.tolist()} accuracy {np.trace(cm) / n:.3f}")
    return out


def cmd_fit(a) -> None:
    from sklearn.linear_model import LogisticRegression
    targets = json.loads(Path(a.targets).read_text()); split = json.loads(SPLITS.read_text()); root = Path(a.root)
    days = sorted({v["day"] for v in targets.values()} - set(split["eval"]))
    X, Y, D, R = _labelled(root, targets, days)
    print(f"{len(X)} labelled instances on {len(set(D))} days (train + val); fluke {int((Y[:, 0] == 1).sum())} yes / {int((Y[:, 0] == 0).sum())} no; dorsal {int((Y[:, 1] == 1).sum())} / {int((Y[:, 1] == 0).sum())}", flush=True)
    fd = np.array(sorted(set(D))); np.random.default_rng(0).shuffle(fd); S = np.full((len(X), 2), np.nan, np.float32)
    for f in np.array_split(fd, a.folds):                  # out-of-fold decision values: a day is scored by classifiers that never saw it
        te = np.isin(D, f)
        for j in range(2):
            m = ~te & (Y[:, j] >= 0); S[te, j] = _decide(_fit_one(X[m], Y[m, j]), X[te])
    model = {"name": a.name, "blocks": list(BLOCKS), "threshold": THRESHOLD, "labels": int(len(X)), "days": [str(d) for d in sorted(set(D))], "platt": {}, "fitted": time.strftime("%Y-%m-%d")}
    P = np.zeros_like(S)
    for j, part in enumerate(PARTS):
        m = R & (Y[:, j] >= 0)                              # the random sample: its mix is the mix of the photos
        lr = LogisticRegression(C=100.0, max_iter=1000).fit(S[m, j:j + 1], Y[m, j]); model["platt"][part] = (float(lr.coef_[0, 0]), float(lr.intercept_[0]))
        P[:, j] = 1.0 / (1.0 + np.exp(-(model["platt"][part][0] * S[:, j] + model["platt"][part][1])))
        m = Y[:, j] >= 0; model[part] = _fit_one(X[m], Y[m, j])
    rep = _lines(Y[R], P[R], "day-grouped cross-validation, the random sample (population rates)") + _lines(Y, P, "day-grouped cross-validation, all labels (hard cases over-represented)")
    print("\n".join(rep)); model["cv"] = rep
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); pickle.dump(model, open(a.out, "wb")); print(f"saved {a.out}; platt {model['platt']}")


def cmd_report(a) -> None:
    targets = json.loads(Path(a.targets).read_text()); split = json.loads(SPLITS.read_text()); model = load_model(Path(a.model))
    days = sorted(split[a.split]); X, Y, D, R = _labelled(Path(a.root), targets, days); P = probability(model, X)
    print(f"{model['name']} (fitted {model['fitted']} on {model['labels']} labels of {len(model['days'])} days) on the {a.split} days: {len(X)} labelled instances")
    print("\n".join(_lines(Y[R], P[R], "the random sample, labelled blind") + _lines(Y[~R], P[~R], "the other sources (researcher clicks, catalogue flukes)")))
    for d in sorted(set(D)):
        cm, n = confusion(Y[R & (D == d)], P[R & (D == d)]); print(f"   {d}: random sample {np.trace(cm)} of {n} right")


def cmd_predict(a) -> None:
    day_dir = Path(a.day_dir); model = load_model(Path(a.model)); keys, X = load_feat(day_dir); P = probability(model, X); thr = model["threshold"]
    views = {k: [round(float(p[0]), 3), round(float(p[1]), 3)] for k, p in zip(keys, P)}
    z = np.load(day_dir / "safe" / "view_feat.npz"); box = {k: [round(float(v), 1) for v in b] for k, b in zip(keys, z["box"])} if "box" in z.files else {}
    counts = {"fluke": int((P[:, 0] >= thr).sum()), "dorsal": int(((P[:, 1] >= thr) & (P[:, 0] < thr)).sum()), "other": int(((P < thr).all(1)).sum())}
    out = day_dir / "safe" / a.out_name; tmp = out.with_suffix(".tmp.json")
    tmp.write_text(json.dumps({"model": model["name"], "fitted": model["fitted"], "threshold": thr, "order": list(PARTS), "counts": counts, "views": views, "box": box}, indent=0)); tmp.replace(out)
    print(f"{day_dir.name}: {len(keys)} instances -> fluke {counts['fluke']}, dorsal fin only {counts['dorsal']}, other {counts['other']} ({out.name})", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("features"); p.add_argument("--day-dir", required=True); p.set_defaults(fn=cmd_features)
    p = sub.add_parser("predict"); p.add_argument("--day-dir", required=True); p.add_argument("--model", default=str(MODEL)); p.add_argument("--out-name", default="view_class.json"); p.set_defaults(fn=cmd_predict)
    p = sub.add_parser("fit"); p.add_argument("--targets", default=str(TARGETS)); p.add_argument("--root", default=str(REPO / "experiments" / "days_all")); p.add_argument("--out", default=str(MODEL))
    p.add_argument("--name", default="view_class_v2"); p.add_argument("--folds", type=int, default=8); p.set_defaults(fn=cmd_fit)
    p = sub.add_parser("report"); p.add_argument("--targets", default=str(TARGETS)); p.add_argument("--root", default=str(REPO / "experiments" / "days_all")); p.add_argument("--model", default=str(MODEL))
    p.add_argument("--split", default="eval", choices=["eval", "val", "train"]); p.set_defaults(fn=cmd_report)
    a = ap.parse_args(argv); a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
