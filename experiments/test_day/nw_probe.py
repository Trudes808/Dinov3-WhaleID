"""A probe that NAMES the non-whale instances (whale / person / gear / boat) on the frozen crop features the live filter
uses (link_scores inst_feat.npz: unit CLS + unit mean patch token of a 224-px square crop), fitted on TRAIN days only,
applied once to another day.

    .venv/bin/python experiments/test_day/nw_probe.py --sub ROOT/nw_subclass_train.json --day-dir DAY --out ROOT/nw_probe.json

Labels: the train-day positives of experiments/stage2_linker/nw_labels.json, given a subclass by eye (ROOT/nw_subclass_train.json,
untracked); "whale" = the by-eye negatives plus 1 500 random instances of the train days that the live filter does not flag.
Settings fixed before applying: standardise, multinomial logistic regression, C = 1, class weights balanced; "other" left out.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
DAYS = REPO / "experiments" / "days_all"


def feats(day_dir: Path):
    z = np.load(day_dir / "safe" / "inst_feat.npz")
    unit = lambda x: x.astype(np.float32) / np.maximum(np.linalg.norm(x.astype(np.float32), axis=1, keepdims=True), 1e-6)
    return [str(k) for k in z["keys"]], np.hstack([unit(z["cls"]), unit(z["mean"])])


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--sub", type=Path, required=True); p.add_argument("--day-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True); a = p.parse_args()
    sub = json.loads(a.sub.read_text()); nw = json.loads((REPO / "experiments/stage2_linker/nw_labels.json").read_text())
    train = json.loads((REPO / "model_fine_tuning/splits/days.json").read_text())["train"]
    rng = np.random.default_rng(0); X, y, g = [], [], []
    pool = []
    for d in train:
        if not (DAYS / d / "safe" / "inst_feat.npz").exists():
            continue
        keys, F = feats(DAYS / d); idx = {k: i for i, k in enumerate(keys)}
        drop = set(json.loads((DAYS / d / "safe" / "not_whale.json").read_text())["drop"]) if (DAYS / d / "safe" / "not_whale.json").exists() else set()
        lab = nw.get(d, {})
        for k in lab.get("pos", []):
            c = sub["labels"].get(k)
            if c and c != "other" and k in idx:
                X.append(F[idx[k]]); y.append(c); g.append(d)
        for k in lab.get("neg", []):
            if k in idx:
                X.append(F[idx[k]]); y.append("whale"); g.append(d)
        known = set(lab.get("pos", [])) | set(lab.get("neg", [])) | set(lab.get("skip", [])) | drop
        pool += [(d, F[i]) for i, k in enumerate(keys) if k not in known]
    for j in rng.choice(len(pool), 1500, replace=False):
        X.append(pool[j][1]); y.append("whale"); g.append(pool[j][0])
    X = np.array(X); y = np.array(y); g = np.array(g)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced"))
    cv = cross_val_predict(clf, X, y, groups=g, cv=GroupKFold(5))
    classes = sorted(set(y)); cm = {c: {c2: int(((y == c) & (cv == c2)).sum()) for c2 in classes} for c in classes}
    clf.fit(X, y)
    keys, F = feats(a.day_dir); P = clf.predict_proba(F)
    out = {"classes": list(clf.classes_), "train_counts": {c: int((y == c).sum()) for c in classes}, "train_cv_by_day_group": cm,
           "instances": {k: [round(float(v), 4) for v in pr] for k, pr in zip(keys, P)}}
    a.out.write_text(json.dumps(out, indent=0)); pickle.dump(clf, open(a.out.with_suffix(".pkl"), "wb"))
    print(out["train_counts"]); print("grouped CV on train days (rows truth, cols predicted):", cm)


if __name__ == "__main__":
    main()
