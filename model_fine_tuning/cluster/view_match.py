"""View-specific match embeddings: compare a fluke with flukes and a dorsal fin with dorsal fins.

The view classifier (cluster/view_class.py) says which instances of a day show a fluke and which a dorsal fin. For
those instances this module turns the classifier's own frozen DINOv3 features into a small embedding per part in which
two photos of one whale lie close: the features are centred on the day (its light and water), whitened, scaled down
along the directions in which one animal varies (pose, distance, wet or dry skin) and reduced to the 32 directions that
separate the animals of a day best. Nothing is fine-tuned; the projection is a matrix fitted on the catalogue's animals
of the train days (experiments/view_match/vm_fit.py -> experiments/safe_clusters/view_match.pkl).

    python -m model_fine_tuning.cluster.view_match embed --day-dir DAY     -> DAY/safe/view_match.npz

needs DAY/safe/view_feat.npz and DAY/safe/view_class.json (view_class features + predict). The file holds, per part,
the instance keys, their unit embeddings (float16) and the calibration (a, b): P(same whale) = sigmoid(a x cosine + b)
for the best cosine between a stack and an animal's stacks. A train day is embedded with the projection fitted without
that day (as the linker does), so no day is scored by a projection that saw its animals.

Whale-Review-Web /labeling blends this with the student embedding's part cosine into the fluke and dorsal-fin
similarities of the ranking and of the merge hints (backend/labeling.py, WHALE_VIEW_MATCH). Measured on stack pairs
inside catalogue encounters, same animal against different animals, on days neither model saw (LOG 2026-11-12):
dorsal fin AUC 0.81 against 0.78 for the student's fin cosine, fluke 0.80 against 0.66 on the pairs both can score;
it also scores the flukes that have no part chart (one fluke stack in nine), which the student cannot.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
MODEL = REPO / "experiments" / "safe_clusters" / "view_match.pkl"
PARTS = ("fluke", "dorsal")
SHRINK_N = 8.0          # the day's mean is trusted fully only with many instances of the part: fewer are pulled to the train mean


def unit_blocks(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32).reshape(len(X), 4, 768)
    return (X / np.maximum(np.linalg.norm(X, axis=2, keepdims=True), 1e-6)).reshape(len(X), -1)


def embed_part(X: np.ndarray, P: np.ndarray, mu_train: np.ndarray | None) -> np.ndarray:
    """Unit embeddings of the part's instances of one day: centre on the day (shrunk to the train mean when few), project."""
    if not len(X):
        return np.zeros((0, P.shape[1]), np.float16)
    mu = X.mean(0) if mu_train is None else (X.sum(0) + SHRINK_N * mu_train) / (len(X) + SHRINK_N)
    Z = (X - mu) @ P
    return (Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)).astype(np.float16)


def cmd_embed(a) -> None:
    day_dir = Path(a.day_dir); model = pickle.load(open(a.model, "rb")); day = day_dir.name
    z = np.load(day_dir / "safe" / "view_feat.npz"); keys = [str(k) for k in z["keys"]]; X = unit_blocks(z["X"])
    V = json.loads((day_dir / "safe" / "view_class.json").read_text()); thr = float(V.get("threshold", model.get("thr", 0.5))); views = V["views"]
    out = {"model": np.array(model.get("name", "view_match_v1")), "view_model": np.array(str(V.get("model"))),
           "student_in_sample": np.array(day in model.get("train_days", []))}; counts = {}      # a train day: the student embedding saw its animals (the page keeps its cosine alone there)
    for j, part in enumerate(PARTS):
        m = model["parts"][part]; idx = [i for i, k in enumerate(keys) if k in views and views[k][j] >= thr]
        P = m["loo"].get(day, m["all"])                       # a train day: the projection fitted without it
        out[f"{part}|keys"] = np.array([keys[i] for i in idx]); out[f"{part}|E"] = embed_part(X[idx], P, m.get("mean")); out[f"{part}|cal"] = np.array(m["cal"], np.float32)
        counts[part] = len(idx)
    tmp = day_dir / "safe" / "view_match.tmp.npz"; np.savez(tmp, **out); tmp.replace(day_dir / "safe" / "view_match.npz")
    print(f"{day}: fluke {counts['fluke']} instances, dorsal fin {counts['dorsal']}{' (projection fitted without this day)' if day in model['parts']['fluke']['loo'] else ''}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("embed"); p.add_argument("--day-dir", required=True); p.add_argument("--model", default=str(MODEL)); p.set_defaults(fn=cmd_embed)
    a = ap.parse_args(argv); a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
