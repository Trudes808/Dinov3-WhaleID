"""Fusion of the per-channel instance-pair scores, fitted on the val days only.

    python -m model_fine_tuning.cluster.fuse --root experiments/days_all --tag cluster_v3 --days splits/days.json --labels labels.json --out experiments/fusion_v3.json

A channel is one (source, part) matrix of cluster_day.py's channels.npz (z-scored cosine of a global embedding, or a
z-scored local match value on the candidate pairs; NaN where it does not apply). Training pairs: same-day pairs of
labelled main instances of the val days. Step 1 calibrates every channel on its own (1-d logistic regression on the
pairs where it is finite: a log-odds of "same whale" from that channel alone). Step 2 fits a logistic regression over
the calibrated channels with a missing indicator per channel (missing = no shared part, or not a candidate pair: both
carry information), so correlated channels (two body embeddings) are not double counted. The result is a pair
log-odds; cluster_day.py uses it as the instance-pair score. Per-channel and fused pair AUC are reported on the val
days (fit) and, for the record only, on the eval days whose channels exist.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LOCAL = ("lightglue", "hrlocal")


def pair_table(root: Path, day: str, tag: str, labels: dict) -> tuple[dict, np.ndarray] | None:
    f = root / day / tag / "channels.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=True); keys = [str(k) for k in z["keys"]]
    L = labels["days"].get(day, {}).get("photos", {})
    main = [i for i, k in enumerate(keys) if "~" not in k or k.endswith("~0")]
    ident = np.array([L.get((keys[i].rsplit("~", 1)[0] if "~" in keys[i] else keys[i]).split("__")[-1], "?") for i in main])
    known = np.array([(g != "?" and not g.split("_")[1].startswith("O")) for g in ident])
    main = np.array(main)[known]; ident = ident[known]
    iu = np.triu_indices(len(main), 1)
    X = {k: z[k][np.ix_(main, main)][iu] for k in z.files if "|" in k}
    y = ident[iu[0]] == ident[iu[1]]
    return X, y


def fit(tables: list[tuple[dict, np.ndarray]], channels: list[str], missing: bool = True) -> dict:
    """missing=False drops the missing-channel indicators: the pair score is then pure evidence (a pair without a fluke
    chart gets no bonus or penalty for that), which average linkage needs; the indicators help pair AUC but shift whole
    groups of pairs (body-only pairs scored +1 in fusion_B) and hurt the clustering."""
    from sklearn.linear_model import LogisticRegression

    X = {k: np.concatenate([t[0][k] if k in t[0] else np.full(len(t[1]), np.nan) for t in tables]) for k in channels}
    y = np.concatenate([t[1] for t in tables])
    calib = {}
    for k in channels:
        v = X[k]; ok = np.isfinite(v)
        if ok.sum() < 100 or y[ok].sum() < 10:
            calib[k] = [0.0, 0.0]; continue
        lr = LogisticRegression(C=10.0, class_weight=None).fit(v[ok, None], y[ok])
        calib[k] = [float(lr.coef_[0, 0]), float(lr.intercept_[0])]
    F = features(X, calib, channels, len(y), missing)
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(F, y)
    w = lr.coef_[0].tolist()
    if not missing:                                                     # same layout as with indicators, indicator weights 0
        w = [x for pair in zip(w, [0.0] * len(w)) for x in pair]
    return {"channels": channels, "calib": calib, "weights": w, "bias": float(lr.intercept_[0]), "missing_indicators": missing}


def features(X: dict, calib: dict, channels: list[str], n: int, missing: bool = True) -> np.ndarray:
    cols = []
    for k in channels:
        v = X.get(k); v = np.full(n, np.nan) if v is None else v
        ok = np.isfinite(v); a, b = calib[k]
        cols.append(np.where(ok, a * np.nan_to_num(v) + b, 0.0))
        if missing:
            cols.append((~ok).astype(float))
    return np.stack(cols, 1)


def score_pairs(X: dict, model: dict, n: int) -> np.ndarray:
    F = features(X, model["calib"], model["channels"], n, True)
    return F @ np.array(model["weights"]) + model["bias"]


def apply_fusion(ch: dict, model: dict) -> np.ndarray:
    """(n, n) fused log-odds from the channel matrices; NaN where no channel applies."""
    shape = next(iter(ch.values())).shape
    out = np.full(shape, model["bias"], np.float64); any_ok = np.zeros(shape, bool); w = model["weights"]
    for i, k in enumerate(model["channels"]):
        M = ch.get(k); M = np.full(shape, np.nan) if M is None else M
        ok = np.isfinite(M); a, b = model["calib"][k]
        out += np.where(ok, w[2 * i] * (a * np.nan_to_num(M) + b), w[2 * i + 1]); any_ok |= ok
    out[~any_ok] = np.nan
    np.fill_diagonal(out, np.nan)
    return out


def report(tables: dict[str, tuple[dict, np.ndarray]], model: dict) -> dict:
    from sklearn.metrics import roc_auc_score

    rep = {}
    for day, (X, y) in tables.items():
        r = {"pairs": int(len(y)), "positive": int(y.sum())}
        for k in model["channels"]:
            v = X.get(k)
            if v is not None and np.isfinite(v).sum() > 100 and y[np.isfinite(v)].any():
                ok = np.isfinite(v); r[f"auc_{k}"] = round(float(roc_auc_score(y[ok], v[ok])), 4); r[f"cover_{k}"] = round(float(ok.mean()), 3)
        s = score_pairs(X, model, len(y)); r["auc_fused"] = round(float(roc_auc_score(y, s)), 4)
        rep[day] = r
    return rep


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True); p.add_argument("--tag", required=True); p.add_argument("--days", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True); p.add_argument("--out", type=Path, required=True); p.add_argument("--channels", default=None, help="comma list; default: all channels of the val days")
    p.add_argument("--no-missing", action="store_true", help="no missing-channel indicators (pure evidence score, better for the clustering)")
    a = p.parse_args(argv)
    labels = json.loads(a.labels.read_text()); days = json.loads(a.days.read_text())
    val = {d: t for d in days["val"] if (t := pair_table(a.root, d, a.tag, labels)) is not None}
    if not val:
        raise SystemExit("no val-day channels found")
    channels = a.channels.split(",") if a.channels else sorted(set.intersection(*[set(t[0]) for t in val.values()]))
    model = fit(list(val.values()), channels, missing=not a.no_missing)
    model["val_report"] = report(val, model)
    ev = {d: t for d in days["eval"] if (t := pair_table(a.root, d, a.tag, labels)) is not None}
    if ev:
        model["eval_report"] = report(ev, model)
    a.out.write_text(json.dumps(model, indent=1))
    print("channels:", channels)
    for k, (w0, w1) in zip(channels, np.array(model["weights"]).reshape(-1, 2)):
        print(f"  {k:28s} calib {model['calib'][k][0]:+.3f} x {model['calib'][k][1]:+.3f}   weight {w0:+.3f}  missing {w1:+.3f}")
    for split, rep in (("val (fit)", model["val_report"]), ("eval (record)", model.get("eval_report", {}))):
        for d, r in rep.items():
            aucs = "  ".join(f"{k[4:]} {v:.3f}" for k, v in r.items() if k.startswith("auc_") and k != "auc_fused")
            print(f"  {split:13s} {d}: fused AUC {r['auc_fused']:.3f} | {aucs}")
    print("->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
