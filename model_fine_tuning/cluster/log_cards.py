"""Which noted whale of the trip log is the stack in hand most likely to be? The order of the noted-whale cards of the
Labeling Assistant (2026-11-19).

The field researchers' trip log notes per group a numbered row per whale: role, dorsal fin shape (height + form),
fluke type, white pectoral, sex, the views photographed. Where the stack in hand is held against noted whales that have
no photos yet, the page puts their cards up in the order of the score written here (best first) instead of by number.
The order is a suggestion: the researcher still presses the key; whales with photos keep the page's own similarity.

What the score uses (per-whale evidence only; the offline experiment experiments/moonshot_sizing/notes_auto measured
what each part adds): per safe cluster, readers fitted on labelled days read
  height  low / moderate / high       the HW-b6 dorsal-fin embedding (DAY/safe/fin_hw/hwb6.npz; cluster mean, centred
  form    triangle / hook / round / square / point / pyramid          on the day), PCA 64, balanced logistic regression
  fluke   fluke type 1-5              frozen view features of the fluke instances (DAY/safe/view_feat.npz) + lightness
  role    calf / mother / other       size against the other whale of the same photo, apparent size, share of two-whale
                                      photos, cluster size, the view classifier's fin / fluke probabilities
  sex     male / female               frozen view features of all instances
and the noted views: P(the cluster shows a fluke / a left / a right side of the fin | that view is noted for the
whale or not), counted on the fitting days. A whale's score for a set of clusters = the sum over the terms of
(the reader's log-likelihood of the whale's noted value; the mean likelihood where nothing is noted), each term summed
over the clusters that carry it / sqrt(their number) (as infer.py's unary term). The white pectoral reader is left
out (alone it did not beat chance, -0.7 points); the pair terms (fin / fluke matchers, linker) and the joint
inference are left out on purpose: the page ranks whales WITH photos by its own similarity (the same matchers), and
towards a whale without photos a pair term has nothing to compare with.

Honest fitting (labels = the catalogue animal of a cluster -> its noted whale row, by group + number): a labelled
TRAIN day is read by readers fitted on the other train days with a log; a VAL day by readers fitted on the train days;
any other day (the four eval days, new days) by readers fitted on all train + val days. Eval days are never fitted on
and never measured here (their scores are written for the page only).

    # once (CPU, ~1 min): the readers of every fold -> moonshot_cache/notes_rank/readers_<fold>.pkl (private)
    .venv/bin/python -m model_fine_tuning.cluster.log_cards fit
    # per day with a trip log (after view_class predict and fin_hw embed / write) -> DAY/safe/log_cards.json
    .venv/bin/python -m model_fine_tuning.cluster.log_cards write --day-dir experiments/days_all/DAY [DAY2 ...]
    # the readers' quality on train (held out) and val days, as readers.log of the experiment (eval days refused)
    .venv/bin/python -m model_fine_tuning.cluster.log_cards check

DAY/safe/log_cards.json (private: lives in the day folder, outside the repository): {day, fold, fit_days, rates:
{fluke|L|R: [P shows | noted, P shows | not noted]}, clusters: {cid: {read: {attr: {class: log-lik}}, shows: [fluke,
L, R], groups: {g: {w: [whale n], total: [...], terms: {term: [...]}}}}}}. `groups` = the noted groups whose window
(+-30 min) holds the cluster's median photo (else the nearest before and after): what the page would show; the page
recomputes the same score from `read` for whichever group a stack ends up in (Whale-Review-Web lab_core.js
tlCardScore, checked against this file by tests/labeling_triplog_cards_e2e.py). The trip log is read from
Whale-Review-Web/data/trip_log/<day>.json (backend/trip_log.py import; WHALE_TRIP_LOG_ROOT moves it).
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import pickle
import re
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DAYS = REPO / "experiments" / "days_all"
SPLITS = json.loads((REPO / "model_fine_tuning" / "splits" / "days.json").read_text())
TRAIN, VAL, EVAL = list(SPLITS["train"]), list(SPLITS["val"]), set(SPLITS["eval"])
CACHE = Path(os.environ.get("WHALE_NOTES_RANK", REPO.parent / "moonshot_cache" / "notes_rank"))      # private: the readers saw the log's values
LOG_ROOT = Path(os.environ.get("WHALE_TRIP_LOG_ROOT", REPO.parent / "Whale-Review-Web" / "data" / "trip_log"))
OUT = "log_cards.json"
HEIGHTS = ("low", "moderate", "high")
FORMS = ("triangle", "hook", "round", "square", "point", "pyramid")
CLASSES = dict(height=list(HEIGHTS), form=list(FORMS), fluke=[1, 2, 3, 4, 5], role=["calf", "mother", "other"], sex=["Male", "Female"])
VIEWS = ("fluke", "L", "R")
TERMS = ("views", "height", "form", "fluke", "role", "sex")
NEAR = 1800.0


# ---- the log, as the page's importer wrote it -------------------------------------------------------------------------
def read_log(day: str) -> dict | None:
    f = LOG_ROOT / f"{day}.json"
    return json.loads(f.read_text()) if f.exists() else None


def whale_attrs(w: dict) -> dict:
    """A noted whale row (backend/trip_log.py) -> the readers' classes (None = not noted / not one of them)."""
    d = w.get("dorsal") or {}
    fl = w.get("fluke")
    role = {"Calf": "calf", "Mother": "mother", "Escort": "other", "Primary": "other", "Secondary": "other", "Challenger": "other", "Nuclear": "other"}.get(w.get("role") or "")
    cov = None
    if w.get("coverage_raw"):
        c = set(w.get("coverage") or [])
        cov = {v for v, k in (("fluke", "fluke"), ("L", "dorsal_left"), ("R", "dorsal_right")) if k in c}
    return dict(height=d.get("height") if d.get("height") in HEIGHTS else None, form=d.get("form") if d.get("form") in FORMS else None,
                fluke=int(fl) if isinstance(fl, str) and fl in ("1", "2", "3", "4", "5") else None, role=role,
                sex={"F": "Female", "M": "Male"}.get(w.get("sex") or ""), cov=cov)


def log_groups(L: dict) -> dict:
    """{group n: {"whales": {whale n: attrs}, "a": start s, "b": end s}} for the humpback groups (windows as trip_log._windows)."""
    gs = [g for g in L.get("groups", []) if g.get("humpback") and g.get("n") is not None]
    starts = sorted(g["start_s"] for g in L.get("groups", []) if g.get("start_s") is not None)
    out = {}
    for g in gs:
        a, b = g.get("start_s"), g.get("end_s")
        if a is not None and (b is None or b < a):
            nxt = [s for s in starts if s > a]
            b = min(nxt) if nxt else a + 3600.0
        out[g["n"]] = {"whales": {w["n"]: whale_attrs(w) for w in g.get("whales") or [] if w.get("n") is not None}, "a": a, "b": b}
    return out


# ---- per-cluster inputs (as experiments/moonshot_sizing/notes_auto/data.py clusters_day; no labels) --------------------
def _sod(dt: str) -> float | None:
    m = re.match(r"\d{4}:\d{2}:\d{2} (\d{2}):(\d{2}):(\d{2})", dt or "")
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) if m else None


def clusters(day_dir: Path) -> dict:
    d = day_dir / "safe"
    C = json.loads((d / "clusters.json").read_text())["clusters"]
    ph = json.loads((d / "photos.json").read_text())
    ts = json.loads((day_dir / "timestamps.json").read_text())
    vc = json.loads((d / "view_class.json").read_text())
    V, BOX = vc["views"], vc.get("box") or {}
    vf = np.load(d / "view_feat.npz")
    vkeys = [str(k) for k in vf["keys"]]
    X = vf["X"].astype(np.float32).reshape(len(vkeys), 4, 768)
    X = (X / np.maximum(np.linalg.norm(X, axis=2, keepdims=True), 1e-6)).reshape(len(vkeys), -1)
    vidx = {k: i for i, k in enumerate(vkeys)}
    hf = d / "fin_hw" / "hwb6.npz"
    if hf.exists():
        z = np.load(hf)
        H = z["X"].astype(np.float32)
        H /= np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-9)
        hidx = {str(k): i for i, k in enumerate(z["keys"])}
    else:
        H, hidx = None, {}
    inst = {it["key"]: it for v in ph.values() for it in v.get("instances") or []}
    out = {}
    for c in C:
        keys = c["keys"]
        fins, flukes, flank, light, rel, app, multi, fin_h = [], [], collections.Counter(), [], [], [], 0, []
        for key in keys:
            v = V.get(key)
            p = key.rsplit("~", 1)[0]
            P = ph.get(p, {})
            it = inst.get(key)
            if v and v[1] >= 0.5:
                if key in hidx:
                    fins.append(hidx[key])
                if P.get("head_side") in (1, -1) and (P.get("cue_confidence") or 0) >= 0.5:
                    flank[P["head_side"]] += 1
            if v and v[0] >= 0.5 and key in vidx:
                flukes.append(vidx[key])
                if it and it.get("lab_whale"):
                    light.append([it["lab_whale"][0] - (it.get("lab_water") or [0])[0], it["lab_whale"][0], it.get("lum_std") or 0, it.get("contrast") or 0])
            if it:
                a = float(it.get("area") or 0)
                if P.get("focal_mm") and a > 0:
                    app.append(0.5 * np.log(a) - np.log(P["focal_mm"]))
                others = [float(o.get("area") or 0) for o in P.get("instances") or [] if o["key"] != key]
                if others and a > 0:
                    multi += 1
                    rel.append(np.log(a / max(max(others), 1.0)))
        nb = [vidx[k] for k in keys if k in vidx]
        sods = sorted(s for s in (_sod((ts.get(p) or {}).get("datetime", "")) for p in c["photos"]) if s is not None)
        out[c["id"]] = dict(
            id=c["id"], size=len(c["photos"]), catalog=c.get("catalog") or {}, tmid=sods[len(sods) // 2] if sods else None,
            fluke=bool(flukes), flankL=flank[-1] > 0, flankR=flank[1] > 0,
            pfluke=max([V[k][0] for k in keys if k in V] or [0]), pfin=max([V[k][1] for k in keys if k in V] or [0]),
            hw=H[fins].mean(0) if fins else None, base_fluke=X[flukes].mean(0) if flukes else None, base_all=X[nb].mean(0) if nb else None,
            light=np.mean(light, 0) if light else None, frac_multi=multi / max(1, len(keys)),
            rel=(float(np.median(rel)), float(np.min(rel)), float(np.max(rel))) if rel else None, app=float(np.median(app)) if app else None)
    return out


def shows(c: dict, v: str) -> bool:
    return c["fluke"] if v == "fluke" else c["flankL"] if v == "L" else c["flankR"]


def x_of(C: dict, attr: str) -> dict:
    """{cluster: feature vector} for the clusters the reader can read (readers.py x_of)."""
    if attr in ("height", "form"):
        hw = [c["hw"] for c in C.values() if c["hw"] is not None]
        mu = np.mean(hw, 0) if hw else 0
        out = {}
        for k, c in C.items():
            if c["hw"] is not None:
                x = c["hw"] - mu
                out[k] = x / max(np.linalg.norm(x), 1e-9)
        return out
    if attr == "fluke":
        return {k: np.concatenate([c["base_fluke"], 0.02 * c["light"] if c["light"] is not None else np.zeros(4)]) for k, c in C.items() if c["base_fluke"] is not None}
    if attr == "sex":
        return {k: c["base_all"] for k, c in C.items() if c["base_all"] is not None}
    if attr == "role":
        out = {}
        for k, c in C.items():
            r = c["rel"] or (0.0, 0.0, 0.0)
            out[k] = np.array([c["frac_multi"], *r, float(c["rel"] is not None), c["app"] if c["app"] is not None else 0.0,
                               float(c["app"] is not None), np.log(c["size"]), c["pfin"], c["pfluke"]], float)
        return out
    raise KeyError(attr)


# ---- labels (fitting days only) ---------------------------------------------------------------------------------------
def truth(c: dict):
    e = sorted(c["catalog"].items(), key=lambda kv: -kv[1])
    if not e:
        return None
    m = re.search(r"_G(\d+)A(\d+)", e[0][0])
    return ("G", int(m.group(1)), int(m.group(2))) if m else ("O", None, None)


def whale_of(rec: dict, c: dict):
    t = truth(c)
    if not t or t[0] != "G":
        return None
    return rec["groups"].get(t[1], {}).get("whales", {}).get(t[2])


def label(attr: str, w):
    return None if w is None else w[attr]


def labelled(day: str) -> dict:
    """A fitting day's record: clusters with their labels and the log. Refuses eval days."""
    if day in EVAL:
        raise ValueError(f"{day} is an eval day: never fitted on")
    L = read_log(day)
    return dict(day=day, clusters=clusters(DAYS / day), groups=log_groups(L) if L else {})


class Reader:
    """readers.py Reader: PCA 64 (role: standardised), class-balanced logistic regression; read = log P per class."""

    def __init__(s, attr: str, pca: int = 64):
        s.attr, s.pca = attr, pca

    def fit(s, recs: list[dict]):
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        X, y = [], []
        for rec in recs:
            xs = x_of(rec["clusters"], s.attr)
            for k, c in rec["clusters"].items():
                lab = label(s.attr, whale_of(rec, c))
                if k in xs and lab is not None and lab in CLASSES[s.attr]:
                    X.append(xs[k]); y.append(lab)
        X = np.array(X, np.float64)
        s.n_train = len(y)
        s.pre = StandardScaler().fit(X) if s.attr == "role" else PCA(min(s.pca, len(X) - 1), random_state=0).fit(X)
        s.lr = LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced").fit(s.pre.transform(X), y)
        return s

    def read(s, C: dict) -> dict:
        xs = x_of(C, s.attr)
        if not xs:
            return {}
        ks = list(xs)
        Lg = np.log(np.clip(s.lr.predict_proba(s.pre.transform(np.array([xs[k] for k in ks], np.float64))), 1e-4, 1))
        out = {}
        for i, k in enumerate(ks):
            d = dict(zip(s.lr.classes_, Lg[i])); m = float(np.mean(Lg[i]))
            out[k] = {str(c): round(float(d.get(c, m)), 3) for c in CLASSES[s.attr]}
        return out


def view_rates(recs: list[dict]) -> dict:
    n = collections.Counter()
    for rec in recs:
        for c in rec["clusters"].values():
            w = whale_of(rec, c)
            if w is None or w["cov"] is None:
                continue
            for v in VIEWS:
                noted = v in w["cov"]
                n[(v, noted)] += 1
                n[(v, noted, "shows")] += shows(c, v)
    return {v: [round((n[(v, True, "shows")] + 1) / (n[(v, True)] + 2), 4), round((n[(v, False, "shows")] + 1) / (n[(v, False)] + 2), 4)] for v in VIEWS}


# ---- folds ------------------------------------------------------------------------------------------------------------
def logged(days) -> list[str]:
    return [d for d in days if d not in EVAL and (LOG_ROOT / f"{d}.json").exists() and (DAYS / d / "safe" / "clusters.json").exists()]


def fold_of(day: str) -> tuple[str, list[str]]:
    """The readers a day is read by: a train day -> the other train days; a val day -> the train days; any other day
    (eval, new) -> all train + val days. Only days with a log and a day folder are fitted on."""
    tr = logged(TRAIN)
    if day in TRAIN:
        return f"train-{day}", [d for d in tr if d != day]
    if day in VAL:
        return "train", tr
    return "all", tr + logged(VAL)


def fit_fold(name: str, days: list[str], recs: dict | None = None) -> dict:
    recs = recs if recs is not None else {}
    R = [recs.setdefault(d, labelled(d)) for d in days]
    model = {"fold": name, "fit_days": days, "readers": {a: Reader(a).fit(R) for a in CLASSES}, "rates": view_rates(R), "fitted": time.strftime("%Y-%m-%d %H:%M")}
    CACHE.mkdir(parents=True, exist_ok=True)
    pickle.dump(model, open(CACHE / f"readers_{name}.pkl", "wb"))
    return model


def load_fold(day: str, recs: dict | None = None) -> dict:
    name, days = fold_of(day)
    f = CACHE / f"readers_{name}.pkl"
    if f.exists():
        m = pickle.load(open(f, "rb"))
        if m["fit_days"] == days:
            return m
    return fit_fold(name, days, recs)


# ---- the score ----------------------------------------------------------------------------------------------------------
def _lse(v):
    m = max(v)
    return m + math.log(sum(math.exp(x - m) for x in v))


def cluster_terms(entry: dict, rates: dict, whales: list[dict]) -> dict:
    """{term: [value per whale]} for one cluster (infer.py cluster_unary, without the white pectoral and the anchors);
    only the terms that read something. Mirrored in Whale-Review-Web lab_core.js tlCardScore."""
    out = {}
    if any(w["cov"] is not None for w in whales):
        v = []
        for w in whales:
            if w["cov"] is None:
                v.append(None); continue
            s = 0.0
            for i, view in enumerate(VIEWS):
                q1, q0 = rates[view]
                p = q1 if view in w["cov"] else q0
                s += math.log(p) if entry["shows"][i] else math.log(1 - p)
            v.append(s)
        known = [x for x in v if x is not None]
        out["views"] = [x if x is not None else sum(known) / len(known) for x in v]
    for attr in ("height", "form", "fluke", "role", "sex"):
        L = entry["read"].get(attr)
        vals = [None if w[attr] is None else str(w[attr]) for w in whales]
        if not L or all(x is None for x in vals):
            continue
        un = _lse(list(L.values())) - math.log(len(L))
        out[attr] = [L[x] if x is not None and x in L else un for x in vals]
    return out


def unit_score(entries: list[dict], rates: dict, whales: list[dict]) -> tuple[list[float], dict]:
    """The score of each whale for a set of clusters: per term the sum over the clusters carrying it / sqrt(their number)."""
    acc = collections.defaultdict(list)
    for e in entries:
        for t, v in cluster_terms(e, rates, whales).items():
            acc[t].append(v)
    terms = {t: [sum(col) / math.sqrt(len(vs)) for col in zip(*vs)] for t, vs in acc.items()}
    total = [sum(terms[t][i] for t in terms) for i in range(len(whales))]
    return total, terms


def write_day(day_dir: Path, recs: dict | None = None) -> Path | None:
    day = day_dir.name
    L = read_log(day)
    if not L:
        print(f"{day}: no trip log, nothing written"); return None
    G = log_groups(L)
    if not G:
        print(f"{day}: no humpback group in the log, nothing written"); return None
    M = load_fold(day, recs)
    C = clusters(day_dir)
    reads = {a: M["readers"][a].read(C) for a in CLASSES}
    win = sorted((g["a"], g["b"], n) for n, g in G.items() if g["a"] is not None)
    out = {}
    for k, c in C.items():
        e = {"read": {a: reads[a][k] for a in CLASSES if k in reads[a]}, "shows": [bool(shows(c, v)) for v in VIEWS]}
        t = c["tmid"]; cand = []
        if t is not None and win:
            cand = [n for a, b, n in win if a - NEAR <= t <= b + NEAR]
            if not cand:
                before = [x for x in win if x[1] < t]; after = [x for x in win if x[0] > t]
                cand = ([max(before, key=lambda x: x[1])[2]] if before else []) + ([min(after)[2]] if after else [])
        e["groups"] = {}
        for n in cand:
            wn = sorted(G[n]["whales"])
            if not wn:
                continue
            ws = [G[n]["whales"][x] for x in wn]
            total, terms = unit_score([e], M["rates"], ws)
            e["groups"][str(n)] = {"w": wn, "total": [round(x, 3) for x in total], "terms": {a: [round(x, 3) for x in v] for a, v in terms.items()}}
        out[str(k)] = e
    doc = {"day": day, "fold": M["fold"], "fit_days": M["fit_days"], "fitted": M.get("fitted"), "written": time.strftime("%Y-%m-%d %H:%M:%S"),
           "terms": list(TERMS), "views": list(VIEWS), "rates": M["rates"], "clusters": out}
    f = day_dir / "safe" / OUT; tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, separators=(",", ":"))); tmp.replace(f)
    print(f"{day}: {len(out)} clusters, fold {M['fold']} ({len(M['fit_days'])} days) -> {f}", flush=True)
    return f


# ---- quality on train (held out) and val days -------------------------------------------------------------------------
def check() -> None:
    days = logged(TRAIN) + logged(VAL)
    recs = {d: labelled(d) for d in days}
    for tag, part in (("train days, each read by readers fitted on the other train days", logged(TRAIN)), ("val days, readers fitted on the train days", logged(VAL))):
        print(f"\n== {tag} ({len(part)} days) ==")
        hit = collections.Counter(); tot = collections.Counter(); first = [0, 0, 0, 0]
        for d in part:
            M = load_fold(d, recs); rec = recs[d]; C = rec["clusters"]
            reads = {a: M["readers"][a].read(C) for a in CLASSES}
            for k, c in C.items():
                w = whale_of(rec, c)
                if w is None:
                    continue
                t = truth(c); grp = rec["groups"][t[1]]["whales"]
                for a in CLASSES:
                    if k not in reads[a] or label(a, w) is None:
                        continue
                    vals = {label(a, x) for x in grp.values()} - {None}
                    if len(vals) > 1:
                        tot[a] += 1; hit[a] += max(vals, key=lambda v: reads[a][k][str(v)]) == label(a, w)
                wn = sorted(grp)
                if len(wn) < 2:
                    continue
                e = {"read": {a: reads[a][k] for a in CLASSES if k in reads[a]}, "shows": [bool(shows(c, v)) for v in VIEWS]}
                total, _ = unit_score([e], M["rates"], [grp[x] for x in wn])
                i = wn.index(t[2]); order = sorted(range(len(wn)), key=lambda j: (-total[j], wn[j]))
                first[0] += 1; first[1] += order[0] == i; first[2] += order.index(i) + 1; first[3] += i + 1
        for a in CLASSES:
            print(f"   {a:<7s} own whale's value first among its group's {hit[a]}/{tot[a]} = {hit[a] / max(1, tot[a]):.3f}")
        print(f"   single labelled clusters in a group of 2+ noted whales: own whale first by score {first[1]}/{first[0]} = {first[1] / max(1, first[0]):.3f}, "
              f"mean position by score {first[2] / max(1, first[0]):.2f} against by number {first[3] / max(1, first[0]):.2f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fit", help="fit the readers of every fold (train days with a log; val days; never eval)")
    w = sub.add_parser("write", help="DAY/safe/log_cards.json for days with a trip log")
    w.add_argument("--day-dir", nargs="+", required=True)
    sub.add_parser("check", help="the readers' quality on train (held out) and val days")
    a = ap.parse_args(argv)
    if a.cmd == "fit":
        recs: dict = {}
        tr, va = logged(TRAIN), logged(VAL)
        for d in tr:
            fit_fold(f"train-{d}", [x for x in tr if x != d], recs); print(f"fold train-{d}: {len(tr) - 1} days", flush=True)
        fit_fold("train", tr, recs); print(f"fold train: {len(tr)} days")
        fit_fold("all", tr + va, recs); print(f"fold all: {len(tr) + len(va)} days")
        return 0
    if a.cmd == "write":
        recs = {}
        for d in a.day_dir:
            write_day(Path(d).resolve(), recs)
        return 0
    check()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
