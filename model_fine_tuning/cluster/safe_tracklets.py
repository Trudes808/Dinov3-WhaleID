"""Safe clusters, stage two: tracklets (cluster/tracklets.py with the identity embedding in the link cost: pure, but
fragmented) are re-linked along the capture sequence by a classifier on tracklet pairs.

    python -m model_fine_tuning.cluster.safe_tracklets fit    --root ROOT --days days.json --labels L --out experiments/safe_clusters/T.pkl [--split train]
    python -m model_fine_tuning.cluster.safe_tracklets sweep  --root ROOT --days days.json --labels L --model T.pkl [--thresholds ...]
    python -m model_fine_tuning.cluster.safe_tracklets report --root ROOT --days days.json --labels L --model T.pkl --threshold T [--baselines] [--write]

Stage one: bursts of one camera (gaps <= `gap` s) are cut into tracklets by the Hungarian assignment of consecutive
photos' instances on position, size and the student body-embedding cosine (max cost `cost`). Stage two: candidate pairs
are tracklets of one camera at most `skip` tracklets apart in start order whose gap (B.start - A.end) is at most
`window` s and which share no photo. Features: the gap and the photos between, the continuity between A's last and
B's first frame (safe_clusters.pair_features: position, size, exposure, range, rotation, colour), the appearance over
all instance pairs of the two tracklets (max / mean cosine per part, best over parts), the tracklet sizes and the
mean colour difference. Labels from the catalog on pure labelled tracklets. P(same) above the threshold links
tracklets best first by union-find, one instance per photo per cluster. Needs DAY/safe/photos.json
(safe_clusters features) and DAY/cluster_v3/embeddings.npz.

--filter (2026-11-11) leaves out the instances DAY/safe/not_whale.json lists (link_scores not-whale: people on the bow,
the deck, gear, drones that got a whale mask): they form no tracklet, a frame with a person and a whale counts as a
single-whale frame for the strict tier, and when the dropped instance was the largest one (~0) the catalog label of
the photo goes to the largest instance that is left.

--guard-fused (2026-11-23, with --strict; default off): the fused-instance guard B_G (cluster/fused_guard.py). No strict
link whose end instances (A's last, B's first) include an instance flagged by flag_G (a fluke-like mask component
joined far from the back of probably another whale). Instances and keys do not change; a refused link leaves two
clusters where there was one (later ids shift). The report gains "guard_fused" (flagged keys, refused candidates).
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .safe_clusters import (APP_FEATURES, PARTS, baselines, cluster_metrics, day_labels, fit_model, importance, load_embeddings, load_model, pair_features,
                            predict)
from .tracklets import load_instances, make_tracklets

TRACK_FEATURES = ["gap", "log_gap", "between", "rank", "n_a", "n_b", "singleton", "span_a", "span_b",
                  "dt", "d_pos", "rate_pos", "d_size", "rate_size", "log_area_min", "d_focal", "d_ev", "d_range", "d_rot", "rate_rot", "d_nose", "side_flip",
                  "d_lab_whale", "d_lab_water", "d_asym", "d_contrast", "n_inst_max", "parts_shared", "parts_union",
                  "app_body_max", "app_body_mean", "app_fin_max", "app_fin_mean", "app_fluke_max", "app_fluke_mean", "app_best", "app_n", "lab_whale_mean_d"]


def build_tracklets(day_dir: Path, feats: dict, emb: dict, gap: float, cost: float, drop: frozenset = frozenset()) -> list[dict]:
    """Stage one: tracklets with the body embedding in the link cost; each gets its photos' feature rows. drop: instance
    keys left out (not whales); feats must come without them already."""
    ts = json.loads((day_dir / "timestamps.json").read_text())
    inst = load_instances(day_dir, "parts_hr")
    if drop:
        inst = {s: [i for i in v if i["key"] not in drop] for s, v in inst.items()}
    tracks = make_tracklets(ts, inst, gap, emb.get("body"), cost)
    inst_feat = {i["key"]: i for s, f in feats.items() for i in f.get("instances", [])}
    # the instance the photo's catalog label describes: the largest one (~0), or the largest one left when that was dropped
    promoted = {f["instances"][0]["key"] for s, f in feats.items() if f"{s}~0" in drop and f.get("instances")}
    out = []
    for t in tracks:
        keys = [k for k in t["keys"] if k in inst_feat]
        if not keys:
            continue
        photos = [k.rsplit("~", 1)[0] for k in keys]
        out.append({"id": t["id"], "camera": feats[photos[0]]["camera"], "keys": keys, "photos": photos, "start": float(feats[photos[0]]["time"]),
                    "end": float(feats[photos[-1]]["time"]), "main_photos": [p for k, p in zip(keys, photos) if k.endswith("~0") or k in promoted]})
    out.sort(key=lambda t: (t["camera"], t["start"], t["end"]))
    for k, t in enumerate(out):
        t["id"] = k
    return out


def tracklet_label(t: dict, labels: dict | None) -> str | None:
    """The catalog animal of a tracklet (its labelled main photos), None if unlabelled or impure."""
    if labels is None:
        return None
    ids = {labels.get(p.split("__")[-1]) for p in t["main_photos"]} - {None}
    return ids.pop() if len(ids) == 1 else None


def track_pair_features(feats: dict, inst_feat: dict, a: dict, b: dict, rank: int, between: int, emb: dict) -> dict:
    gap = b["start"] - a["end"]
    f = {"gap": round(gap, 3), "log_gap": round(math.log1p(max(gap, 0.0)), 4), "between": between, "rank": rank,
         "n_a": len(a["photos"]), "n_b": len(b["photos"]), "singleton": float(len(a["photos"]) == 1 or len(b["photos"]) == 1),
         "span_a": round(a["end"] - a["start"], 3), "span_b": round(b["end"] - b["start"], 3)}
    pa, pb = a["photos"][-1], b["photos"][0]
    pf = pair_features(feats[pa], inst_feat[a["keys"][-1]], feats[pb], inst_feat[b["keys"][0]], between + 1, {})
    for k in ("dt", "d_pos", "rate_pos", "d_size", "rate_size", "log_area_min", "d_focal", "d_ev", "d_range", "d_rot", "rate_rot", "d_nose", "side_flip",
              "d_lab_whale", "d_lab_water", "d_asym", "d_contrast", "n_inst_max", "parts_shared", "parts_union"):
        if pf.get(k) is not None:
            f[k] = pf[k]
    best, n = None, 0
    for part in PARTS:
        E = emb.get(part, {})
        A = [E[k] for k in a["keys"] if k in E]; B = [E[k] for k in b["keys"] if k in E]
        if A and B:
            S = np.asarray(A) @ np.asarray(B).T
            f[f"app_{part}_max"] = round(float(S.max()), 4); f[f"app_{part}_mean"] = round(float(S.mean()), 4); n += 1
            best = f[f"app_{part}_max"] if best is None else max(best, f[f"app_{part}_max"])
    if best is not None:
        f["app_best"] = best
    f["app_n"] = float(n)
    la = [inst_feat[k]["lab_whale"] for k in a["keys"] if inst_feat[k].get("lab_whale")]; lb = [inst_feat[k]["lab_whale"] for k in b["keys"] if inst_feat[k].get("lab_whale")]
    if la and lb:
        f["lab_whale_mean_d"] = round(float(np.linalg.norm(np.mean(la, 0) - np.mean(lb, 0))), 3)
    return f


def candidate_track_pairs(feats: dict, tracks: list[dict], emb: dict, window: float, skip: int, labels: dict | None) -> list[dict]:
    inst_feat = {i["key"]: i for s, f in feats.items() for i in f.get("instances", [])}
    by_cam = defaultdict(list)
    for t in tracks:
        by_cam[t["camera"]].append(t)
    photo_order = {}
    for s, f in feats.items():
        if f.get("time") is not None:
            photo_order.setdefault(f["camera"], []).append((float(f["time"]), s))
    pos = {}
    for cam, items in photo_order.items():
        items.sort(); pos.update({s: i for i, (_, s) in enumerate(items)})
    out = []
    for cam, items in by_cam.items():
        items.sort(key=lambda t: (t["start"], t["end"]))
        for i, a in enumerate(items):
            la = tracklet_label(a, labels); pa = set(a["photos"])
            for r in range(1, skip + 1):
                if i + r >= len(items):
                    break
                b = items[i + r]
                if b["start"] - a["end"] > window:
                    break
                if pa & set(b["photos"]):
                    continue
                between = max(pos[b["photos"][0]] - pos[a["photos"][-1]] - 1, 0)
                row = {"a": a["id"], "b": b["id"], "camera": cam}
                row.update(track_pair_features(feats, inst_feat, a, b, r, between, emb))
                lb = tracklet_label(b, labels)
                row["same"] = (la == lb) if (la and lb) else None
                out.append(row)
    return out


class TrackUnion:
    def __init__(self, tracks):
        self.p = {t["id"]: t["id"] for t in tracks}; self.photos = {t["id"]: set(t["photos"]) for t in tracks}

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]; i = self.p[i]
        return i

    def union(self, a, b) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        if self.photos[ra] & self.photos[rb]:
            return False
        if len(self.photos[ra]) < len(self.photos[rb]):
            ra, rb = rb, ra
        self.p[rb] = ra; self.photos[ra] |= self.photos.pop(rb)
        return True


def link_tracklets(tracks: list[dict], pairs: list[dict], prob: np.ndarray, threshold: float, allow: np.ndarray | None = None) -> list[dict]:
    """Union-find over the links with P >= threshold (and allow[j], e.g. the strict rule), best first."""
    uf = TrackUnion(tracks); by_id = {t["id"]: t for t in tracks}
    accepted = []
    for j in np.argsort(-prob):
        if prob[j] < threshold:
            break
        if allow is not None and not allow[j]:
            continue
        p = pairs[j]
        if uf.find(p["a"]) != uf.find(p["b"]) and uf.union(p["a"], p["b"]):
            accepted.append((p["a"], p["b"], round(float(prob[j]), 4)))
    groups = defaultdict(list)
    for t in tracks:
        groups[uf.find(t["id"])].append(t)
    links = defaultdict(list)
    for a, b, pr in accepted:
        links[uf.find(a)].append((a, b, pr))
    out = []
    for root, ts in groups.items():
        ts.sort(key=lambda t: t["start"])
        out.append({"tracklets": [t["id"] for t in ts], "keys": [k for t in ts for k in t["keys"]], "photos": [p for t in ts for p in t["photos"]],
                    "main_photos": [p for t in ts for p in t["main_photos"]], "camera": ts[0]["camera"], "start": ts[0]["start"], "end": max(t["end"] for t in ts),
                    "links": links.get(root, [])})
    out.sort(key=lambda c: (c["start"], c["camera"]))
    for k, c in enumerate(out):
        c["id"] = k
    return out


def photo_clusters(clusters: list[dict], feats: dict) -> list[list[str]]:
    out = [c["main_photos"] for c in clusters if c["main_photos"]]
    seen = {s for c in out for s in c}
    return out + [[s] for s, f in feats.items() if f.get("time") is not None and s not in seen]


def day_data(root: Path, day: str, labels_path: Path, gap: float, cost: float, window: float, skip: int, use_filter: bool = False) -> dict:
    d = root / day; feats = json.loads((d / "safe" / "photos.json").read_text()); emb = load_embeddings(d); labels = day_labels(labels_path, day)
    drop, info = frozenset(), None
    if use_filter:
        nw = json.loads((d / "safe" / "not_whale.json").read_text()); drop = frozenset(nw.get("drop", []))
        info = {"name": nw.get("model"), "threshold": nw.get("threshold"), "dropped": len(drop)}
        for f in feats.values():
            if f.get("instances"):
                f["instances"] = [i for i in f["instances"] if i["key"] not in drop]
    tracks = build_tracklets(d, feats, emb, gap, cost, drop)
    pairs = candidate_track_pairs(feats, tracks, emb, window, skip, labels)
    return {"day": day, "dir": d, "feats": feats, "labels": labels, "tracks": tracks, "pairs": pairs, "filter": info}


STRICT_MIN_P = 0.995          # the automatic (strict) tier: P at least this and no end frame with several whale instances
SUGGEST_MIN_P = 0.2           # weaker stage-two candidates are kept as suggestions for the reviewer
SUGGEST_WINDOW = 900.0        # s: appearance-ranked suggestions between clusters of one interaction
SUGGEST_MIN_COS = 0.5
SUGGEST_TOP = 5


def suggestions(dd: dict, clusters: list[dict], prob: np.ndarray) -> list[dict]:
    """Candidate links between different clusters for the reviewer: the stage-two pairs below the strict tier
    (best P per cluster pair) and, within SUGGEST_WINDOW s, the cluster pairs with the highest identity-embedding
    cosine over their instances (top SUGGEST_TOP per cluster, best over parts)."""
    cl_of = {t: c["id"] for c in clusters for t in c["tracklets"]}
    by_id = {t["id"]: t for t in dd["tracks"]}
    best: dict = {}
    for p, pr in zip(dd["pairs"], prob):
        ca, cb = cl_of.get(p["a"]), cl_of.get(p["b"])
        if ca is None or cb is None or ca == cb or pr < SUGGEST_MIN_P:
            continue
        k = (min(ca, cb), max(ca, cb))
        if k not in best or pr > best[k]["p"]:
            best[k] = {"a": k[0], "b": k[1], "p": round(float(pr), 4), "key_a": by_id[p["a"]]["keys"][-1] if ca == k[0] else by_id[p["b"]]["keys"][0],
                       "key_b": by_id[p["b"]]["keys"][0] if ca == k[0] else by_id[p["a"]]["keys"][-1], "why": "sequence"}
    emb = load_embeddings(dd["dir"])
    mats = {}
    for part, E in emb.items():
        for c in clusters:
            keys = [k for k in c["keys"] if k in E]
            if keys:
                mats[(part, c["id"])] = (keys, np.asarray([E[k] for k in keys], np.float32))
    per_cluster: dict = {c["id"]: [] for c in clusters}
    for i, a in enumerate(clusters):
        for b in clusters[i + 1:]:
            gap = max(b["start"] - a["end"], a["start"] - b["end"], 0.0)
            if gap > SUGGEST_WINDOW:
                continue
            top = None
            for part in PARTS:
                ma, mb = mats.get((part, a["id"])), mats.get((part, b["id"]))
                if ma is None or mb is None:
                    continue
                S = ma[1] @ mb[1].T; j = int(np.argmax(S)); v = float(S.flat[j])
                if top is None or v > top[0]:
                    top = (v, part, ma[0][j // S.shape[1]], mb[0][j % S.shape[1]])
            if top and top[0] >= SUGGEST_MIN_COS:
                per_cluster[a["id"]].append((top[0], b["id"], top)); per_cluster[b["id"]].append((top[0], a["id"], top))
    for cid, lst in per_cluster.items():
        lst.sort(key=lambda t: -t[0])
        for v, other, top in lst[:SUGGEST_TOP]:
            k = (min(cid, other), max(cid, other))
            e = best.setdefault(k, {"a": k[0], "b": k[1], "p": None, "why": "appearance"})
            e.update({"app": round(v, 4), "app_part": top[1]})
            if "key_a" not in e:
                e["key_a"], e["key_b"] = (top[2], top[3]) if cid == k[0] else (top[3], top[2])
    out = list(best.values())
    out.sort(key=lambda e: -(e["p"] if e["p"] is not None else 0.0) - (e.get("app") or 0.0))
    return out


def evaluate(dd: dict, model: dict, threshold: float, window: float, strict: bool = False, with_suggestions: bool = False, guard: frozenset | None = None) -> dict:
    """guard (with strict): instance keys flagged by fused_guard.flag_G; no strict link whose end instances include one."""
    prob = predict(model, dd["pairs"]) if dd["pairs"] else np.zeros(0)
    allow = np.array([p["n_inst_max"] <= 1 for p in dd["pairs"]], bool) if (strict and dd["pairs"]) else None
    if guard is not None and allow is not None:
        from .fused_guard import guard_allow
        g = guard_allow(dd["tracks"], dd["pairs"], guard)
        dd["guard_refused"] = int(np.sum(allow & ~g & (prob >= threshold)))       # candidate links of the strict tier the guard refused
        allow = allow & g
    clusters = link_tracklets(dd["tracks"], dd["pairs"], prob, threshold, allow)
    inst_count = {s: len(f.get("instances", [])) for s, f in dd["feats"].items()}
    for c in clusters:
        c["n_multi"] = sum(1 for s in c["photos"] if inst_count.get(s, 0) > 1)
        if dd["labels"]:
            ids = Counter(dd["labels"].get(s.split("__")[-1]) for s in c["main_photos"]); ids.pop(None, None)
            c["catalog"] = dict(ids.most_common())
    if with_suggestions:
        dd["suggestions"] = suggestions(dd, clusters, prob)
    pc = photo_clusters(clusters, dd["feats"])
    rep = {"day": dd["day"], "threshold": threshold, "strict": strict, "tracklets": len(dd["tracks"]), "pairs": len(dd["pairs"]), "links": sum(len(c["links"]) for c in clusters)}
    if dd.get("filter"):
        rep["filter"] = dd["filter"]
    if dd["labels"]:
        rep["metrics"] = cluster_metrics(pc, dd["labels"], dd["feats"], window)
        rep["metrics_single"] = cluster_metrics(pc, dd["labels"], dd["feats"], window, single_only=True)
        rep["stage1"] = cluster_metrics([t["main_photos"] for t in dd["tracks"] if t["main_photos"]], dd["labels"], dd["feats"], window)
        y = np.array([p.get("same") for p in dd["pairs"]], dtype=object); known = np.array([v is not None for v in y])
        if known.sum() and len(set(y[known])) == 2:
            from sklearn.metrics import roc_auc_score
            rep["pair_auc"] = round(float(roc_auc_score(y[known].astype(bool), prob[known])), 4)
        acc = {(a, b) for c in clusters for a, b, _ in c["links"]}; by_id = {t["id"]: t for t in dd["tracks"]}
        rep["wrong_links"] = [(by_id[p["a"]]["photos"][-1], by_id[p["b"]]["photos"][0], round(float(pr), 3),
                               {k: p.get(k) for k in ("gap", "between", "n_a", "n_b", "d_pos", "d_size", "app_best", "app_body_max", "app_fin_max", "n_inst_max")})
                              for p, pr in zip(dd["pairs"], prob) if (p["a"], p["b"]) in acc and p.get("same") is False]
    return rep, clusters


def audit(dd: dict, clusters: list[dict]) -> dict:
    """The test on true small clusters: every 1.5-s burst that holds several catalog animals (a multi-whale scene) and
    what the safe clustering made of it, plus every impure safe cluster with its animals."""
    from .bursts import make_bursts

    labels = dd["labels"]; feats = dd["feats"]
    ts = json.loads((dd["dir"] / "timestamps.json").read_text())
    cl_of = {s: c["id"] for c in clusters for s in c["main_photos"]}
    lab = lambda s: labels.get(s.split("__")[-1])
    multi = []
    for b in make_bursts(ts, 1.5):
        ids = Counter(lab(s) for s in b["photos"] if lab(s)); ids.pop(None, None)
        if len(ids) < 2:
            continue
        cls = defaultdict(set)
        for s in b["photos"]:
            if lab(s) and s in cl_of:
                cls[cl_of[s]].add(lab(s))
        multi.append({"burst": b["id"], "photos": len(b["photos"]), "animals": dict(ids), "clusters": len(cls), "mixed_clusters": sum(1 for v in cls.values() if len(v) > 1),
                      "singletons": sum(1 for s in b["photos"] if s in cl_of and len(clusters[cl_of[s]]["main_photos"]) == 1)})
    impure = [{"cluster": c["id"], "photos": len(c["main_photos"]), "catalog": c.get("catalog"), "links": c["links"], "first": c["photos"][0], "last": c["photos"][-1]}
              for c in clusters if len(c.get("catalog") or {}) > 1]
    return {"multi_whale_bursts": len(multi), "multi_whale_bursts_split_pure": sum(1 for m in multi if m["mixed_clusters"] == 0), "multi_whale_photos": sum(m["photos"] for m in multi),
            "impure_clusters": len(impure), "bursts": multi, "impure": impure}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("fit", "sweep", "report"):
        s = sub.add_parser(name)
        s.add_argument("--root", type=Path, required=True); s.add_argument("--days", type=Path, required=True); s.add_argument("--labels", type=Path, required=True)
        s.add_argument("--gap", type=float, default=1.5); s.add_argument("--cost", type=float, default=1.5)
        s.add_argument("--window", type=float, default=300.0); s.add_argument("--skip", type=int, default=3)
        s.add_argument("--strict", action="store_true", help="the automatic tier: refuse links whose end frames hold several whale instances")
        s.add_argument("--filter", action="store_true", help="leave out the instances of DAY/safe/not_whale.json (link_scores not-whale)")
        if name == "fit":
            s.add_argument("--out", type=Path, required=True); s.add_argument("--split", default="train"); s.add_argument("--kind", default="gbt", choices=["gbt", "logit"])
            s.add_argument("--importance-split", default="val")
        else:
            s.add_argument("--model", type=Path, required=True); s.add_argument("--splits", default="val,eval"); s.add_argument("--out", type=Path, default=None)
        if name == "sweep":
            s.add_argument("--thresholds", default="0.5,0.6,0.7,0.8,0.9,0.95,0.98,0.99")
        if name == "report":
            s.add_argument("--threshold", type=float, required=True); s.add_argument("--baselines", action="store_true")
            s.add_argument("--write", action="store_true", help="write DAY/safe/clusters.json")
    s = sub.add_parser("cluster", help="one day, labels optional -> DAY/safe/clusters.json")
    s.add_argument("--day-dir", type=Path, required=True); s.add_argument("--model", type=Path, required=True); s.add_argument("--threshold", type=float, required=True)
    s.add_argument("--labels", type=Path, default=None); s.add_argument("--gap", type=float, default=1.5); s.add_argument("--cost", type=float, default=1.5)
    s.add_argument("--window", type=float, default=300.0); s.add_argument("--skip", type=int, default=3); s.add_argument("--strict", action="store_true")
    s.add_argument("--filter", action="store_true")
    s.add_argument("--guard-fused", action="store_true", help="(with --strict) no strict link whose end instance holds two whales by flag_G (cluster/fused_guard.py, rule B_G)")
    s.add_argument("--guard-workers", type=int, default=8)
    a = p.parse_args(argv)
    if a.cmd == "cluster":
        if a.guard_fused and not a.strict:
            p.error("--guard-fused guards the strict tier: it needs --strict")
        model = load_model(a.model)
        dd = day_data(a.day_dir.parent, a.day_dir.name, a.labels, a.gap, a.cost, a.window, a.skip, a.filter)
        guard, ginfo = None, None
        if a.guard_fused:
            from .fused_guard import flagged_keys
            guard, ginfo = flagged_keys(a.day_dir, a.guard_workers)
        rep, clusters = evaluate(dd, model, a.threshold, a.window, a.strict, with_suggestions=True, guard=guard)
        if ginfo is not None:
            ginfo["refused_candidates"] = dd.get("guard_refused", 0); rep["guard_fused"] = ginfo
        rep.update({"gap": a.gap, "cost": a.cost, "window": a.window, "skip": a.skip, "model": str(a.model)})
        if dd["labels"]:
            rep["audit"] = audit(dd, clusters)
        (a.day_dir / "safe").mkdir(exist_ok=True)
        (a.day_dir / "safe" / "clusters.json").write_text(json.dumps({"report": rep, "tracklets": dd["tracks"], "clusters": clusters, "suggestions": dd.get("suggestions", [])}, indent=1))
        n_multi = sum(1 for c in clusters if len(c["photos"]) > 1)
        print(f"{a.day_dir.name}: {len(dd['tracks'])} tracklets -> {len(clusters)} clusters ({n_multi} with several photos), {rep['links']} links" +
              (f"; impure photos {rep['metrics']['photos_in_impure_clusters']} / {rep['metrics']['photos']}, clusters/animal {rep['metrics']['clusters_per_animal']}" if rep.get("metrics") else "") +
              (f"; guard {ginfo['rule']}: {ginfo['flagged']} flagged instances, {ginfo['refused_candidates']} strict candidates refused" if ginfo else ""))
        return 0
    days_js = json.loads(a.days.read_text())

    def load_split(split):
        return [day_data(a.root, day, a.labels, a.gap, a.cost, a.window, a.skip, a.filter) for day in days_js[split]]

    if a.cmd == "fit":
        data = load_split(a.split); pairs = [q for dd in data for q in dd["pairs"]]
        model = fit_model(pairs, TRACK_FEATURES, a.kind)
        model.update({"split": a.split, "gap": a.gap, "cost": a.cost, "window": a.window, "skip": a.skip, "stage": "tracklets"})
        a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_bytes(pickle.dumps(model))
        print(f"{a.kind} fitted on {model['n_pairs']} labelled tracklet pairs ({model['n_same']} same) of {a.split} -> {a.out}")
        if a.importance_split != "none":
            vp = [q for dd in load_split(a.importance_split) for q in dd["pairs"]]
            from sklearn.metrics import roc_auc_score
            rows = [q for q in vp if q.get("same") is not None]
            print(f"  {a.importance_split} pair AUC {roc_auc_score([q['same'] for q in rows], predict(model, rows)):.4f} on {len(rows)} pairs; permutation importance (AUC drop):")
            for name, drop in importance(model, vp)[:16]:
                print(f"    {name:16s} {drop:+.4f}")
        return 0
    model = load_model(a.model)
    cols = ("clusters", "impure_clusters", "photos_in_impure_clusters", "pair_precision", "clusters_per_animal", "oracle_clusters_per_animal", "ari")
    if a.cmd == "sweep":
        ths = [float(t) for t in a.thresholds.split(",")]; table = {}
        for split in a.splits.split(","):
            data = load_split(split)
            s1 = [cluster_metrics([t["main_photos"] for t in dd["tracks"] if t["main_photos"]], dd["labels"], dd["feats"], a.window) for dd in data]
            print(f"== {split}: stage-one tracklets: impure clusters {sum(m['impure_clusters'] for m in s1)}, photos in impure {sum(m['photos_in_impure_clusters'] for m in s1)} / {sum(m['photos'] for m in s1)}, clusters/animal {np.mean([m['clusters_per_animal'] for m in s1]):.2f}")
            print(f"   {'thr':5s} {'impure_cl':>9s} {'impure_ph':>9s} {'ph_frac':>8s} {'pair_prec':>9s} {'cl/animal':>9s} {'oracle':>7s} {'ari':>6s} | single-whale photos: {'impure_ph':>9s} {'pair_prec':>9s}")
            for th in ths:
                ms = [evaluate(dd, model, th, a.window, a.strict)[0] for dd in data]
                m = [r["metrics"] for r in ms]; s = [r["metrics_single"] for r in ms]
                row = {"impure_clusters": sum(x["impure_clusters"] for x in m), "photos_in_impure": sum(x["photos_in_impure_clusters"] for x in m),
                       "impure_fraction": round(sum(x["photos_in_impure_clusters"] for x in m) / max(sum(x["photos"] for x in m), 1), 4),
                       "pair_precision_mean": round(float(np.mean([x["pair_precision"] for x in m])), 4), "clusters_per_animal_mean": round(float(np.mean([x["clusters_per_animal"] for x in m])), 2),
                       "oracle_mean": round(float(np.mean([x["oracle_clusters_per_animal"] for x in m])), 2), "ari_mean": round(float(np.mean([x["ari"] for x in m])), 4),
                       "single_photos_in_impure": sum(x["photos_in_impure_clusters"] for x in s), "single_pair_precision_mean": round(float(np.mean([x["pair_precision"] for x in s])), 4)}
                table[f"{split}@{th}"] = row
                print(f"   {th:5.2f} {row['impure_clusters']:9d} {row['photos_in_impure']:9d} {row['impure_fraction']:8.4f} {row['pair_precision_mean']:9.4f} {row['clusters_per_animal_mean']:9.2f} {row['oracle_mean']:7.2f} {row['ari_mean']:6.3f} | {row['single_photos_in_impure']:>29d} {row['single_pair_precision_mean']:9.4f}")
        if a.out:
            a.out.write_text(json.dumps(table, indent=1))
        return 0
    if a.cmd == "report":
        table = {}
        print(f"{'day':11s} {'split':5s} {'auc':6s} " + " ".join(f"{c[:14]:>14s}" for c in cols))
        for split in a.splits.split(","):
            for dd in load_split(split):
                rep, clusters = evaluate(dd, model, a.threshold, a.window, a.strict, with_suggestions=a.write)
                if a.write and dd["labels"]:
                    rep["audit"] = audit(dd, clusters)
                m = rep["metrics"]; print(f"{dd['day']:11s} {split:5s} {str(rep.get('pair_auc')):6s} " + " ".join(f"{str(m.get(c)):>14s}" for c in cols))
                if rep.get("audit"):
                    au = rep["audit"]; print(f"  audit: {au['multi_whale_bursts']} multi-whale bursts ({au['multi_whale_photos']} photos), {au['multi_whale_bursts_split_pure']} split into pure clusters; {au['impure_clusters']} impure clusters; {len(dd.get('suggestions', []))} suggestions")
                print(f"  {'stage-one tracklets':22s}       " + " ".join(f"{str(rep['stage1'].get(c)):>14s}" for c in cols))
                row = {"split": split, "pair_auc": rep.get("pair_auc"), "safe": m, "safe_single": rep["metrics_single"], "stage1": rep["stage1"], "wrong_links": rep.get("wrong_links", [])}
                if a.baselines:
                    row["baselines"] = baselines(dd["dir"], dd["feats"], dd["labels"], a.window)
                    for name, b in row["baselines"].items():
                        print(f"  {name:22s}       " + " ".join(f"{str(b.get(c)):>14s}" for c in cols))
                if a.write:
                    (dd["dir"] / "safe").mkdir(exist_ok=True)
                    rep.update({"gap": a.gap, "cost": a.cost, "window": a.window, "skip": a.skip, "model": str(a.model)})
                    (dd["dir"] / "safe" / "clusters.json").write_text(json.dumps({"report": rep, "tracklets": dd["tracks"], "clusters": clusters, "suggestions": dd.get("suggestions", [])}, indent=1))
                table[dd["day"]] = row
        if a.out:
            a.out.write_text(json.dumps(table, indent=1))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
