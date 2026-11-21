"""Scoring of live_e2e.py (see its docstring and RESULTS/live_e2e/rule.json). The order-of-work simulation (prep, feats,
run_of, simulate, the blend fit) is copied from the 2026-11-12 scratch scripts sim2.py / blend.py that produced
labeling_model.json "blend" chosen_on; the pair measures use label_assist.py's own functions (imported read-only).
The page's backend is imported read-only for its stacks, matrices and the view-match code."""
from __future__ import annotations

import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from . import CACHE, DAYS, ROOT, day_list
from .live_e2e import OUT, SEEDS, VAL, centre_emb, fin_up, regions
from ..cluster import label_assist as LA

WEB = ROOT.parent / "Whale-Review-Web"
sys.path.insert(0, str(WEB))
from backend import labeling as L  # noqa: E402  (read-only: load_day only reads the day folders and existing caches)

L61 = math.log1p(60.0)
GRE = re.compile(r"_(G\d+)A\d+$")
THR = [0.5, 0.6, 0.8, 0.85, 0.9]
SA_THR = 0.85


# ---- inputs ------------------------------------------------------------------------------------------------------
def shown_sets(d: Path) -> dict:
    """The page's `shown` (view classifier, stale boxes dropped as load_day does)."""
    vclass, _ = L._view_class(d)
    for line in open(d / "parts_hr" / "index.jsonl"):
        r = json.loads(line)
        for i in r.get("instances", []):
            vc = vclass.get(i["key"])
            if vc and not L._same_box(vc[3], i.get("bbox_2560")):
                vclass.pop(i["key"], None)
    return {"fluke": {k for k, v in vclass.items() if v[2] & L.PART_BIT["fluke"]}, "fin": {k for k, v in vclass.items() if v[2] & L.PART_BIT["fin"]}}


def arm_inputs(day: str, arm: str, emb_live: dict, shown: dict) -> tuple[dict, dict | None, dict]:
    """-> (embeddings for P and S.A, allow sets for P (None = all), allow sets for the ranking)."""
    base, seed = (arm[:2], arm[2:] or None)
    emb = dict(emb_live)
    if base in ("L3", "L4"):
        fin = dict(fin_up(day)); c = centre_emb(day, seed)
        fin.update({k: v / np.linalg.norm(v) for k, v in c.items() if k in fin})
        emb["fin"] = fin
    rank_allow = {"fluke": shown["fluke"], "fin": shown["fin"]}
    p_allow = None
    if base == "L1":
        p_allow = {"fluke": shown["fluke"], "fin": shown["fin"]}
    if base in ("L2", "L4"):
        R = regions(day, seed); w3 = {k for k, r in R.items() if r.get("reg") is not None}
        p_allow = {"fin": w3}; rank_allow = {"fluke": shown["fluke"], "fin": shown["fin"] & w3}
    return emb, p_allow, rank_allow


def la_app(clusters: list, emb: dict, allow: dict | None) -> dict:
    """label_assist.load_stacks' appearance matrices with an optional per-part allow set."""
    n = len(clusters); app = {}
    for part in LA.PARTS:
        E = emb.get(part) or {}; ok = (allow or {}).get(part); rows, own = [], []
        for i, c in enumerate(clusters):
            for k in c["keys"]:
                if k in E and (ok is None or k in ok):
                    rows.append(E[k]); own.append(i)
        A = np.full((n, n), np.nan, np.float32)
        if rows:
            own = np.asarray(own); M = np.asarray(rows, np.float32); starts = np.flatnonzero(np.r_[True, own[1:] != own[:-1]]); ids = own[starts]
            A[np.ix_(ids, ids)] = np.maximum.reduceat(np.maximum.reduceat(M @ M.T, starts, axis=0), starts, axis=1)
        app[part] = A
    return app


def page_P(stacks, app, pseq, coef, icpt) -> np.ndarray:
    """labeling._pair_matrix's P (uint8) with given weights."""
    n = len(stacks); w = dict(zip(LA.PAIR_NAMES, coef))
    t0 = np.array([s["t0"] for s in stacks]); t1 = np.array([s["t1"] for s in stacks])
    gap = np.maximum(0.0, np.maximum(t0[:, None], t0[None, :]) - np.minimum(t1[:, None], t1[None, :])); lg = np.log1p(gap)
    z = np.full((n, n), float(icpt), np.float64) + w["log_gap"] * lg + w["log_gap_cap60"] * np.minimum(lg, L61)
    for part in L.PARTS:
        A = app.get(part)
        if A is None:
            continue
        has = np.isfinite(A); z += np.where(has, w[f"app_{part}"] * np.nan_to_num(A) + w[f"has_{part}"], 0.0)
    Pq = np.zeros((n, n), np.float32)
    for (a, b), p in pseq.items():
        Pq[a, b] = Pq[b, a] = max(Pq[a, b], p)
    z += np.where(Pq > 0, w["p_seq"] * Pq + w["has_p_seq"], 0.0)
    P = 1.0 / (1.0 + np.exp(-z)); np.fill_diagonal(P, 0.0)
    return np.round(P * 255).astype(np.uint8)


def load_page(day: str) -> dict:
    """The page's day as load_day builds it with the module's current settings (read-only)."""
    L._DAYS.pop(day, None); D = L.load_day(day); L._DAYS.pop(day, None)
    z = json.loads((D["dir"] / "safe" / "clusters.json").read_text()); by_cl = {s["cl"]: s["id"] for s in D["stacks"] if s["cl"] is not None}; pseq = {}
    for g in z.get("suggestions", []):                  # as load_day (D["pseq"] is rounded to 3 decimals)
        if g.get("p") is not None and g["a"] in by_cl and g["b"] in by_cl:
            a, b = sorted((by_cl[g["a"]], by_cl[g["b"]])); pseq[(a, b)] = max(pseq.get((a, b), 0.0), float(g["p"]))
    bits = []
    for s in D["stacks"]:
        b = 0
        for p, k in s["items"]:
            if k >= 0:
                b |= D["photos"][p]["parts"][k]
        bits.append(b)
    return dict(D=D, pseq=pseq, bits=bits, emb=L._embeddings(D["dir"]))


def code2f(C: np.ndarray) -> np.ndarray:
    C = np.asarray(C); return np.where(C > 0, (C.astype(np.float32) - 1.0) / 254.0, np.nan).astype(np.float32)


# ---- the order-of-work simulation (sim2.py / blend.py of 2026-11-12, copied) -------------------------------------
def prep(photos, stacks, P8, F, Dn, B, Q, bits):
    ph, st, n = photos, stacks, len(stacks); truth = []
    for s in st:
        c = Counter(ph[p].get("cat") for p, k in s["items"] if k <= 0 and ph[p].get("cat")); truth.append(c.most_common(1)[0][0] if c else None)
    starts, groups, cur = [0], [], None
    for i, p in enumerate(ph):
        m = GRE.search(p["cat"]) if p.get("cat") else None
        if not m:
            continue
        if cur is None:
            cur = m.group(1); continue
        if m.group(1) != cur:
            groups.append(int(cur[1:])); starts.append(i); cur = m.group(1)
    groups.append(int(cur[1:]) if cur else 1)
    seg = np.zeros(len(ph), int)
    for j, a in enumerate(starts):
        seg[a:] = j
    segs = [{"idx": j, "group": g, "stacks": []} for j, g in enumerate(groups)]; sseg = []
    for s in st:
        j = int(seg[s["items"][0][0]]); sseg.append(j)
        if s["kind"] != "extra":
            segs[j]["stacks"].append(s["id"])
    byp = defaultdict(list)
    for s in st:
        for p, k in s["items"]:
            byp[p].append(s["id"])
    conf = [set() for _ in st]
    for ids in byp.values():
        for a in ids:
            for b in ids:
                if a != b:
                    conf[a].add(b)
    t0 = np.array([s["t0"] for s in st]); t1 = np.array([s["t1"] for s in st])
    gap = np.maximum(0.0, np.maximum(t0[:, None], t0[None, :]) - np.minimum(t1[:, None], t1[None, :]))
    return dict(n=n, truth=truth, segs=segs, sseg=sseg, conf=conf, P=P8.astype(np.float32) / 255, F=F, Dn=Dn, B=B, Q=Q, gap=gap,
                bits=bits, miss=[ph[s["items"][0][0]].get("miss") for s in st], st=st)


def feats(X, c, stacks):
    idx = np.asarray(stacks); f = X["F"][c, idx]; d = X["Dn"][c, idx]; b = X["B"][c, idx]
    mx = lambda v: float(np.nanmax(v)) if np.isfinite(v).any() else np.nan
    return dict(f=mx(f), d=mx(d), b=mx(b), q=float(X["Q"][c, idx].max()), g=float(X["gap"][c, idx].min()), P=float(X["P"][c, idx].max()), conflict=any(t in X["conf"][c] for t in stacks))


def run_of(X, g, head, owner, others, tau=0.9, maxrun=8):
    P = X["P"]; cur = [head]; run = []; gs = g["stacks"]; i0 = gs.index(head)

    def tied(c):
        if any(c in X["conf"][x] for x in cur):
            return False
        p = max(P[x, c] for x in cur)
        return p >= tau and not (others and max(P[c, t] for t in others) > p)
    for c in gs[i0 + 1:]:
        if len(run) >= maxrun:
            break
        if owner[c] is not None:
            continue
        if (X["miss"][c] or 0) > 0 or not tied(c):
            break
        run.append(c); cur.append(c)
    prev = head
    for c in reversed(gs[:i0]):
        if len(run) >= maxrun:
            break
        if owner[c] is not None:
            continue
        if (X["miss"][prev] or 0) > 0 or not tied(c):
            break
        run.append(c); cur.append(c); prev = c
    return run


def simulate_order(X, score):
    """-> per-segment Counters, stack-animal rows, per head stack {stack: first right (bool)} for the stacks named."""
    segT = {}; owner = [None] * X["n"]; animals = {}; by_cat = {}; rows = []; named = {}
    for g in X["segs"]:
        T = Counter()
        scope = lambda: [a for a in animals.values() if a["seg"] == g["idx"] or a["group"] == g["group"]]
        while True:
            opened = [c for c in g["stacks"] if owner[c] is None and X["truth"][c] is not None]
            if not opened:
                break
            sc = [a for a in scope() if a["stacks"]]

            def top(c):
                r = [(score(fe), a) for a in sc for fe in [feats(X, c, a["stacks"])]]
                return max((x[0] * (0.15 if feats(X, c, x[1]["stacks"])["conflict"] else 1) for x in r), default=0.0)
            fl = [c for c in opened if X["bits"][c] & 4]; fi = [c for c in opened if X["bits"][c] & 2 and not X["bits"][c] & 4]
            if fl:
                head = fl[0]; phase = "fluke"
            elif fi:
                head = fi[0]; phase = "fin"
            else:
                head = max(opened, key=lambda c: (top(c), -c)) if sc else opened[0]; phase = "rest"
            truth = X["truth"][head]
            if not sc:
                a = {"aid": len(animals) + 1, "seg": g["idx"], "group": g["group"], "stacks": [head], "cat": truth, "n": 1}; animals[a["aid"]] = a; by_cat[truth] = a; owner[head] = a["aid"]; T["auto"] += 1; continue
            r = []
            for a in sc:
                fe = feats(X, head, a["stacks"]); s = score(fe) * (0.15 if fe["conflict"] else 1); r.append((s, -a["n"], a, fe))
            r.sort(key=lambda x: (-x[0], -x[1]))
            others = [t for a in sc if a is not r[0][2] for t in a["stacks"]]
            run = [c for c in run_of(X, g, head, owner, others) if X["truth"][c] == truth or X["truth"][c] is None]
            a = by_cat.get(truth); T["decisions"] += 1; T[phase] += 1; T["run"] += len(run)
            for s_, _, an, fe in r:
                rows.append((fe, int(an["cat"] == truth)))
            if a is None or a not in [x[2] for x in r]:
                if a is None:
                    a = {"aid": len(animals) + 1, "seg": g["idx"], "group": g["group"], "stacks": [], "cat": truth, "n": 1 + max((x["n"] for x in sc), default=0)}; animals[a["aid"]] = a; by_cat[truth] = a
                    T["new"] += 1
                else:
                    T["deeper"] += 1; named[head] = False
            else:
                pos = [x[2] for x in r].index(a) + 1; T["first" if pos == 1 else "top3" if pos <= 3 else "deeper"] += 1; named[head] = pos == 1
            for c in [head] + run:
                owner[c] = a["aid"]; a["stacks"].append(c)
        segT[g["idx"]] = T
    return segT, rows, named


def misc_x(fe):
    lg = math.log1p(fe["g"]); return [0.0 if math.isnan(fe["b"]) else fe["b"], 0.0 if math.isnan(fe["b"]) else 1.0, fe["q"], 1.0 if fe["q"] > 0 else 0.0, lg, min(lg, L61)]


def blend_fit(rows):
    """blend.py fit(): fluke / dorsal one-feature logistics and the 'other' logistic, C=10."""
    y = np.array([r[1] for r in rows]); fes = [r[0] for r in rows]; M = {}
    for k in ("f", "d"):
        m = np.array([not math.isnan(fe[k]) for fe in fes]); lr = LogisticRegression(C=10, max_iter=2000).fit(np.array([[fe[k]] for fe in fes])[m], y[m])
        M[k] = (float(lr.coef_[0][0]), float(lr.intercept_[0]))
    lr = LogisticRegression(C=10, max_iter=2000).fit(np.array([misc_x(fe) for fe in fes]), y); M["m"] = (lr.coef_[0].astype(float), float(lr.intercept_[0]))
    return M


def frozen_blend():
    b = L.MODEL["blend"]; return {"f": (b["fluke"]["coef"], b["fluke"]["intercept"]), "d": (b["dorsal"]["coef"], b["dorsal"]["intercept"]), "m": (np.asarray(b["other"]["coef"], float), b["other"]["intercept"])}


def blend_score(M, w=(3, 2, 1)):
    sig = lambda z: 1.0 / (1.0 + math.exp(-z))

    def s(fe):
        pm = sig(float(np.dot(M["m"][0], misc_x(fe))) + M["m"][1]); num = w[2] * pm; den = w[2]
        if not math.isnan(fe["f"]):
            num += w[0] * sig(M["f"][0] * fe["f"] + M["f"][1]); den += w[0]
        if not math.isnan(fe["d"]):
            num += w[1] * sig(M["d"][0] * fe["d"] + M["d"][1]); den += w[1]
        return num / den
    return s


LINK = lambda fe: fe["P"]


# ---- label_assist-level (pair) measures ------------------------------------------------------------------------------
def sim_top(D, P):
    """label_assist.simulate at tau 1.5 with the per-stack outcome: {stack: (first, top3)} for the stacks with their animal present; per-scope counters."""
    true = D["true"]; per = {}; segT = {}
    for name, sc in LA.scopes(D).items():
        sc = sorted(sc, key=lambda i: (D["clusters"][i]["start"], i)); animals = {}; T = Counter()
        for i in sc:
            names = list(animals); score = [max(P[i, j] for j in animals[a]) for a in names]
            order = [names[x] for x in np.argsort(-np.asarray(score))] if names else []
            if true[i] in animals:
                r = order.index(true[i]) + 1; per[i] = (r == 1, r <= 3); T["existing"] += 1; T["first"] += r == 1; T["top3"] += r <= 3
            animals.setdefault(true[i], []).append(i)
        segT[name] = T
    return per, segT


def pair_rows(D, P):
    """(scope, a, b, same, P) for every pair inside a scope, as pair_training."""
    out = []
    for name, sc in LA.scopes(D).items():
        for x in range(len(sc)):
            for y in range(x + 1, len(sc)):
                a, b = sc[x], sc[y]; out.append((name, a, b, D["true"][a] == D["true"][b], float(P[a, b])))
    return out


def hints(D, app, merge) -> list:
    """(scope_a, animal_a, scope_b, animal_b, S.A, same) for unit pairs across scopes with S.A >= 0.85 (uint8 as the page)."""
    best = np.full(app["fin"].shape, np.nan, np.float32)
    for part in LA.PARTS:
        best = np.fmax(best, app[part])
    A8 = np.round(np.clip(np.nan_to_num(best, nan=0.0), 0.0, 1.0) * 255) / 255
    units = defaultdict(list)
    for name, sc in LA.scopes(D).items():
        for i in sc:
            units[(name, D["true"][i])].append(i)
    keys = sorted(units); out = []
    canon = lambda a: merge.get(a, a)
    for x in range(len(keys)):
        for y in range(x + 1, len(keys)):
            (sa, aa), (sb, ab) = keys[x], keys[y]
            if sa == sb:
                continue
            m = float(A8[np.ix_(units[keys[x]], units[keys[y]])].max())
            if m >= SA_THR:
                out.append((sa, aa, sb, ab, round(m, 3), canon(aa) == canon(ab)))
    return out


# ---- run --------------------------------------------------------------------------------------------------------
def la_days(tag: str) -> dict:
    return {d: LA.load_stacks(DAYS / d) for d in VAL} if tag == "v3" else None


def fit_pair(Ds: list, apps: list):
    X, y = [], []
    for D, app in zip(Ds, apps):
        D2 = dict(D, app=app); x, yy = LA.pair_training(D2); X.append(x); y.append(yy)
    lr = LogisticRegression(C=1.0, max_iter=3000).fit(np.concatenate(X), np.concatenate(y))
    return lr.coef_[0].astype(float), float(lr.intercept_[0])


def touched_sets(day, D_la, page, shown, fin_live):
    """stacks (label_assist index / page id) holding a fin fingerprint of an instance the view classifier does not call dorsal."""
    bad = {k for k in fin_live if k not in shown["fin"]}
    la = {i for i, c in enumerate(D_la["clusters"]) if any(k in bad for k in c["keys"])}
    pg = {s["id"] for s, keys in zip(page["D"]["stacks"], page["D"]["keys_of"]) if any(k in bad for k in keys)}
    has_fin = {i for i, c in enumerate(D_la["clusters"]) if any(k in fin_live for k in c["keys"])}
    return la, pg, has_fin


def repro():
    """The recorded numbers from the inputs they were made with."""
    out = {}
    # the "pair" weights were fitted before the safe clusters were re-made (2026-11-11 evening; safe/clusters_v1.json =
    # before): a shadow day folder with links to cluster_v3, timestamps.json and clusters_v1.json as safe/clusters.json
    from .live_e2e import XCACHE
    sh = XCACHE / "shadow_v1"
    for d in VAL:
        (sh / d / "safe").mkdir(parents=True, exist_ok=True)
        for src, dst in ((DAYS / d / "cluster_v3", sh / d / "cluster_v3"), (DAYS / d / "timestamps.json", sh / d / "timestamps.json"), (DAYS / d / "safe" / "clusters_v1.json", sh / d / "safe" / "clusters.json")):
            if not dst.exists():
                dst.symlink_to(src.resolve())
    Dv1 = {d: LA.load_stacks(sh / d) for d in VAL}
    T = Counter()
    for d in VAL:
        cf, ic = fit_pair([Dv1[x] for x in VAL if x != d], [Dv1[x]["app"] for x in VAL if x != d]); T.update(LA.simulate(Dv1[d], LA.pair_prob(Dv1[d], cf, ic)))
    cf, ic = fit_pair([Dv1[x] for x in VAL], [Dv1[x]["app"] for x in VAL])
    out["pair_lodo_clusters_v1_cluster_v3"] = dict(top1=round(T["first"] / T["existing"], 3), top3=round(T["top3"] / T["existing"], 3), stacks=T["stacks"], photos=T["photos"], animals=T["new"],
                                                   refit_all_val=[round(float(c), 4) for c in cf] + [round(ic, 4)], live=list(L.MODEL["pair"]["coef"]) + [L.MODEL["pair"]["intercept"]])
    print("repro pair (clusters_v1, cluster_v3):", out["pair_lodo_clusters_v1_cluster_v3"], flush=True)
    Ds = {d: LA.load_stacks(DAYS / d) for d in VAL}                     # cluster_v3 with today's clusters
    T = Counter(); Tf = Counter(); m = L.MODEL["pair"]
    for d in VAL:
        cf, ic = fit_pair([Ds[x] for x in VAL if x != d], [Ds[x]["app"] for x in VAL if x != d])
        T.update(LA.simulate(Ds[d], LA.pair_prob(Ds[d], cf, ic))); Tf.update(LA.simulate(Ds[d], LA.pair_prob(Ds[d], m["coef"], m["intercept"])))
    out["pair_lodo_cluster_v3"] = dict(top1=round(T["first"] / T["existing"], 3), top3=round(T["top3"] / T["existing"], 3), stacks=T["stacks"], photos=T["photos"], animals=T["new"], existing=T["existing"])
    out["pair_frozen_cluster_v3"] = dict(top1=round(Tf["first"] / Tf["existing"], 4), top3=round(Tf["top3"] / Tf["existing"], 4))
    cf_all, ic_all = fit_pair([Ds[x] for x in VAL], [Ds[x]["app"] for x in VAL])
    out["pair_refit_all_val_cluster_v3"] = dict(coef=[round(c, 4) for c in cf_all], intercept=round(ic_all, 4), live=m["coef"], live_intercept=m["intercept"])
    print("repro pair:", out["pair_lodo_cluster_v3"], out["pair_frozen_cluster_v3"], flush=True)
    print("  refit on all val days with cluster_v3:", out["pair_refit_all_val_cluster_v3"], flush=True)
    # blend.py inputs of 2026-11-12 14:15: cluster_v3, chart flags, no gate, float cosines, live P
    save = (L.CUTOUTS, L.VIEW_CLASS)
    L.CUTOUTS, L.VIEW_CLASS = "first", False
    try:
        X = {}
        for d in VAL:
            pg = load_page(d); D = pg["D"]
            app = L._stack_appearance(D["stacks"], D["keys_of"], pg["emb"]); n = len(D["stacks"]); nan = np.full((n, n), np.nan, np.float32)
            Q = np.zeros((n, n), np.float32)
            for (a, b), p in pg["pseq"].items():
                Q[a, b] = Q[b, a] = max(Q[a, b], p)
            X[d] = prep(D["photos"], D["stacks"], D["P"], app.get("fluke", nan), app.get("fin", nan), app.get("body", nan), Q.astype(np.float16).astype(np.float32), pg["bits"])
    finally:
        L.CUTOUTS, L.VIEW_CLASS = save
    res = {}
    for name in ("link", "blend"):
        T = Counter()
        for d in VAL:
            if name == "link":
                sc = LINK
            else:
                rows = [r for x in VAL if x != d for r in simulate_order(X[x], LINK)[1]]; sc = blend_score(blend_fit(rows))
            segT, _, _ = simulate_order(X[d], sc)
            for t in segT.values():
                T.update(t)
        named = T["first"] + T["top3"] + T["deeper"]
        res[name] = dict(first=T["first"], named=named, rate=round(T["first"] / named, 4), decisions=T["decisions"], auto=T["auto"], new=T["new"])
    out["blend_2026_11_12_inputs"] = res
    print("repro blend:", res, flush=True)
    return out


def score(arms: list, boot: int):
    out = {"arms": arms}
    m = L.MODEL["pair"]; merge = {}
    lab = json.loads((ROOT / "model_fine_tuning" / "splits" / "labels_all_days.json").read_text())
    for x in lab.get("merges", []):
        if x["from"][:10] in VAL and x["to"][:10] in VAL:
            merge[x["from"]] = x["to"]
    del lab
    # per day: label_assist stacks (main, labelled) and the page's day (live settings)
    base = {}
    for d in day_list(",".join(VAL)):
        D_la = LA.load_stacks(DAYS / d); pg = load_page(d); dd = pg["D"]["dir"]; shown = shown_sets(dd)
        vm_emb, vm_cal = L._view_match(dd)
        live_fin = pg["emb"].get("fin", {})
        la_t, pg_t, la_fin = touched_sets(d, D_la, pg, shown, live_fin)
        base[d] = dict(D_la=D_la, pg=pg, shown=shown, vm=(vm_emb, vm_cal), la_t=la_t, pg_t=pg_t, la_fin=la_fin)
        print(d, "label_assist stacks", len(D_la["clusters"]), "page stacks", len(pg["D"]["stacks"]), "touched stacks", len(la_t), "/", len(pg_t), "view_match", bool(vm_emb), flush=True)
    # per arm and day: inputs
    I = {}
    for arm in arms:
        for d in VAL:
            b = base[d]; pg = b["pg"]; D = pg["D"]
            emb, p_allow, r_allow = arm_inputs(d, arm, pg["emb"], b["shown"])
            app_la = la_app(b["D_la"]["clusters"], emb, p_allow)
            app_pg = L._stack_appearance(D["stacks"], D["keys_of"], emb, p_allow)
            parts = list(L._pair_matrix(D["stacks"], L._stack_appearance(D["stacks"], D["keys_of"], emb, r_allow), pg["pseq"])[2:])
            vm_emb, vm_cal = b["vm"]; dors = parts[1]
            if vm_emb and L.MODEL.get("blend"):
                sh = {"fluke": b["shown"]["fluke"], "fin": r_allow["fin"]}
                vs = L._stack_appearance(D["stacks"], D["keys_of"], vm_emb, sh)
                dors = L._match_code(parts[1], vs.get("fin"), vm_cal.get("fin"), L.MODEL["blend"].get("dorsal"))
            I[(arm, d)] = dict(app_la=app_la, app_pg=app_pg, F=parts[0], Dn=dors, B=parts[2], Q=parts[3])
            if arm == "L0":       # the page's own matrices must come out
                P8 = page_P(D["stacks"], app_pg, pg["pseq"], m["coef"], m["intercept"])
                ok = dict(P=bool((P8 == D["P"]).all()), F=bool((parts[0] == (D["match"][0] if D["match"] else D["parts"][0])).all()),
                          Dn=bool((dors == (D["match"][1] if D["match"] else D["parts"][1])).all()), B=bool((parts[2] == D["parts"][2]).all()), Q=bool((parts[3] == D["parts"][3]).all()))
                print("  L0 equals the page's matrices", d, ok, flush=True); out.setdefault("L0_equals_page", {})[d] = ok
    # weights per arm / mode / day
    W = {}
    for arm in arms:
        for d in VAL:
            W[(arm, "frozen", d)] = (np.asarray(m["coef"], float), float(m["intercept"]))
            others = [x for x in VAL if x != d]
            W[(arm, "refit", d)] = fit_pair([base[x]["D_la"] for x in others], [I[(arm, x)]["app_la"] for x in others])
    R = {}
    for arm in arms:
        for mode in ("frozen", "refit"):
            res = dict(pairs=[], top={}, topT={}, dec=Counter(), order_seg={}, named={}, named_link={}, order_seg_link={}, hints=[])
            X = {}
            for d in VAL:
                b = base[d]; D_la = dict(b["D_la"], app=I[(arm, d)]["app_la"]); cf, ic = W[(arm, mode, d)]
                P = LA.pair_prob(D_la, cf, ic)
                res["pairs"] += [(d,) + r for r in pair_rows(D_la, P)]
                per, segT = sim_top(D_la, P)
                res["top"].update({(d, i): v for i, v in per.items()}); res["topT"].update({(d, k): v for k, v in segT.items()})
                res["dec"].update(LA.simulate(D_la, P, 0.9))
                res["hints"] += [(d,) + h for h in hints(D_la, I[(arm, d)]["app_la"], merge)]
                pg = b["pg"]; D = pg["D"]; ii = I[(arm, d)]
                P8 = page_P(D["stacks"], ii["app_pg"], pg["pseq"], cf, ic)
                X[d] = prep(D["photos"], D["stacks"], P8, code2f(ii["F"]), code2f(ii["Dn"]), code2f(ii["B"]), ii["Q"].astype(np.float32) / 255, pg["bits"])
            for d in VAL:
                if mode == "frozen":
                    M = frozen_blend()
                else:
                    M = blend_fit([r for x in VAL if x != d for r in simulate_order(X[x], LINK)[1]])
                segT, _, named = simulate_order(X[d], blend_score(M))
                res["order_seg"].update({(d, k): v for k, v in segT.items()}); res["named"].update({(d, k): v for k, v in named.items()})
                segL, _, namedL = simulate_order(X[d], LINK)
                res["order_seg_link"].update({(d, k): v for k, v in segL.items()}); res["named_link"].update({(d, k): v for k, v in namedL.items()})
            R[(arm, mode)] = res
            print(f"{arm} {mode}: done", flush=True)
    return out, R, base


# ---- summaries and bootstrap -------------------------------------------------------------------------------------------
def sign_p(found, lost):
    n = found + lost
    if n == 0:
        return 1.0
    k = min(found, lost); p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return round(min(1.0, 2 * p), 3)


def summarise(out, R, base, arms, boot, rng):
    T = {}
    units_pair = sorted({(r[0], r[1]) for r in R[(arms[0], "frozen")]["pairs"]})
    units_order = sorted(R[(arms[0], "frozen")]["order_seg"])
    out["units"] = {"scopes": len(units_pair), "segments": len(units_order)}
    B_pair = [rng.integers(0, len(units_pair), len(units_pair)) for _ in range(boot)]
    B_ord = [rng.integers(0, len(units_order), len(units_order)) for _ in range(boot)]
    touched_pair = {}
    for d in VAL:
        b = base[d]
        for i in b["la_fin"]:
            for j in b["la_fin"]:
                if i < j and (i in b["la_t"] or j in b["la_t"]):
                    touched_pair[(d, i, j)] = True

    def pair_stats(res, which):
        rows = [r for r in res["pairs"] if which is None or ((r[0], min(r[2], r[3]), max(r[2], r[3])) in touched_pair) == which]
        by = defaultdict(list)
        for r in rows:
            by[(r[0], r[1])].append((r[4], r[5]))
        return by

    def auc_of(by, idx):
        ys, ps = [], []
        for u in idx:
            for y, p in by.get(units_pair[u], []):
                ys.append(y); ps.append(p)
        ys = np.array(ys); return roc_auc_score(ys, ps) if 0 < ys.sum() < len(ys) else np.nan

    def pr_counts(by, t):
        c = np.zeros((len(units_pair), 3))
        for k, u in enumerate(units_pair):
            for y, p in by.get(u, []):
                c[k, 0] += y and p >= t; c[k, 1] += (not y) and p >= t; c[k, 2] += y and p < t
        return c

    def top_counts(res, which):
        c = np.zeros((len(units_pair), 3))
        idx = {u: k for k, u in enumerate(units_pair)}
        # per stack outcomes mapped to their scope
        for d in VAL:
            D = base[d]["D_la"]
            for name, sc in LA.scopes(D).items():
                for i in sc:
                    v = res["top"].get((d, i))
                    if v is None or (which is not None and (i in base[d]["la_t"]) != which):
                        continue
                    k = idx.get((d, name))
                    if k is None:
                        continue
                    c[k, 0] += v[0]; c[k, 1] += v[1]; c[k, 2] += 1
        return c

    def order_counts(res, key, which, named_key):
        c = np.zeros((len(units_order), 4))
        for k, u in enumerate(units_order):
            t = res[key][u]
            if which is None:
                c[k] = [t["first"], t["first"] + t["top3"] + t["deeper"], t["decisions"], t["auto"]]
        if which is not None:      # by head stack, counted in its segment
            idx = {u: k for k, u in enumerate(units_order)}
            for (d, s), ok in res[named_key].items():
                if (s in base[d]["pg_t"]) != which:
                    continue
                k = idx[(d, SEG_OF[(d, s)])]; c[k, 0] += ok; c[k, 1] += 1
        return c

    def ratio_ci(c0, c1, num, den, Bs):
        r = lambda c, ix: c[ix, num].sum() / max(c[ix, den].sum(), 1)
        full = np.arange(len(c0)); diff = [r(c1, ix) - r(c0, ix) for ix in Bs]
        return round(float(r(c1, full)), 4), round(100 * float(r(c1, full) - r(c0, full)), 2), [round(100 * float(np.percentile(diff, 5)), 2), round(100 * float(np.percentile(diff, 95)), 2)]

    for mode in ("frozen", "refit"):
        r0 = R[("L0", mode)]
        for arm in arms:
            r1 = R[(arm, mode)]; row = {}
            for which, tag in ((None, "all"), (True, "touched"), (False, "untouched")):
                b0, b1 = pair_stats(r0, which), pair_stats(r1, which)
                full = np.arange(len(units_pair)); a0, a1 = auc_of(b0, full), auc_of(b1, full)
                da = [auc_of(b1, ix) - auc_of(b0, ix) for ix in B_pair[: boot // 4]]
                da = [x for x in da if np.isfinite(x)]
                npairs = sum(len(v) for v in b1.values()); npos = sum(y for v in b1.values() for y, _ in v)
                row[f"auc_{tag}"] = dict(value=round(float(a1), 4), diff=round(float(a1 - a0), 4), ci=[round(float(np.percentile(da, 5)), 4), round(float(np.percentile(da, 95)), 4)] if da else None, pairs=npairs, same=int(npos))
                for t in THR:
                    c0, c1 = pr_counts(b0, t), pr_counts(b1, t)
                    prec = ratio_ci(np.c_[c0[:, 0], c0[:, 0] + c0[:, 1]], np.c_[c1[:, 0], c1[:, 0] + c1[:, 1]], 0, 1, B_pair)
                    rec = ratio_ci(np.c_[c0[:, 0], c0[:, 0] + c0[:, 2]], np.c_[c1[:, 0], c1[:, 0] + c1[:, 2]], 0, 1, B_pair)
                    row[f"pr_{tag}_{t}"] = dict(precision=prec, recall=rec, flagged=int(c1[:, 0].sum() + c1[:, 1].sum()))
                c0, c1 = top_counts(r0, which), top_counts(r1, which)
                found = sum(1 for k, v in r1["top"].items() if (which is None or (k[1] in base[k[0]]["la_t"]) == which) and v[0] and not r0["top"][k][0])
                lost = sum(1 for k, v in r1["top"].items() if (which is None or (k[1] in base[k[0]]["la_t"]) == which) and not v[0] and r0["top"][k][0])
                row[f"top1_{tag}"] = dict(zip(("value", "diff", "ci"), ratio_ci(c0, c1, 0, 2, B_pair)), n=int(c1[:, 2].sum()), found=found, lost=lost, p=sign_p(found, lost))
                row[f"top3_{tag}"] = dict(zip(("value", "diff", "ci"), ratio_ci(c0, c1, 1, 2, B_pair)))
                for key, nk, nm in (("order_seg", "named", "first_blend"), ("order_seg_link", "named_link", "first_link")):
                    c0, c1 = order_counts(r0, key, which, nk), order_counts(r1, key, which, nk)
                    both = [k for k in r1[nk] if k in r0[nk] and (which is None or (k[1] in base[k[0]]["pg_t"]) == which)]
                    found = sum(1 for k in both if r1[nk][k] and not r0[nk][k]); lost = sum(1 for k in both if not r1[nk][k] and r0[nk][k])
                    row[f"{nm}_{tag}"] = dict(zip(("value", "diff", "ci"), ratio_ci(c0, c1, 0, 1, B_ord)), named=int(c1[:, 1].sum()), found=found, lost=lost, p=sign_p(found, lost))
                    if which is None:
                        dd = ratio_ci(c0, c1, 2, 3, B_ord)      # decisions: sum (ratio against auto is not used; report the counts)
                        row[f"decisions_{nm}"] = dict(value=int(c1[:, 2].sum()), L0=int(c0[:, 2].sum()), per_day=round(float(c1[:, 2].sum()) / 3, 1))
            row["decisions_capture_tau0.9"] = dict(value=r1["dec"]["decisions"], L0=r0["dec"]["decisions"], per_day=round(r1["dec"]["decisions"] / 3, 1))
            h1 = {(h[0], h[1], h[2], h[3], h[4]) for h in r1["hints"]}; h0 = {(h[0], h[1], h[2], h[3], h[4]) for h in r0["hints"]}
            row["same_as"] = dict(hints=len(r1["hints"]), true=sum(1 for h in r1["hints"] if h[6]), new=len(h1 - h0), gone=len(h0 - h1),
                                  by_day={d: sum(1 for h in r1["hints"] if h[0] == d) for d in VAL})
            T[(arm, mode)] = row
    return T


SEG_OF: dict = {}


def main_cmd(a):
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "repro":
        r = repro(); (OUT / "repro.json").write_text(json.dumps(r, indent=1, default=float)); return
    if a.cmd == "tables":
        tables(); return
    if a.cmd == "perday":
        perday(); return
    arms = a.arms.split(",") if a.arms else ["L0", "L1", "L2s0", "L2s1", "L3s0", "L3s1", "L4s0", "L4s1"]
    assert arms[0] == "L0"
    out, R, base = score(arms, a.boot)
    for d in VAL:                               # page stack -> its segment, as prep cuts them
        D = base[d]["pg"]["D"]; n = len(D["stacks"]); z = np.zeros((n, n), np.float32)
        X = prep(D["photos"], D["stacks"], D["P"], z, z, z, z, base[d]["pg"]["bits"])
        for s, j in enumerate(X["sseg"]):
            SEG_OF[(d, s)] = j
    rng = np.random.default_rng(0)
    T = summarise(out, R, base, arms, a.boot, rng)
    res = {"rule": "rule.json", **out, "table": {f"{arm} {mode}": v for (arm, mode), v in T.items()},
           "hints_detail": {f"{arm} {mode}": [list(h) for h in R[(arm, mode)]["hints"]] for arm in arms for mode in ("frozen", "refit")}}
    name = "scores_L0.json" if arms == ["L0"] else "scores.json"
    (OUT / name).write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print("wrote", OUT / name)


def tables(path=None):
    """RESULTS/live_e2e/tables.md from scores.json (numbers as written there)."""
    z = json.loads((OUT / "scores.json").read_text()); T = z["table"]; arms = z["arms"]
    f = lambda v, pct=True: "" if v is None else (f"{100 * v:.1f}" if pct else f"{v:.3f}")
    ci = lambda c: "" if not c else f"[{c[0]:+.1f}, {c[1]:+.1f}]"
    L = [f"# W3c in the live path, part 3: what the page computes (validation days)\n",
         f"Rule: rule.json (fixed before any arm other than L0 was scored). Bootstrap over encounters ({z['units']['scopes']} catalogue scopes for pair / top-1 measures, "
         f"{z['units']['segments']} catalogue segments for the order of work), 2000 resamples (AUC 500), 90 % intervals of the difference to L0 in points. "
         "found / lost = stacks right only in the arm / only in L0 (sign test p). L1's ranking = L0's by construction (its first-choice rows differ only through P's runs and the link score).\n"]

    def block(title, key, kind):
        L.append(f"\n## {title}\n\n| arm | frozen | change | found / lost | refit LODO | change | found / lost |\n|---|---|---|---|---|---|---|")
        for arm in arms:
            cells = []
            for mode in ("frozen", "refit"):
                r = T[f"{arm} {mode}"][key]
                if kind == "auc":
                    cells += [f"{r['value']:.3f}", "" if arm == "L0" else f"{r['diff']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]", f"{r['pairs']} pairs"]
                else:
                    n = r.get("n", r.get("named"))
                    cells += [f"{f(r['value'])}" + (f" ({n})" if n is not None else ""), "" if arm == "L0" else f"{100 * r['diff'] / 100:+.1f} {ci(r['ci'])}", "" if arm == "L0" or "found" not in r else f"{r['found']} / {r['lost']} (p {r['p']})"]
            L.append(f"| {arm} | " + " | ".join(cells) + " |")
    for tag, nm in (("all", "all pairs / stacks"), ("touched", "touched (a fin fingerprint on a non-fin view; small, hints only)"), ("untouched", "untouched")):
        block(f"P: pair AUC inside encounters, {nm}", f"auc_{tag}", "auc")
        block(f"P: stack top-1 (label_assist 'pair' measure), {nm}", f"top1_{tag}", "rate")
        if tag == "all":
            block("P: stack top-3, all", "top3_all", "rate")
        block(f"Step 2: first choice right, blend 3:2:1 (order of work), {nm}", f"first_blend_{tag}", "rate")
        block(f"Step 2: first choice right, link score P (order of work), {nm}", f"first_link_{tag}", "rate")
    L.append("\n## P: precision / recall at the page's thresholds, all pairs (change to L0 in points, 90 % interval)\n")
    for t in THR:
        L.append(f"\n### P >= {t}\n\n| arm | frozen precision | change | frozen recall | change | refit precision | change | refit recall | change |\n|---|---|---|---|---|---|---|---|---|")
        for arm in arms:
            cells = []
            for mode in ("frozen", "refit"):
                r = T[f"{arm} {mode}"][f"pr_all_{t}"]
                for k in ("precision", "recall"):
                    v, dlt, c = r[k]; cells += [f(v), "" if arm == "L0" else f"{dlt:+.1f} {ci(c)}"]
            L.append(f"| {arm} | " + " | ".join(cells) + " |")
    L.append("\n## Decisions on the 3 days (order of work with the blend / with the link score; label_assist.simulate tau 0.9 in capture order)\n\n| arm | frozen blend | frozen link | frozen capture | refit blend | refit link | refit capture |\n|---|---|---|---|---|---|---|")
    for arm in arms:
        cells = []
        for mode in ("frozen", "refit"):
            r = T[f"{arm} {mode}"]; cells += [str(r["decisions_first_blend"]["value"]), str(r["decisions_first_link"]["value"]), str(r["decisions_capture_tau0.9"]["value"])]
        L.append(f"| {arm} | " + " | ".join(cells) + " |")
    L.append("\n## Cross-encounter 'same as' hints (S.A >= 0.85)\n\n| arm | frozen hints (true) | refit hints (true) |\n|---|---|---|")
    for arm in arms:
        a, b = T[f"{arm} frozen"]["same_as"], T[f"{arm} refit"]["same_as"]
        L.append(f"| {arm} | {a['hints']} ({a['true']}) | {b['hints']} ({b['true']}) |")
    rp = json.loads((OUT / "repro.json").read_text())
    L.append("\n## Reproduction of the recorded numbers (repro.json)\n")
    a = rp["pair_lodo_clusters_v1_cluster_v3"]; b = rp["pair_lodo_cluster_v3"]; c = rp["blend_2026_11_12_inputs"]
    L.append(f"- 'pair' val top-1 / top-3 (recorded 0.843 / 0.987, 441 stacks): {a['top1']} / {a['top3']}, {a['stacks']} stacks, on the inputs it was fitted with (safe/clusters_v1.json, cluster_v3); "
             f"refit on all three days = the live weights exactly. With today's clusters and cluster_v3: {b['top1']} / {b['top3']}, {b['stacks']} stacks.")
    L.append(f"- 'blend' chosen_on (recorded 77.2 % / 77.9 % of 276, 315 decisions): blend {100 * c['blend']['rate']:.1f} % / link {100 * c['link']['rate']:.1f} % of {c['blend']['named']}, {c['blend']['decisions']} decisions, on blend.py's inputs of 2026-11-12 14:15 (cluster_v3, chart flags, no gate).")
    L.append("- L0 here = today's page inputs (safe clusters of 2026-11-11, cluster_up of 2026-11-16, view classifier gate 'blend', view_match): P, fluke, dorsal (match), body and sequence matrices byte-equal to labeling.load_day on all three days.")
    pdz = json.loads((OUT / "perday.json").read_text())
    L.append("\n## Within-day pair AUC and top-1 of P, refit leave-one-day-out (perday.json)\n\n| arm | " + " | ".join(f"{d} AUC / top-1" for d in VAL) + " |\n|---|---|---|---|")
    for arm in arms:
        L.append(f"| {arm} | " + " | ".join(f"{pdz[arm][f'{d} refit']['auc']:.3f} / {pdz[arm][f'{d} refit']['top1']:.3f}" for d in VAL) + " |")
    L.append("\nFrozen, same layout:\n\n| arm | " + " | ".join(f"{d} AUC / top-1" for d in VAL) + " |\n|---|---|---|---|")
    for arm in arms:
        L.append(f"| {arm} | " + " | ".join(f"{pdz[arm][f'{d} frozen']['auc']:.3f} / {pdz[arm][f'{d} frozen']['top1']:.3f}" for d in VAL) + " |")
    L.append("\nRefit weight of app_fin (fold without 01-15 / 02-27 / 03-14): " + "; ".join(f"{arm} " + " / ".join(f"{pdz[arm][f'{d} refit weights'][2]:.2f}" for d in VAL) for arm in arms))
    (OUT / "tables.md").write_text("\n".join(L) + "\n"); print("wrote", OUT / "tables.md")


def perday():
    """Follow-up after scores.json: per validation day, stack top-1 and the within-day pair AUC of P, frozen and refit
    (leave the day out), and the refit weights. The pooled refit AUC mixes three differently calibrated day models."""
    m = L.MODEL["pair"]; base = {}; out = {"names": LA.PAIR_NAMES}
    for d in VAL:
        base[d] = (LA.load_stacks(DAYS / d), L._embeddings(DAYS / d), shown_sets(DAYS / d))
    for arm in ["L0", "L1", "L2s0", "L2s1", "L3s0", "L3s1", "L4s0", "L4s1"]:
        apps = {d: la_app(base[d][0]["clusters"], *arm_inputs(d, arm, base[d][1], base[d][2])[:2]) for d in VAL}; row = {}
        for d in VAL:
            o = [x for x in VAL if x != d]; cf, ic = fit_pair([base[x][0] for x in o], [apps[x] for x in o]); D = dict(base[d][0], app=apps[d])
            for mode, (c, i) in (("frozen", (np.asarray(m["coef"]), m["intercept"])), ("refit", (cf, ic))):
                per, _ = sim_top(D, LA.pair_prob(D, c, i)); X, y = LA.pair_training(D); p = 1 / (1 + np.exp(-(X @ np.asarray(c) + i)))
                row[f"{d} {mode}"] = dict(top1=round(float(np.mean([v[0] for v in per.values()])), 3), auc=round(float(roc_auc_score(y, p)), 3))
            row[f"{d} refit weights"] = [round(float(x), 3) for x in cf] + [round(float(ic), 3)]
        out[arm] = row; print(arm, {k: v for k, v in row.items() if "weights" not in k}, flush=True)
    (OUT / "perday.json").write_text(json.dumps(out, indent=1)); print("wrote", OUT / "perday.json")
