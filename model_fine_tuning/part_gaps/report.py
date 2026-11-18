"""The part-model gap table: which kinds of real whale views the part model gets wrong, how much each costs, whether
the renders hold that kind of view, and whether the model fails on renders of it too.

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.part_gaps.report [--model parts_seg_r05_a] [--variant hr] \
        [--arms parts_seg_r05_sc parts_seg_r05_rw parts_seg_r05_w1 parts_seg_r05_s1] \
        [--renders <dataset> ...] [--out experiments/part_gaps/report_<model>.json]

Real side: the stratified sample of real whale instances (sample.json, train + validation days, never the evaluation
days) with AI by-eye labels (labels_ai.txt: one category + attributes per instance, made blind to every model's output).
Every model's fluke / dorsal / other call comes from its own real-photo outputs (DAY/<model>/instances.json) with its
decision rule re-chosen on the TRAINING days in evaluate_real's two-part family (as experiments/realwater/compare.py).
Shares and error rates are weighted by the sampling weights (population estimates); the training-day sample is the
analysis set (ranking), the validation-day sample the reported numbers. Cost = share of real instances x error rate =
the share of all real instances that are this category AND called wrong.
Render side: render_eval.py outputs (per rendered whale: meta category, truth parts, the model's pixels), mapped to the
same taxonomy by render_category(); coverage = render share / real share; the model's error on renders of a category
uses the same rule. Missing render outputs are listed, not guessed.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..parts_seg import evaluate_real as ER
from . import common as C

DEFAULT_ARMS = ["parts_seg_r05_sc", "parts_seg_r05_rw", "parts_seg_r05_w1", "parts_seg_r05_s1", "parts_seg_r05_sw",
                "parts_seg_r05_fw", "parts_seg_r05_rwall"]
R = C.FACTORY
BASE_SETS = ("run05", "r06r_test_final", "r06r_test_check", "r06r_test_r3", "r06r_test_r4b", "r06r_test_r5", "r06r_test_r6")
TRAIN_SET = "run05"                       # the render set the r05 models were trained on
# factory checks from r6b on (r06r_test_r6b, _r7, _r8, ...): own column each, not pooled into "run06 tests"
OLD_TESTS = BASE_SETS + ("r06r_test_r2", "r06r_test_r4")


def _check_key(t: str):
    import re
    m = re.match(r"r06r_test_r(\d+)([a-z]*)$", t)
    return (int(m.group(1)), m.group(2)) if m else (999, t)


CHECK_SETS = tuple(sorted((q.parent.parent.name for q in R.glob("r06r_test_r*/dataset/manifest.jsonl")
                           if q.parent.parent.name not in OLD_TESTS and _check_key(q.parent.parent.name)[0] >= 6
                           and _check_key(q.parent.parent.name)[0] < 999), key=_check_key))
DEFAULT_RENDERS = [R / t / "dataset" for t in BASE_SETS + CHECK_SETS]
CAT_ORDER = list(C.CATS)

# what the factory would have to add, per category (hand-written from the by-eye sheets; the cause column of the table
# is decided from the numbers, see cause())
ACTION = {
    "F_up": "keep (fluke raised from behind is well covered)",
    "F_low": "flukes rising low / rolling out flat at the surface, water sheeting off, blades half in the water, seen from behind",
    "F_edge": "flukes seen side-on / single blade raised (tail throw, slap, dive at an angle), with water streams",
    "F_tips": "fluke tips only / sinking",
    "D_tall": "keep; vary fin height / sweep",
    "D_small": "hump + small, stubby, knob-like or hooked fins (the rig's fin is a tall clean blade on a back with no hump)",
    "D_tip": "fin seen end-on from behind on the arched back (dive), far / small fins, fin partly under water",
    "B_back": "backs and tail stocks with no fin in view: arched peduncle from behind, flat awash backs, backs in foam",
    "P_pec": "pectoral fins: raised, slapping, lying white on the surface, with splash; dark top side as well as white",
    "H_head": "heads: head rise / spyhop / lunge from the front with ventral grooves, rostrum with tubercles, blowholes from the front",
    "R_roll": "rolls / belly up / breaches",
    "X_none": "not a whale (splash, blow, birds, poles, land) passing the not-whale filter: negatives for the segmentation",
}


# ---------------------------------------------------------------------------------------------------------- labels
def load_labels(path: Path | None = None) -> dict:
    path = path or (C.OUT / "labels_ai.txt")
    sample = json.loads((C.OUT / "sample.json").read_text())["sample"]
    by_tile = {(s["sheet"], int(s["tile"])): s for s in sample}
    out = {}
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sh, tile, code = line.split(None, 2)
        s = by_tile[(sh, int(tile))]
        out[s["key"]] = dict(s, **C.parse_code(code), code=code)
    relook = C.OUT / "labels_ai_relook.txt"
    if relook.exists():                       # second look at one ambiguous view (see the file's header)
        for line in open(relook):
            line = line.strip()
            if line and not line.startswith("#"):
                k, code = line.split(None, 1)
                out[k].update(**C.parse_code(code), code=code, relooked=True)
    return out


# ---------------------------------------------------------------------------------------------------------- model calls
def rules_for(name: str, rows: list[dict]) -> dict:
    cache = C.CACHE / "rules" / f"{name}.json"
    if cache.exists():
        r = json.loads(cache.read_text())
        if r.get("n_rows") == len(rows):
            return {v: ((tuple(x[0]), x[1])) for v, x in r["rules"].items()}
    rules = {}
    for variant in ("full", "hr"):
        if not any(variant in r["pred"] for r in rows):
            continue
        tf, td = C.choose_rule(rows, variant)
        rules[variant] = (tuple(tf), td)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"n_rows": len(rows), "rules": {v: [list(x[0]), x[1]] for v, x in rules.items()}}))
    return rules


def call_of(pred: dict, rule) -> str:
    return ER.rule_three(pred, rule[0], rule[1])


def wrong_part(pred: dict, truth: str) -> str:
    """the largest part class painted on the instance other than body / the true part (e.g. 'head' for a fin painted head)."""
    s = C.pred_shares(pred)
    cand = {"fluke": s["fluke"], "fin": s["dorsal_fin"], "pec": s["pectoral_fin"], "head": s["head"]}
    t = {"fluke": "fluke", "dorsal": "fin"}.get(truth)
    if t:
        cand.pop(t)
    k = max(cand, key=cand.get)
    return f"{k} {cand[k]:.2f}" if cand[k] >= 0.02 else "body"


def model_outcomes(name: str, labels: dict, variant: str) -> dict | None:
    if C.arm_root(name) is None:
        return None
    rows = C.arm_rows(name, C.TRAIN_DAYS + C.VAL_DAYS)
    rules = rules_for(name, rows)
    if variant not in rules:
        return None
    byk = {r["key"]: r for r in rows}
    out = {}
    for k, lab in labels.items():
        r = byk.get(k)
        if r is None or variant not in r["pred"] or lab["three"] is None:
            continue
        p = r["pred"][variant]
        c = call_of(p, rules[variant])
        out[k] = {"call": c, "ok": int(c == lab["three"]), "conf": f"{lab['three']}->{c}" + ("" if c == lab["three"] else f" ({wrong_part(p, lab['three'])})"),
                  "shares": {kk: round(v, 4) for kk, v in C.pred_shares(p).items()}}
    return {"rule": [list(rules[variant][0]), rules[variant][1]], "outcomes": out, "all_rows": rows, "rules": rules}


# ---------------------------------------------------------------------------------------------------------- statistics
def wstats(items: list[tuple[float, int]], n_boot: int = 1000, seed: int = 0) -> dict:
    """items = (weight, error 0/1); weighted error rate with a bootstrap 95 % interval over instances."""
    if not items:
        return {"n": 0, "err": None, "ci": [None, None], "k": 0}
    w = np.array([x[0] for x in items], float)
    e = np.array([x[1] for x in items], float)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(len(w), size=len(w))
        bs.append((w[i] * e[i]).sum() / w[i].sum())
    return {"n": len(items), "k": int(e.sum()), "err": float((w * e).sum() / w.sum()),
            "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


def split_weights(labels: dict, split: str) -> dict:
    return {k: v["weight"] for k, v in labels.items() if v["split"] == split}


def table_by(labels: dict, outcomes: dict, key, split: str) -> dict:
    """per group: weighted share of real instances (X_none included in the denominator), weighted error, cost."""
    lab = {k: v for k, v in labels.items() if v["split"] == split}
    W = sum(v["weight"] for v in lab.values())
    groups = defaultdict(list)
    for k, v in lab.items():
        for g in key(v):
            groups[g].append(k)
    out = {}
    for g, ks in groups.items():
        share = sum(lab[k]["weight"] for k in ks) / W
        items = [(lab[k]["weight"], 1 - outcomes[k]["ok"]) for k in ks if k in outcomes]
        st = wstats(items)
        out[g] = {"n_labelled": len(ks), "share": share, **st, "cost": share * st["err"] if st["err"] is not None else None}
    return out


def main_confusion(labels, outcomes, cat) -> str:
    c = defaultdict(float)
    for k, v in labels.items():
        if v["cat"] == cat and k in outcomes and not outcomes[k]["ok"]:
            conf = outcomes[k]["conf"]
            base = conf.split(" (")[0]
            part = conf.split(" (")[1].split()[0].rstrip(")") if " (" in conf else ""
            c[f"{base} {part}".strip()] += v["weight"]
    if not c:
        return "-"
    tot = sum(c.values())
    top = sorted(c.items(), key=lambda x: -x[1])[:2]
    return "; ".join(f"{a} {b / tot:.0%}" for a, b in top)


# ---------------------------------------------------------------------------------------------------------- renders
def render_category(rec: dict) -> str:
    """the real-photo taxonomy for one rendered whale, from its truth parts and the shot's pose category."""
    t = rec["truth"]
    whale = sum(t[1:6])
    if whale <= 0:
        return "X_none"
    fl, fin, pec, head = t[1] / whale, t[2] / whale, t[3] / whale, t[4] / whale
    cat = (rec.get("meta_category") or "").lower()
    if cat in ("breach", "roll_belly"):
        return "R_roll"
    if fl >= 0.3:
        if cat in ("fluke_rollout", "fluke_down", "fluke_sinking"):
            return "F_low"
        if cat in ("fluke_edge", "tail_slap", "peduncle_throw"):
            return "F_edge"
        return "F_up"
    if pec >= 0.15:
        return "P_pec"
    if head >= 0.2 and head >= 5 * fin:
        return "H_head"
    if fin > 0 and t[2] * (2560.0 / rec["W"]) ** 2 >= 150:
        return "D_tall"            # run05 / run06 tests: one rig fin, a tall clean blade, no hump (by eye about a third of
                                   # them read as low / small at distance, none stubby, knobby or on a hump): no D_small
    if fin > 0:
        return "D_tip"
    return "B_back"


def load_renders(ds: Path, model: str) -> list[dict] | None:
    tag = ds.parent.name if ds.name == "dataset" else ds.name
    p = C.CACHE / "renders" / tag / f"{model}.jsonl"
    if not p.exists():
        return None
    return [json.loads(l) for l in open(p) if l.strip()]


def render_side(datasets, model: str, rule_by_variant: dict, variant: str) -> dict:
    """per render set: what it holds (coverage.py, all whales, labels only) and how the model does on it (render_eval.py
    outputs of <model>, a sample), both in the real-photo taxonomy."""
    out = {}
    for ds in datasets:
        tag = ds.parent.name if ds.name == "dataset" else ds.name
        cov_p = C.CACHE / "coverage" / f"{tag}.jsonl"
        cov = [json.loads(l) for l in open(cov_p)] if cov_p.exists() else None
        recs = load_renders(ds, model)
        if cov is None and recs is None:
            out[tag] = {"missing": f"run coverage.py --dataset {ds} (and render_eval.py --dataset {ds} --model {model})"}
            continue
        src = cov if cov is not None else recs
        cats = defaultdict(list)
        for r in src:
            if sum(r["truth"][1:6]) == 0:          # no labelled whale pixel (run05 calves are ignore): not counted
                continue
            cats[render_category(r)].append(r["L2560"])
        n = sum(len(v) for v in cats.values())
        d = {"n_whales": n, "coverage_source": "labels (coverage.py)" if cov is not None else "render_eval sample",
             "cats": {c: {"n": len(v), "share": len(v) / n, "median_L2560": float(np.median(v))} for c, v in cats.items()}}
        if recs is None:
            d["model_missing"] = f"no render outputs for {model}: run render_eval.py --dataset {ds} --model {model}"
        else:
            sc = defaultdict(list)
            for r in recs:
                c = render_category(r)
                if C.THREE_OF[c] is None or variant not in r:
                    continue
                # truth three-way of a rendered whale = the same rule applied to its TRUE part pixels (so a head shot with
                # a visible fin counts as dorsal, as the rule would call it); the model is wrong when its call differs
                tc = r["truth"]
                three = call_of({"pixels_near": dict(zip(C.CLASSES, tc)), "pixels": dict(zip(C.CLASSES, tc)), "covered": max(sum(tc), 1)},
                                rule_by_variant[variant])
                cnt = r[variant]
                pred = {"pixels_near": dict(zip(C.CLASSES, cnt)), "pixels": dict(zip(C.CLASSES, cnt)), "covered": max(sum(cnt), 1)}
                sc[c].append(int(call_of(pred, rule_by_variant[variant]) == three))
            d["n_scored"] = sum(len(v) for v in sc.values())
            for c, v in sc.items():
                d["cats"].setdefault(c, {"n": 0, "share": 0.0, "median_L2560": None})
                d["cats"][c]["err"] = 1 - float(np.mean(v))
                d["cats"][c]["n_scored"] = len(v)
        out[tag] = d
    return out


def cause(real_err, render_err, render_n, coverage_train, cat) -> str:
    """coverage_train = coverage in the set the model was trained on (run05 for the r05 models); render_err = the model's
    error on renders of the category (needs >= 15 scored whales to count)."""
    if cat == "X_none":
        return "not a whale (segmentation / filter)"
    if real_err is None:
        return "-"
    if real_err < 0.12:
        return "minor"
    if render_err is not None and render_n >= 15 and render_err >= 0.25:
        return "model / label rule / resolution (fails on renders too)"
    if coverage_train is None or coverage_train < 0.5:
        return "view missing from the training renders (coverage < 0.5)"
    return "appearance / shape (covered, and right on renders, but wrong on real)"


# ---------------------------------------------------------------------------------------------------------- main
def fmt(x, p=2):
    return "-" if x is None else f"{x:.{p}f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="parts_seg_r05_a")
    ap.add_argument("--variant", default="hr", choices=["full", "hr"])
    ap.add_argument("--arms", nargs="*", default=DEFAULT_ARMS)
    ap.add_argument("--renders", nargs="*", type=Path, default=DEFAULT_RENDERS)
    ap.add_argument("--render-model", default=None, help="model whose render outputs to read (default --model)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    labels = load_labels()
    res = {"model": a.model, "variant": a.variant, "labels": {"n": len(labels), "note": "AI by-eye labels (blind to model output), not the researcher's"}}
    base = model_outcomes(a.model, labels, a.variant)
    if base is None:
        raise SystemExit(f"no real-photo outputs for {a.model} ({a.variant})")
    oc = base["outcomes"]
    res["rule"] = base["rule"]
    # overall
    for split in ("train", "val"):
        items = [(labels[k]["weight"], 1 - o["ok"]) for k, o in oc.items() if labels[k]["split"] == split]
        res[f"overall_{split}"] = wstats(items)
    # per category
    cats = {sp: table_by(labels, oc, lambda v: [v["cat"]], sp) for sp in ("train", "val")}
    attrs = {sp: table_by(labels, oc, lambda v: (v["attrs"] or ["none"]) + ["all"], sp) for sp in ("train", "val")}
    sizes = {sp: table_by(labels, oc, lambda v: [v["size"]], sp) for sp in ("train", "val")}
    three = {sp: table_by(labels, oc, lambda v: [str(v["three"])], sp) for sp in ("train", "val")}
    res.update({"by_category": cats, "by_attribute": attrs, "by_size": sizes, "by_three": three})
    res["main_confusion"] = {c: main_confusion(labels, oc, c) for c in C.CATS}
    # sensitivity: the same table without the instances still ambiguous after the second look (attribute a / a2)
    clear = {k: v for k, v in labels.items() if not ({"a", "a2"} & set(v["attrs"]))}
    res["by_category_unambiguous"] = {sp: table_by(clear, oc, lambda v: [v["cat"]], sp) for sp in ("train", "val")}
    # within-category splits that matter for the factory: hump / no hump, splash / dry, size
    res["within"] = {sp: table_by(clear, oc, lambda v: [f"{v['cat']}|{'hump' if 'h' in v['attrs'] else 'nohump'}",
                                                        f"{v['cat']}|{'splash' if {'s1', 's2'} & set(v['attrs']) else 'dry'}",
                                                        f"{v['cat']}|{v['size']}"], sp) for sp in ("train", "val")}
    # pooled real share (both splits, each weighted to its own population) for coverage
    pooled = defaultdict(float)
    tot = sum(v["weight"] for v in labels.values())
    for v in labels.values():
        pooled[v["cat"]] += v["weight"] / tot
    res["real_share_pooled"] = dict(pooled)
    res["real_median_L2560"] = {c: float(np.median([v["L"] for v in labels.values() if v["cat"] == c])) for c in C.CATS
                                if any(v["cat"] == c for v in labels.values())}
    # other variant, size x variant
    other = "full" if a.variant == "hr" else "hr"
    ob = model_outcomes(a.model, labels, other)
    if ob:
        res["by_size_other_variant"] = {sp: table_by(labels, ob["outcomes"], lambda v: [v["size"]], sp) for sp in ("train", "val")}
        res["by_category_other_variant"] = {sp: table_by(labels, ob["outcomes"], lambda v: [v["cat"]], sp) for sp in ("train", "val")}
    # arms
    arms = {}
    arm_names = list(a.arms) + (["parts_seg_r05_a"] if a.model != "parts_seg_r05_a" and "parts_seg_r05_a" not in a.arms else [])
    for name in arm_names:
        if name == a.model:
            continue
        o = model_outcomes(name, labels, a.variant)
        if o is None:
            arms[name] = {"missing": "no real-photo outputs"}
            continue
        paired = {}
        for sp in ("train", "val", "all"):
            for c in list(C.CATS) + ["ALL"]:
                ks = [k for k in oc if k in o["outcomes"] and (sp == "all" or labels[k]["split"] == sp) and (c == "ALL" or labels[k]["cat"] == c)]
                if not ks:
                    continue
                w = np.array([labels[k]["weight"] for k in ks]); d = np.array([o["outcomes"][k]["ok"] - oc[k]["ok"] for k in ks], float)
                rng = np.random.default_rng(0)
                bs = [(w[i] * d[i]).sum() / w[i].sum() for i in (rng.integers(len(ks), size=len(ks)) for _ in range(1000))]
                paired[f"{sp}|{c}"] = {"n": len(ks), "gain": float((w * d).sum() / w.sum()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                       "fixed": int((d > 0).sum()), "broken": int((d < 0).sum())}
        arms[name] = {"paired_vs_base": paired, "rule": o["rule"], "by_category": {sp: table_by(labels, o["outcomes"], lambda v: [v["cat"]], sp) for sp in ("train", "val")},
                      "overall": {sp: wstats([(labels[k]["weight"], 1 - x["ok"]) for k, x in o["outcomes"].items() if labels[k]["split"] == sp]) for sp in ("train", "val")}}
        if "w1" in name:
            arms[name]["note"] = "trained on real TRAINING-day photos (weak labels): its training-day numbers are not a test"
    res["arms"] = arms
    # renders
    rm = a.render_model or a.model
    res["renders"] = render_side(a.renders, rm, base["rules"], a.variant)
    # per-instance outcomes (keys only, no pictures)
    res["instances"] = {k: {"cat": labels[k]["cat"], "attrs": labels[k]["attrs"], "split": labels[k]["split"], "size": labels[k]["size"],
                            "weight": labels[k]["weight"], **o} for k, o in oc.items()}
    res["gap_table"] = gap_rows(res)
    res["coverage_by_eye"] = by_eye_coverage(res)
    out = a.out or (C.OUT / f"report_{a.model}_{a.variant}.json")
    out.write_text(json.dumps(res, indent=1))
    print_report(res, a)
    print("wrote", out)
    return 0


def render_cov(res, cat):
    """(run05 coverage, run06 tests coverage, render error run05, render error run06 tests)"""
    real = res["real_share_pooled"].get(cat, 0)
    r = res["renders"]
    def agg(tags):
        n = sum(r[t]["n_whales"] for t in tags if "n_whales" in r[t])
        if not n:
            return None, None, 0
        c = sum(r[t]["cats"].get(cat, {}).get("n", 0) for t in tags if "n_whales" in r[t])
        errs = [(r[t]["cats"][cat]["err"], r[t]["cats"][cat]["n_scored"]) for t in tags if "n_whales" in r[t] and r[t]["cats"].get(cat, {}).get("err") is not None]
        ne = sum(x[1] for x in errs)
        e = sum(x[0] * x[1] for x in errs) / ne if ne else None
        return (c / n) / real if real else None, e, ne
    r05 = [t for t in r if t == TRAIN_SET]
    r06 = [t for t in r if t.startswith("r06") and t not in CHECK_SETS]
    c5, e5, n5 = agg(r05)
    c6, e6, n6 = agg(r06)
    return c5, c6, e5, e6, n5, n6


def by_eye_coverage(res) -> dict:
    """coverage from AI by-eye labels of rendered whales (labels_ai_renders_sample.txt: random run05 / run06-test whales;
    labels_ai_<tag>.txt for a check set, every whale): share / real share, and the #S (spike) / #R (real-like) fin marks."""
    groups = defaultdict(list)
    p = C.OUT / "labels_ai_renders_sample.txt"
    if p.exists():
        for line in open(p):
            if line.strip() and not line.startswith("#"):
                tag, _, _, code = line.split(None, 3)
                groups["run05" if tag == "run05" else "run06 tests"].append(code)
    for f in sorted(C.OUT.glob("labels_ai_r06r_test_*.txt")):           # every whale of a check set, by eye
        name = f.stem.replace("labels_ai_", "")
        for line in open(f):
            if line.strip() and not line.startswith("#"):
                groups[name].append(line.split(None, 2)[2])
    out = {}
    for g, codes in groups.items():
        cats = defaultdict(int)
        fins = defaultdict(int)
        for c in codes:
            body, _, mark = c.partition("#")
            cat = body.split()[0]
            cats[cat] += 1
            if mark.strip():
                fins[mark.strip()] += 1
        n = len(codes)
        out[g] = {"n": n, "fin_marks": dict(fins),
                  "cats": {c: {"n": k, "share": k / n, "coverage": (k / n) / res["real_share_pooled"][c] if res["real_share_pooled"].get(c) else None}
                           for c, k in cats.items()}}
    return out


def gap_rows(res) -> list[dict]:
    """the ranked gap table (ranked by cost on the TRAINING-day sample; validation numbers are the reported ones)."""
    tr, va = res["by_category"]["train"], res["by_category"]["val"]
    order = sorted([c for c in tr if c != "X_none"], key=lambda c: -(tr[c]["cost"] or 0)) + (["X_none"] if "X_none" in tr else [])
    rows = []
    for i, c in enumerate(order):
        t, v = tr.get(c, {}), va.get(c, {})
        c5, c6, e5, e6, n5, n6 = render_cov(res, c)
        rows.append({"rank": i + 1, "category": c, "what": C.CATS[c][1], "three_way": C.THREE_OF[c],
                     "share_train": t.get("share"), "share_val": v.get("share"),
                     "err_train": t.get("err"), "k_n_train": [t.get("k", 0), t.get("n", 0)],
                     "err_val": v.get("err"), "err_val_ci": v.get("ci"), "k_n_val": [v.get("k", 0), v.get("n", 0)],
                     "cost_train": t.get("cost"), "cost_val": v.get("cost"),
                     "coverage_run05": c5, "coverage_run06_tests": c6, "render_err_run05": e5, "render_n_run05": n5,
                     "render_err_run06_tests": e6, "render_n_run06_tests": n6,
                     "main_confusion": res.get("main_confusion", {}).get(c),
                     "cause": cause(t.get("err"), e5, n5, c5, c),
                     "factory": ACTION.get(c, "")})
    return rows


def print_report(res, a):
    print(f"\n== part-model gap table: {res['model']} ({res['variant']} variant, rule {res['rule']}); AI by-eye labels, blind")
    for sp in ("train", "val"):
        o = res[f"overall_{sp}"]
        print(f"   overall three-way error {sp}: {fmt(o['err'], 3)} [{fmt(o['ci'][0], 3)}, {fmt(o['ci'][1], 3)}] n {o['n']}")
    hdr = f"{'rank':4s} {'category':8s} {'share tr/val':>12s} {'err train':>16s} {'err val':>20s} {'cost tr/val':>12s} {'cov r05/r06':>12s} {'err rend r05/r06':>18s}  main confusion (train+val) | cause | factory"
    print(hdr)
    tr, va = res["by_category"]["train"], res["by_category"]["val"]
    order = sorted([c for c in tr if c != "X_none"], key=lambda c: -(tr[c]["cost"] or 0)) + (["X_none"] if "X_none" in tr else [])
    labels = None
    for i, c in enumerate(order):
        t, v = tr.get(c, {}), va.get(c, {})
        c5, c6, e5, e6, n5, n6 = render_cov(res, c)
        cz = cause(t.get("err"), e5, n5, c5, c)          # the training-render set only (run06 tests are a new look for r05 models)
        conf = res.get("main_confusion", {}).get(c, "")
        print(f"{i + 1:<4d} {c:8s} {fmt(t.get('share'))}/{fmt(v.get('share'))}   "
              f"{fmt(t.get('err'))} ({t.get('k', 0)}/{t.get('n', 0)})".ljust(16) + "  "
              f"{fmt(v.get('err'))} [{fmt((v.get('ci') or [None])[0])},{fmt((v.get('ci') or [None, None])[1])}] ({v.get('k', 0)}/{v.get('n', 0)})".ljust(22) +
              f" {fmt(t.get('cost'), 3)}/{fmt(v.get('cost'), 3)}  {fmt(c5)}/{fmt(c6)}  {fmt(e5)}({n5})/{fmt(e6)}({n6})  | {conf} | {cz} | {ACTION.get(c, '')}")
    print("\n   without the instances still ambiguous after the second look (a / a2): error train / val")
    for c in order:
        t, v = res["by_category_unambiguous"]["train"].get(c, {}), res["by_category_unambiguous"]["val"].get(c, {})
        print(f"   {c:8s} {fmt(t.get('err'))} ({t.get('k', 0)}/{t.get('n', 0)}) / {fmt(v.get('err'))} ({v.get('k', 0)}/{v.get('n', 0)})")
    print("\n   within categories (unambiguous; train + val counted separately): error (k/n)")
    for g in sorted(set(res["within"]["train"]) | set(res["within"]["val"])):
        if g.split("|")[0] not in ("D_small", "B_back", "P_pec", "F_low", "D_tall", "D_tip", "H_head"):
            continue
        t, v = res["within"]["train"].get(g, {}), res["within"]["val"].get(g, {})
        print(f"   {g:18s} train {fmt(t.get('err'))} ({t.get('k', 0)}/{t.get('n', 0)})   val {fmt(v.get('err'))} ({v.get('k', 0)}/{v.get('n', 0)})")
    print("\n   attributes (train | val): share, error (k/n)")
    for g in sorted(set(res["by_attribute"]["train"]) | set(res["by_attribute"]["val"])):
        t, v = res["by_attribute"]["train"].get(g, {}), res["by_attribute"]["val"].get(g, {})
        print(f"   {g:5s} {C.ATTRS.get(g, g)[:44]:44s} train {fmt(t.get('share'))} {fmt(t.get('err'))} ({t.get('k', 0)}/{t.get('n', 0)})   val {fmt(v.get('share'))} {fmt(v.get('err'))} ({v.get('k', 0)}/{v.get('n', 0)})")
    print("\n   size (long side of the instance box, 2560 frame): error train / val, this variant | other variant")
    for g in ["<200", "200-400", "400-800", ">=800"]:
        t, v = res["by_size"]["train"].get(g, {}), res["by_size"]["val"].get(g, {})
        ot = res.get("by_size_other_variant", {}).get("train", {}).get(g, {})
        ov = res.get("by_size_other_variant", {}).get("val", {}).get(g, {})
        print(f"   {g:8s} {fmt(t.get('err'))} ({t.get('k', 0)}/{t.get('n', 0)}) / {fmt(v.get('err'))} ({v.get('k', 0)}/{v.get('n', 0)})   | {fmt(ot.get('err'))} / {fmt(ov.get('err'))}")
    print("\n   arms, error per category (val; train in brackets; w1's train numbers are not a test)")
    names = [n for n, x in res["arms"].items() if "by_category" in x]
    print("   " + f"{'category':8s} {'base':>12s} " + " ".join(f"{n.replace('parts_seg_r05_', ''):>12s}" for n in names))
    for c in order:
        row = f"   {c:8s} {fmt(res['by_category']['val'].get(c, {}).get('err'))} ({fmt(res['by_category']['train'].get(c, {}).get('err'))})"
        for n in names:
            bc = res["arms"][n]["by_category"]
            row += f" {fmt(bc['val'].get(c, {}).get('err'))} ({fmt(bc['train'].get(c, {}).get('err'))})"
        print(row)
    row = f"   {'overall':8s} {fmt(res['overall_val']['err'])} ({fmt(res['overall_train']['err'])})"
    for n in names:
        o = res["arms"][n]["overall"]
        row += f" {fmt(o['val']['err'])} ({fmt(o['train']['err'])})"
    print(row)
    print("\n   arms - base, accuracy gain on the same instances (val | train; weighted; 95 % bootstrap; fixed/broken counts)")
    for n in names:
        pv = res["arms"][n]["paired_vs_base"]
        for c in ["ALL"] + order:
            v_, t_ = pv.get(f"val|{c}"), pv.get(f"train|{c}")
            if not v_ and not t_:
                continue
            f = lambda x: "-" if not x else f"{x['gain']:+.2f} [{x['ci'][0]:+.2f},{x['ci'][1]:+.2f}] {x['fixed']}/{x['broken']}"  # noqa: E731
            print(f"   {n.replace('parts_seg_r05_', ''):6s} {c:8s} val {f(v_):32s} train {f(t_)}")
    miss = {n: x["missing"] for n, x in res["arms"].items() if "missing" in x}
    if miss:
        print("   arms without real-photo outputs:", ", ".join(miss))
    print("\n   coverage detail: real share (pooled) and median size (px, 2560 frame) | per render set: share, median size, model error on its renders (n)")
    tags = [t for t, x in res["renders"].items() if "n_whales" in x]
    print("   " + f"{'category':8s} {'real':>13s} " + " ".join(f"{t.replace('r06r_test_', 'r06 '):>24s}" for t in tags))
    for c in order:
        row = f"   {c:8s} {res['real_share_pooled'].get(c, 0):.2f} {res['real_median_L2560'].get(c, 0):6.0f}  "
        for t in tags:
            d = res["renders"][t]["cats"].get(c)
            row += (f" {d['share']:.2f} {d['median_L2560'] or 0:5.0f} {fmt(d.get('err'))}({d.get('n_scored', 0)})".rjust(25) if d else " " * 25)
        print(row)
    print("\n   coverage table (render share / real share): from labels (all whales) | by eye (AI, blind; run05 / run06 tests = 50 random")
    print("   whales each, check sets = every whale); model error on the check set's renders (n)")
    eye = res["coverage_by_eye"]
    checks = [t for t in CHECK_SETS if "n_whales" in res["renders"].get(t, {})]
    hdr = f"   {'category':8s} {'run05 lab/eye':>14s} {'r06 tests lab/eye':>18s}" + "".join(f" {t.replace('r06r_test_', '') + ' lab/eye err(n)':>24s}" for t in checks)
    print(hdr)
    for c in order:
        c5, c6, *_ = render_cov(res, c)
        e5 = eye.get("run05", {}).get("cats", {}).get(c, {}).get("coverage", 0.0 if "run05" in eye else None)
        e6 = eye.get("run06 tests", {}).get("cats", {}).get(c, {}).get("coverage", 0.0 if "run06 tests" in eye else None)
        row = f"   {c:8s} {fmt(c5):>6s}/{fmt(e5):<7s} {fmt(c6):>9s}/{fmt(e6):<8s}"
        for t in checks:
            d = res["renders"][t]["cats"].get(c, {})
            real = res["real_share_pooled"].get(c)
            lab = d.get("share", 0.0) / real if real else None
            ee = eye.get(t, {}).get("cats", {}).get(c, {}).get("coverage", 0.0 if t in eye else None)
            row += f" {fmt(lab):>8s}/{fmt(ee):<5s} {fmt(d.get('err'))}({d.get('n_scored', 0)})".ljust(25)
        print(row)
    for g, d in eye.items():
        print(f"   by-eye fin marks {g}: {d['fin_marks']} of {d['n']} whales (S = thin sharp spike on a mound, R = low broad rounded like real)")
    print("\n   renders (whales in the set; whales the model was scored on):")
    for t, x in res["renders"].items():
        print(f"   {t:16s} {x.get('n_whales', x.get('missing'))}  scored {x.get('n_scored', x.get('model_missing', '-'))}")


if __name__ == "__main__":
    raise SystemExit(main())
