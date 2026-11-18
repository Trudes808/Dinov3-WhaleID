"""Baseline parts_seg_r05_a against the weak-label runs, like with like.

    PYTHONPATH=. .venv/bin/python experiments/parts_weak/compare.py --arms parts_seg_r05_w1 [parts_seg_r05_w1_noneed] \
        --out experiments/parts_weak/compare.json

Per model and variant (full = whole photo at 1280, hr = per-instance window):
  three-way fluke / fin / other: the decision rule re-chosen here on the TRAINING days for every model in the same
    family (evaluate_real.choose_rule's two-part grid; and the one-threshold grid as a check), read on the VALIDATION days
    against the by-eye labels (primary) and the view classifier (partly circular for the weak runs: they were trained
    on the classifier's answers of the training days); paired difference to the baseline on the same instances with a
    bootstrap over photos (2000 resamples, 95 %).
  validation days: dorsal-fin views with no fin pixel, fluke views with no fluke pixel (view = by-eye D / F, else the
    classifier's confident answer), and what the cut-outs need (cutouts_v3/make.py): dorsal views (classifier
    P(dorsal) >= 0.5 and a parts_up fin cut-out) that get a fin region (hr), fluke views (P(fluke) >= 0.5) whose fluke
    class covers >= 25 % of the instance (hr, pixels near the instance / instance area). Same on the training days for
    reference (the baseline's 88.9 % / 81 % were measured there).
  clicks of 2025-03-18 only (never trained on): evaluate_real.click_metrics with its click days set to that day.
  synthetic validation IoU (run05 val renders) from each run's val_metrics.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from model_fine_tuning.common import geometry, paths  # noqa: E402
from model_fine_tuning.parts_seg import evaluate_real as ER  # noqa: E402

SHADOW = Path("/var/tmp/sat3737-parts-cache/parts_weak/days_shadow")
RUNS = Path("/var/tmp/sat3737-parts-cache/parts_weak/runs")
DAYS_ALL = paths.EXPERIMENTS_DIR / "days_all"
GRID = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3]


def root_of(name: str, day: str) -> Path:
    return SHADOW if (SHADOW / day / name / "instances.json").exists() else DAYS_ALL


def rows_of(name: str, days: list[str]) -> list[dict]:
    ER.DAYS = root_of(name, days[0])
    rows = ER.load_all(name, days)
    for r in rows:
        r["t_eye"] = ER.eye_three(r["eye"])
        r["t_vc"] = ER.vc_three(r["vc"]) if not r["not_whale"] else None
    ER.DAYS = DAYS_ALL
    return rows


def choose(rows, variant, family):
    if family == "two_part":
        tf, td, _ = ER.choose_rule(rows, variant)
        return tuple(tf), td
    tr = [r for r in rows if r["split"] == "train" and variant in r["pred"]]
    eye = [r for r in tr if r["t_eye"]]; vc = [r for r in tr if r["t_vc"]]
    best = None
    for tf in GRID:
        for td in GRID:
            a1 = np.mean([r["t_eye"] == ER.rule_three(r["pred"][variant], (tf, 0.0), td) for r in eye])
            a2 = np.mean([r["t_vc"] == ER.rule_three(r["pred"][variant], (tf, 0.0), td) for r in vc])
            if best is None or 0.5 * (a1 + a2) > best[0]:
                best = (0.5 * (a1 + a2), tf, td)
    return (best[1], 0.0), best[2]


def correct(rows, variant, tf, td, truth) -> dict:
    return {r["key"]: (r["stem"], int(ER.rule_three(r["pred"][variant], tf, td) == r[truth]))
            for r in rows if r["split"] == "val" and r.get(truth) is not None and variant in r["pred"]}


def paired(base: dict, arm: dict, n_boot: int = 2000, seed: int = 0) -> dict:
    keys = sorted(set(base) & set(arm))
    if not keys:
        return {}
    stems = sorted({base[k][0] for k in keys}); sidx = {s: i for i, s in enumerate(stems)}
    d = np.zeros(len(stems)); n = np.zeros(len(stems))
    for k in keys:
        i = sidx[base[k][0]]; d[i] += arm[k][1] - base[k][1]; n[i] += 1
    rng = np.random.default_rng(seed)
    boots = [d[s].sum() / n[s].sum() for s in (rng.integers(len(stems), size=len(stems)) for _ in range(n_boot))]
    return {"n": len(keys), "photos": len(stems), "base_acc": round(float(np.mean([base[k][1] for k in keys])), 4),
            "arm_acc": round(float(np.mean([arm[k][1] for k in keys])), 4), "diff": round(float(d.sum() / n.sum()), 4),
            "ci95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]}


def view_of(r) -> str | None:
    if r["t_eye"]:
        return r["t_eye"]
    return r["t_vc"]


def coverage(rows, variant, split) -> dict:
    out = {}
    sel = [r for r in rows if r["split"] == split and variant in r["pred"] and not r["not_whale"]]
    dv = [r for r in sel if view_of(r) == "dorsal"]
    fv = [r for r in sel if view_of(r) == "fluke"]
    out["dorsal_views"] = len(dv)
    out["dorsal_views_no_fin_px"] = sum(r["pred"][variant]["pixels_near"]["dorsal_fin"] == 0 for r in dv)
    out["fluke_views"] = len(fv)
    out["fluke_views_no_fluke_px"] = sum(r["pred"][variant]["pixels_near"]["fluke"] == 0 for r in fv)
    return out


def cutout_needs(rows, split, parts_up: dict) -> dict:
    sel = [r for r in rows if r["split"] == split and "hr" in r["pred"] and r["vc"]]
    fin_v = [r for r in sel if r["vc"][1] >= 0.5 and "fin" in parts_up.get(r["key"], [])]
    fl_v = [r for r in sel if r["vc"][0] >= 0.5]
    got_fin = sum("fin" in r["pred"]["hr"] for r in fin_v)
    cov = sum(r["pred"]["hr"]["pixels_near"]["fluke"] >= 0.25 * max(r["pred"]["hr"]["area"], 1) for r in fl_v)
    return {"dorsal_views": len(fin_v), "with_fin_region": got_fin, "share_fin_region": round(got_fin / max(len(fin_v), 1), 4),
            "fluke_views": len(fl_v), "fluke_covers_quarter": cov, "share_fluke_quarter": round(cov / max(len(fl_v), 1), 4)}


def clicks_test(name: str, days: list[str], table: dict) -> dict:
    ER.DAYS = root_of(name, days[0])
    ER.CLICK_DAYS = ("2025-03-18",)
    out = {}
    for v in ("full", "hr"):
        s, recs = ER.click_metrics(name, v, table)
        out[v] = s
    ER.DAYS = DAYS_ALL
    return out


def val_iou(name: str) -> dict:
    for p in (paths.EXPERIMENTS_DIR / name / "val_metrics.json", RUNS / name / "val_metrics.json"):
        if p.exists():
            m = json.loads(p.read_text())
            return {"epoch": m["epoch"], "fluke": round(m["iou"]["fluke"], 4), "dorsal_fin": round(m["iou"]["dorsal_fin"], 4),
                    "pectoral_fin": round(m["iou"]["pectoral_fin"], 4), "head": round(m["iou"]["head"], 4), "body": round(m["iou"]["body"], 4),
                    "score": round(m["score"], 4)}
    return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="parts_seg_r05_a")
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    split = json.loads((paths.SPLITS_DIR / "days.json").read_text())
    days = split["train"] + split["val"]
    assert not set(days) & set(split["eval"])
    names = [a.base] + [n for n in a.arms if (SHADOW / days[0] / n / "instances.json").exists()]
    rows = {n: rows_of(n, days) for n in names}
    t = geometry.load_folded_table(paths.FOLDED_TABLE)
    table = {"x_min_m": float(t["x_min_m"]), "length_m": float(t["length_m"])}
    parts_up = {}
    for day in days:
        for l in open(DAYS_ALL / day / "parts_up" / "index.jsonl"):
            r = json.loads(l)
            for i in r.get("instances", []):
                parts_up[i["key"]] = i.get("parts", [])
    res = {"models": names, "note": "rule chosen on the training days per model; numbers on the validation days unless marked; "
           "classifier agreement is partly circular for the weak runs (trained on the classifier's training-day answers)", "three_way": {}}
    for family in ("two_part", "single"):
        res["three_way"][family] = {}
        for variant in ("full", "hr"):
            rules = {n: choose(rows[n], variant, family) for n in names}
            corr = {n: {tt: correct(rows[n], variant, *rules[n], tt) for tt in ("t_eye", "t_vc")} for n in names}
            res["three_way"][family][variant] = {
                "rule": {n: [rules[n][0][0], rules[n][0][1], rules[n][1]] for n in names},
                "val_eye": {n: {"acc": round(float(np.mean([v[1] for v in corr[n]["t_eye"].values()])), 4), "n": len(corr[n]["t_eye"])} for n in names},
                "val_clf": {n: {"acc": round(float(np.mean([v[1] for v in corr[n]["t_vc"].values()])), 4), "n": len(corr[n]["t_vc"])} for n in names},
                "val_eye_confusion_rows_true_fluke_dorsal_other": {n: ER.confusion([(r["t_eye"], ER.rule_three(r["pred"][variant], *rules[n]))
                                                                                  for r in rows[n] if r["split"] == "val" and r["t_eye"] and variant in r["pred"]])["matrix_rows_true_fluke_dorsal_other"] for n in names},
                "paired_vs_base": {n: {tt: paired(corr[a.base][tt], corr[n][tt]) for tt in ("t_eye", "t_vc")} for n in names[1:]}}
    res["coverage"] = {sp: {v: {n: coverage(rows[n], v, sp) for n in names} for v in ("full", "hr")} for sp in ("val", "train")}
    res["cutout_needs"] = {sp: {n: cutout_needs(rows[n], sp, parts_up) for n in names} for sp in ("val", "train")}
    res["clicks_2025_03_18"] = {n: clicks_test(n, days, table) for n in names}
    res["synthetic_val_iou"] = {n: val_iou(n) for n in names}
    a.out.write_text(json.dumps(res, indent=1))
    # ---- print
    for family in ("two_part", "single"):
        for variant in ("full", "hr"):
            f = res["three_way"][family][variant]
            print(f"== three-way, {family} rule, {variant}")
            for n in names:
                print(f"  {n:26s} rule {f['rule'][n]}  val eye {f['val_eye'][n]['acc']:.3f} (n {f['val_eye'][n]['n']})  clf {f['val_clf'][n]['acc']:.3f} (n {f['val_clf'][n]['n']})")
            for n, p in f["paired_vs_base"].items():
                print(f"     {n} - base: eye {p['t_eye'].get('diff')} {p['t_eye'].get('ci95')}  clf {p['t_vc'].get('diff')} {p['t_vc'].get('ci95')}")
    for sp in ("val", "train"):
        for v in ("full", "hr"):
            print(f"== coverage {sp} {v}: " + "  ".join(f"{n}: {res['coverage'][sp][v][n]}" for n in names))
        print(f"== cut-out needs {sp}: " + "  ".join(f"{n}: {res['cutout_needs'][sp][n]}" for n in names))
    for n in names:
        for v in ("full", "hr"):
            c = res["clicks_2025_03_18"][n][v]
            print(f"== clicks 03-18 {n} {v}: tip in fin region {c['fin_tip_in_largest_fin_region']}  centroid off/base {c.get('fin_centroid_off_over_base')}  "
                  f"notch {c['fluke_notch_in_fluke']}  tipL {c['fluke_tip_left_in_fluke']}  tipR {c['fluke_tip_right_in_fluke']}  all3 {c['fluke_all_three_points_in']}")
    print("== synthetic val IoU", json.dumps(res["synthetic_val_iou"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
