"""Baseline against the real-water arms, from their real_metrics.json (evaluate_real.py) plus a paired bootstrap.

    PYTHONPATH=. .venv/bin/python experiments/realwater/compare.py --base parts_seg_r05_a \
        --arms parts_seg_r05_rw parts_seg_r05_sc parts_seg_r05_rwall --out experiments/realwater/compare.json

Like with like: for every model the fluke / fin / other decision rule is re-chosen HERE on the training days, in the
same family for all (evaluate_real's current two-part grid, and the older single-threshold grid as a check), then read
on the validation days: three-way accuracy against the by-eye labels and the view classifier, the classifier accuracy
by whale size, and for each arm the paired difference to the baseline on the same validation instances with a
bootstrap over PHOTOS (2000 resamples, 95 % interval; test-set sampling noise only). Run-to-run training noise: the
baseline recipe with seed 1 (parts_seg_r05_s1) is compared to the baseline the same way, if scored. Click numbers do
not depend on the rule and are copied from each model's real_metrics.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from model_fine_tuning.common import paths  # noqa: E402
from model_fine_tuning.parts_seg import evaluate_real as ER  # noqa: E402

SHADOW = Path("/var/tmp/sat3737-parts-cache/realwater/days_shadow")


def rows_of(name: str, days: list[str]) -> list[dict]:
    root = SHADOW if (SHADOW / days[0] / name / "instances.json").exists() else paths.EXPERIMENTS_DIR / "days_all"
    ER.DAYS = root
    rows = ER.load_all(name, days)
    for r in rows:
        r["t_eye"] = ER.eye_three(r["eye"])
        r["t_vc"] = ER.vc_three(r["vc"]) if not r["not_whale"] else None
    return rows


def correct(rows, variant, tf, td, truth, split="val", key=None) -> dict:
    """{instance key: (photo stem, 1 if the rule's three-way call equals the truth)}; key(r) -> group filter value."""
    return {r["key"]: (r["stem"], int(ER.rule_three(r["pred"][variant], tf, td) == r[truth]))
            for r in rows if (split == "all" or r["split"] == split) and r.get(truth) is not None and variant in r["pred"]}


GROUPS = {"size": lambda r: ER.size_bin(r), "truth_view": None, "whales_in_photo": lambda r: "1" if r["n_inst"] <= 1 else ">=2"}


def grouped(rows_b, rows_a, rb, ra, variant, truth, split) -> dict:
    """paired arm - base accuracy per group (size of the instance, its true view, one or several whales)."""
    byk = {r["key"]: r for r in rows_b}
    cb = correct(rows_b, variant, rb[0], rb[1], truth, split)
    ca = correct(rows_a, variant, ra[0], ra[1], truth, split)
    out = {}
    for g, fn in GROUPS.items():
        fn = fn or (lambda r, t=truth: r[t])
        vals = sorted({fn(byk[k]) for k in cb})
        out[g] = {}
        for v in vals:
            ks = [k for k in cb if fn(byk[k]) == v and k in ca]
            out[g][v] = paired({k: cb[k] for k in ks}, {k: ca[k] for k in ks})
    return out


def paired(base: dict, arm: dict, n_boot: int = 2000, seed: int = 0) -> dict:
    keys = sorted(set(base) & set(arm))
    if not keys:
        return {}
    stems = sorted({base[k][0] for k in keys})
    sidx = {s: i for i, s in enumerate(stems)}
    d = np.zeros(len(stems))
    n = np.zeros(len(stems))
    for k in keys:
        i = sidx[base[k][0]]
        d[i] += arm[k][1] - base[k][1]
        n[i] += 1
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        s = rng.integers(len(stems), size=len(stems))
        boots.append(d[s].sum() / n[s].sum())
    return {"n_instances": len(keys), "n_photos": len(stems), "diff": round(float(d.sum() / n.sum()), 4),
            "ci95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
            "base_acc": round(float(np.mean([base[k][1] for k in keys])), 4), "arm_acc": round(float(np.mean([arm[k][1] for k in keys])), 4)}


def summary(m: dict, variant: str) -> dict:
    v = m["variants"][variant]
    c = v.get("clicks", {})
    g = lambda x: (x or {}).get("within")  # noqa: E731
    return {"rule": [v["rule"]["fluke_share_min"], v["rule"].get("fluke_over_pectoral_min", 0.0), v["rule"]["fin_share_min"]],
            "val_eye": v["val"]["eye"]["ours"]["acc"], "val_eye_n": v["val"]["eye"]["ours"]["n"],
            "val_clf": v["val"]["classifier"]["ours"]["acc"], "val_clf_n": v["val"]["classifier"]["ours"]["n"],
            "train_eye": v["train"]["eye"]["ours"]["acc"], "train_clf": v["train"]["classifier"]["ours"]["acc"],
            "cutout_flags_val_eye": v["val"]["eye"]["baseline"]["acc"], "cutout_flags_val_clf": v["val"]["classifier"]["baseline"]["acc"],
            "clicks_fin_tip_in_largest_fin_region": g(c.get("fin_tip_in_largest_fin_region")),
            "clicks_fin_tip_in_any_fin_px": g(c.get("fin_tip_in_any_fin_pixel")),
            "clicks_fin_tip_within_quarter_base": c.get("fin_tip_within_quarter_base"),
            "clicks_fluke_notch_in": g(c.get("fluke_notch_in_fluke")), "clicks_fluke_all_three": c.get("fluke_all_three_points_in"),
            "by_size_val_clf": {k: x["ours_acc"] for k, x in v["breakdown_val_classifier"]["by_size"].items()},
            "by_truth_val_clf": {k: x["ours_acc"] for k, x in v["breakdown_val_classifier"]["by_truth"].items()},
            "by_instances_val_clf": {k: x["ours_acc"] for k, x in v["breakdown_val_classifier"]["by_instances_in_photo"].items()},
            "by_group_val_clf": {k: x["ours_acc"] for k, x in v["breakdown_val_classifier"]["by_group_type"].items()}}


GRID = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3]


def choose(rows, variant, family: str):
    """the decision rule, chosen on the TRAINING days exactly as evaluate_real.choose_rule does (mean of by-eye and
    classifier three-way accuracy), in one family for every model: "two_part" = evaluate_real's current grid
    (fluke share >= a and >= b x pectoral share), "single" = the same grid with b = 0 (the older one-threshold rule)."""
    if family == "two_part":
        tf, td, _ = ER.choose_rule(rows, variant)
        return tuple(tf) if isinstance(tf, (list, tuple)) else (tf, 0.0), td
    tr = [r for r in rows if r["split"] == "train" and variant in r["pred"]]
    eye = [r for r in tr if r["t_eye"]]
    vc = [r for r in tr if r["t_vc"]]
    best = None
    for tf in GRID:
        for td in GRID:
            a1 = np.mean([r["t_eye"] == ER.rule_three(r["pred"][variant], (tf, 0.0), td) for r in eye])
            a2 = np.mean([r["t_vc"] == ER.rule_three(r["pred"][variant], (tf, 0.0), td) for r in vc])
            if best is None or 0.5 * (a1 + a2) > best[0]:
                best = (0.5 * (a1 + a2), tf, td)
    return (best[1], 0.0), best[2]


def by_size(rows, variant, tf, td, truth="t_vc") -> dict:
    out = {}
    for r in rows:
        if r["split"] != "val" or r.get(truth) is None or variant not in r["pred"]:
            continue
        b = ER.size_bin(r)
        o = out.setdefault(b, [0, 0])
        o[0] += 1
        o[1] += int(ER.rule_three(r["pred"][variant], tf, td) == r[truth])
    return {b: {"n": n, "acc": round(c / n, 3)} for b, (n, c) in sorted(out.items())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="parts_seg_r05_a")
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--noise", default="parts_seg_r05_s1", help="the baseline recipe with another seed, if scored")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    names = [a.base] + [x for x in a.arms if (paths.EXPERIMENTS_DIR / x / "real_metrics.json").exists()]
    mets = {n: json.loads((paths.EXPERIMENTS_DIR / n / "real_metrics.json").read_text()) for n in names}
    days = mets[a.base]["days"]
    rows = {n: rows_of(n, days) for n in names}
    res = {"models": names, "rule_note": "rules re-chosen here on the training days for every model in the same family; "
           "the real_metrics.json numbers (evaluate_real's own choice) are under 'own_json'", "families": {}}
    for family in ("two_part", "single"):
        fam = {}
        for variant in ("full", "hr"):
            rules = {n: choose(rows[n], variant, family) for n in names}
            corr = {n: {t: correct(rows[n], variant, rules[n][0], rules[n][1], t) for t in ("t_eye", "t_vc")} for n in names}
            summ = {}
            for n in names:
                summ[n] = {"rule": [rules[n][0][0], rules[n][0][1], rules[n][1]],
                           "val_eye": round(float(np.mean([v[1] for v in corr[n]["t_eye"].values()])), 4), "val_eye_n": len(corr[n]["t_eye"]),
                           "val_clf": round(float(np.mean([v[1] for v in corr[n]["t_vc"].values()])), 4), "val_clf_n": len(corr[n]["t_vc"]),
                           "val_clf_by_size": by_size(rows[n], variant, rules[n][0], rules[n][1])}
            pairs = {n: {t: paired(corr[a.base][t], corr[n][t]) for t in ("t_eye", "t_vc")} for n in names[1:]}
            groups = {n: {f"{t}_{sp}": grouped(rows[a.base], rows[n], rules[a.base], rules[n], variant, t, sp)
                          for t in ("t_vc", "t_eye") for sp in ("val", "all")} for n in names[1:]}
            fam[variant] = {"summary": summ, "paired_val_vs_base": pairs, "groups_vs_base": groups}
        res["families"][family] = fam
    res["own_json"] = {v: {n: summary(mets[n], v) for n in names} for v in ("full", "hr")}
    a.out.write_text(json.dumps(res, indent=1))
    for family in ("two_part", "single"):
        for variant in ("full", "hr"):
            f = res["families"][family][variant]
            print(f"== {family} rule, {variant}")
            for n in names:
                s = f["summary"][n]
                print(f"{n:22s} rule {s['rule']}  val eye {s['val_eye']:.3f} (n {s['val_eye_n']})  clf {s['val_clf']:.3f} (n {s['val_clf_n']})  >=800px {s['val_clf_by_size'].get('>=800 px', {}).get('acc')}")
            for n, p in f["paired_val_vs_base"].items():
                print(f"   {n} - base: eye {p['t_eye'].get('diff')} {p['t_eye'].get('ci95')}  clf {p['t_vc'].get('diff')} {p['t_vc'].get('ci95')}")
    for variant in ("full", "hr"):
        g = res["families"]["two_part"][variant]["groups_vs_base"]
        for n in names[1:]:
            for tk in ("t_vc_val", "t_vc_all", "t_eye_all"):
                print(f"== groups two_part {variant} {n} - base, truth/split {tk}")
                for grp, d in g[n][tk].items():
                    print("   ", grp, "; ".join(f"{v}: n {p.get('n_instances')} base {p.get('base_acc')} arm {p.get('arm_acc')} d {p.get('diff')} {p.get('ci95')}" for v, p in d.items()))
    print("== clicks (rule-free), own json")
    for v in ("full", "hr"):
        for n in names:
            s = res["own_json"][v][n]
            print(f"{v:4s} {n:22s} fin tip in largest fin region {s['clicks_fin_tip_in_largest_fin_region']}  any fin px {s['clicks_fin_tip_in_any_fin_px']}  notch in fluke {s['clicks_fluke_notch_in']}  all 3 fluke points {s['clicks_fluke_all_three']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
