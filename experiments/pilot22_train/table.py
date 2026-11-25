"""Pilot r06_pilot22 training test: the table and the verdict by the fixed rule of plan.md.

    PYTHONPATH=. .venv/bin/python experiments/pilot22_train/table.py  -> experiments/pilot22_train/results.md (+ results.json)

Inputs (all made by unchanged scripts, see run_p22.sh / run_p22_compare.sh): part-gap reports
(moonshot_cache/pilot22/d/part_gaps/report_<model>_hr.json), new_measures_p22.json (pred_fill.py score),
compare_p22.json (compare.py), finless_p22.json (finless.py), synth_val/<model>.json (w3e_scores.py synth),
pilot_val/<model>.json (score_pilot.py).
Rule (plan.md): an arm "helps" when both its seeds have a lower validation-day total part-gap cost than both W3e seeds
and no standing measure is worse for both seeds (both beyond the worse W3e seed on the bad side); "hurts" the mirror;
else "not separable at this size". "both" column: "better" / "worse" when both arm seeds lie beyond both W3e seeds,
else "-"; "sep" column: W3c's separability rule (gap larger than the larger seed distance) for reference.
"""
from __future__ import annotations

import json
from pathlib import Path

E = Path(__file__).resolve().parent
C = Path("/home/sat3737/NOAA_whales/moonshot_cache/pilot22/d")
SV = Path("/home/sat3737/NOAA_whales/moonshot_cache/parts_tri_expand/synth_val")
ARMS = {"W3e": ("parts_seg_r05_w3e", "parts_seg_r05_w3es1"), "P_add": ("parts_seg_p22add", "parts_seg_p22adds1"),
        "P_only": ("parts_seg_p22only", "parts_seg_p22onlys1")}
CATS = ["F_low", "F_edge", "F_up", "D_small", "D_tip", "D_tall", "B_back", "P_pec", "H_head"]


def jl(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def get(d, *ks):
    try:
        for k in ks:
            d = d[k]
        return d
    except (KeyError, TypeError, IndexError):
        return None


def both(ref, arm, higher_better):
    if None in ref or None in arm:
        return "-"
    if min(arm) > max(ref):
        return "better" if higher_better else "worse"
    if max(arm) < min(ref):
        return "worse" if higher_better else "better"
    return "-"


def sep(ref, arm):
    if None in ref or None in arm:
        return "-"
    dist = max(abs(ref[0] - ref[1]), abs(arm[0] - arm[1]))
    if min(arm) > max(ref) and min(arm) - max(ref) > dist:
        return "higher, sep"
    if max(arm) < min(ref) and min(ref) - max(arm) > dist:
        return "lower, sep"
    return "n.s."


def main() -> int:
    pg = {m: jl(C / "part_gaps" / f"report_{m}_hr.json") for pair in ARMS.values() for m in pair}
    nm, cp, fl = jl(E / "new_measures_p22.json"), jl(E / "compare_p22.json"), jl(E / "finless_p22.json")
    pv = {m: jl(C / "pilot_val" / f"{m}.json") for pair in ARMS.values() for m in pair}
    sv = {m: jl(SV / f"{m}.json") for pair in ARMS.values() for m in pair}
    arms = {a: p for a, p in ARMS.items() if all(pg[m] is not None for m in p)}
    rows = []   # (group, label, fn(model) -> value, n, higher_better, in_rule)

    def pgv(m, split, cat=None, field="err"):
        r = pg[m]
        if r is None:
            return None
        if cat is None:
            return get(r, f"overall_{split}", field)
        return get(r, "by_category", split, cat, field)

    def pgn(split, cat=None):
        r = pg["parts_seg_r05_w3e"]
        o = get(r, f"overall_{split}") if cat is None else get(r, "by_category", split, cat)
        return f"{o['n']}" if o else "-"

    rows.append(("1 part gaps", "VAL total cost (weighted error)", lambda m: pgv(m, "val"), pgn("val"), False, "total"))
    rows.append(("1 part gaps", "train-day total cost (ranking set, reference)", lambda m: pgv(m, "train"), pgn("train"), False, ""))
    for c in CATS:
        rows.append(("1 part gaps", f"VAL {c} error", (lambda c: lambda m: pgv(m, "val", c))(c), pgn("val", c), False, "cat"))
    for c in CATS:
        rows.append(("1 part gaps", f"VAL {c} cost", (lambda c: lambda m: pgv(m, "val", c, "cost"))(c), pgn("val", c), False, ""))
    rows.append(("2 fin shape", "IoU with the expanded-triangle truth (clicks 03-18)", lambda m: get(nm, "clicks", "2025-03-18", m, "iou_expanded", "mean"),
                 str(get(nm, "clicks", "2025-03-18", "n_fins") or "-"), True, ""))
    rows.append(("3 standing", "fin tip inside the largest fin region (hr, clicks 03-18)",
                 lambda m: get(cp, "clicks_2025_03_18", m, "hr", "fin_tip_in_largest_fin_region", "within"),
                 str(get(cp, "clicks_2025_03_18", "parts_seg_r05_w3e", "hr", "fin_tip_in_largest_fin_region", "n") or "-"), True, "guard"))
    rows.append(("3 standing", "finless backs called dorsal, by eye", lambda m: get(fl, "hr", m, "t_eye", "called_dorsal"),
                 str(get(fl, "hr", "parts_seg_r05_w3e", "t_eye", "n") or "33"), False, "guard"))
    rows.append(("3 standing", "finless backs called dorsal, classifier", lambda m: get(fl, "hr", m, "t_vc", "called_dorsal"),
                 str(get(fl, "hr", "parts_seg_r05_w3e", "t_vc", "n") or "151"), False, "guard"))
    rows.append(("3 standing", "three-way call by eye, crops", lambda m: get(cp, "three_way", "two_part", "hr", "val_eye", m, "acc"),
                 str(get(cp, "three_way", "two_part", "hr", "val_eye", "parts_seg_r05_w3e", "n") or "79"), True, "guard"))
    rows.append(("3 standing", "three-way call by eye, whole photo", lambda m: get(cp, "three_way", "two_part", "full", "val_eye", m, "acc"),
                 str(get(cp, "three_way", "two_part", "full", "val_eye", "parts_seg_r05_w3e", "n") or "79"), True, "guard"))
    rows.append(("3 standing", "three-way call by classifier, whole photo", lambda m: get(cp, "three_way", "two_part", "full", "val_clf", m, "acc"),
                 str(get(cp, "three_way", "two_part", "full", "val_clf", "parts_seg_r05_w3e", "n") or "-"), True, "guard"))
    rows.append(("3 standing", "fluke notch inside the fluke (hr, clicks 03-18)", lambda m: get(cp, "clicks_2025_03_18", m, "hr", "fluke_notch_in_fluke", "within"),
                 str(get(cp, "clicks_2025_03_18", "parts_seg_r05_w3e", "hr", "fluke_notch_in_fluke", "n") or "-"), True, "guard"))
    nval = str(get(pv.get("parts_seg_r05_w3e"), "n_val") or "-")
    rows.append(("4 renders", "pilot val dorsal IoU (rule A, 1280)", lambda m: get(pv[m], "iou", "dorsal_fin"), nval, True, ""))
    rows.append(("4 renders", "pilot val fluke IoU (1280)", lambda m: get(pv[m], "iou", "fluke"), nval, True, ""))
    nr = str(get(sv.get("parts_seg_r05_w3e"), "n_val") or "-")
    rows.append(("4 renders", "run05 val dorsal IoU (rule-A label)", lambda m: get(sv[m], "iou", "triA", "dorsal_fin"), nr, True, ""))
    rows.append(("4 renders", "run05 val fluke IoU", lambda m: get(sv[m], "iou", "old", "fluke"), nr, True, ""))

    names = [m for p in arms.values() for m in p]
    others = [a for a in arms if a != "W3e"]
    head = "| measure | n | " + " | ".join(f"{a} s0 | {a} s1" for a in arms) + "".join(f" | {a}: both / sep" for a in others) + " |"
    lines = [head, "|" + "---|" * (2 + 2 * len(arms) + len(others))]
    out = {"arms": arms, "rows": [], "verdict": {}}
    for g, label, fn, n, hb, role in rows:
        v = {m: fn(m) for m in names}
        cells = [f"{v[m]:.3f}" if isinstance(v[m], (int, float)) else "-" for m in names]
        rec = {"group": g, "measure": label, "n": n, "values": v, "higher_better": hb, "role": role}
        extra = ""
        for a in others:
            ref, arm = tuple(v[m] for m in arms["W3e"]), tuple(v[m] for m in arms[a])
            b, s = both(ref, arm, hb), sep(ref, arm)
            rec[a] = {"both": b, "sep": s}
            extra += f" | {b} / {s}"
        out["rows"].append(rec)
        lines.append(f"| {label} | {n} | " + " | ".join(cells) + extra + " |")
    for a in others:
        tot = next(r for r in out["rows"] if r["role"] == "total")[a]["both"]
        guards = [r for r in out["rows"] if r["role"] == "guard"]
        g_worse = [r["measure"] for r in guards if r[a]["both"] == "worse"]
        g_better = [r["measure"] for r in guards if r[a]["both"] == "better"]
        if tot == "better" and not g_worse:
            verdict = "HELPS"
        elif tot == "worse" and not g_better:
            verdict = "HURTS"
        else:
            verdict = "not separable at this size"
        moved = {r["measure"]: r[a]["both"] for r in out["rows"] if r["role"] == "cat" and r[a]["both"] != "-"}
        out["verdict"][a] = {"verdict": verdict, "total": tot, "guards_worse_both": g_worse, "guards_better_both": g_better,
                             "categories_moved_both_seeds": moved}
        lines += ["", f"**{a}: {verdict}** (val total: {tot}; standing measures worse for both seeds: {g_worse or 'none'}; "
                      f"better for both: {g_better or 'none'}; categories moved for both seeds: {moved or 'none'})"]
    lines += ["", __doc__.strip().split("\n\n")[-1].replace("\n", " ")]
    (E / "results.md").write_text("\n".join(lines) + "\n")
    (E / "results.json").write_text(json.dumps(out, indent=1, default=str))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
