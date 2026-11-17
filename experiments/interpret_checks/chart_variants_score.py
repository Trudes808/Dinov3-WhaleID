"""Score the cut-out rules of chart_variants.py against the catalogue (read for scoring only).

    .venv/bin/python experiments/interpret_checks/chart_variants_score.py OUT_DIR [--parts fin,body,fluke] [--base cur]

Per part and rule, over the cut-outs the Labeling Assistant counts for that part ("used"), main instances with a
catalogue animal, pooled over the days in OUT_DIR:
  first    the closest look-alike in ANOTHER stack of the day is the same catalogue animal (queries whose animal has a
           cut-out in another stack: otherwise no answer can be right)
  top 5    mean share of the five closest in other stacks that are the same animal
  AUC      same-animal against different-animal pairs of cut-outs in different stacks
  vs base  queries that turned right / turned wrong against the base rule, two-sided sign test, and a 90 % interval of
           the change in "first" from a bootstrap over stacks (frames of a stack are near copies)
then the same "first" for the queries by what the base rule did to their cut-out (upright, tilted, on its side,
upside down, a quarter or more outside the photo).
"""
import argparse, glob, json
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("out", type=Path); ap.add_argument("--parts", default="fin,body,fluke"); ap.add_argument("--base", default="cur")
ap.add_argument("--only", default=""); ap.add_argument("--json", type=Path, default=None)
a = ap.parse_args()
rng = np.random.default_rng(0)


def auc(pos, neg):
    if not len(pos) or not len(neg):
        return float("nan")
    x = np.concatenate([pos, neg]); r = x.argsort().argsort().astype(float) + 1
    # ties are rare with float cosines; ranks as they come
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def sign_p(b, w):
    n = b + w
    if n == 0:
        return 1.0
    k = min(b, w); return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


days = {}
for f in sorted(glob.glob(str(a.out / "emb_*.npz"))):
    z = np.load(f); days[Path(f).stem[4:]] = (z["E"], json.loads(str(z["rows"])))
res = {}
for part in a.parts.split(","):
    variants = []
    for E, rows in days.values():
        for r in rows:
            if r["part"] == part and r["variant"] not in variants:
                variants.append(r["variant"])
    if a.only:
        variants = [v for v in variants if v in a.only.split(",") or v == a.base]
    per = {v: dict(hit=[], top5=[], pos=[], neg=[], q=[]) for v in variants}
    for day, (E, rows) in days.items():
        by = defaultdict(dict)
        for i, r in enumerate(rows):
            if r["part"] == part and r["used"] and r["cat"] and r["key"].endswith("~0"):
                by[r["variant"]][r["key"]] = i
        keys = sorted(set.intersection(*[set(by[v]) for v in variants])) if all(v in by for v in variants) else []
        if len(keys) < 3:
            continue
        base_rows = {k: rows[by[a.base][k]] for k in keys}
        stack = np.array([base_rows[k]["stack"] for k in keys]); cat = np.array([base_rows[k]["cat"] for k in keys])
        other = stack[:, None] != stack[None, :]; same = cat[:, None] == cat[None, :]
        can = (other & same).any(1); iu = np.triu_indices(len(keys), 1)
        for v in variants:
            X = E[[by[v][k] for k in keys]]; C = X @ X.T; Cm = np.where(other, C, -9.0)
            order = np.argsort(-Cm, 1); nn = order[:, 0]
            P = per[v]
            for i in np.nonzero(can)[0]:
                P["hit"].append(bool(same[i, nn[i]])); t5 = [j for j in order[i, :5] if other[i, j]]; P["top5"].append(float(np.mean(same[i, t5])))
                rv = rows[by[v][keys[i]]]
                P["q"].append(dict(day=day, key=keys[i], stack=f"{day}|{stack[i]}", cat=cat[i], base_angle=base_rows[keys[i]]["angle"], base_outside=base_rows[keys[i]]["outside"],
                                   changed=v != a.base and rv["same"] != a.base, angle=rv["angle"], outside=rv["outside"], zoom=rv["zoom"]))
            m = other[iu]; P["pos"].append(C[iu][m & same[iu]]); P["neg"].append(C[iu][m & ~same[iu]])
    if not per[a.base]["hit"]:
        continue
    base = np.array(per[a.base]["hit"]); q0 = per[a.base]["q"]; st = np.array([q["stack"] for q in q0]); ust = sorted(set(st)); idx_of = {s: np.nonzero(st == s)[0] for s in ust}
    ba = np.abs(np.array([q["base_angle"] for q in q0])); bo = np.array([q["base_outside"] for q in q0])
    if part == "fluke":
        ba = np.minimum(ba, 180 - ba)
    groups = {"upright (under 10 deg)": ba < 10, "tilted 10-30": (ba >= 10) & (ba < 30), "on its side 30-120": (ba >= 30) & (ba < 120), "upside down over 120": ba >= 120,
              "quarter or more outside": bo >= 0.25, "inside the photo": bo < 0.02}
    print(f"\n=== {part}: {len(base)} queries in {len(ust)} stacks on {len(days)} days ({', '.join(days)}); base rule '{a.base}'")
    print(f"    {'rule':11s} first  top5   AUC   | changed cut-outs | vs base: right / wrong, sign p | change in first, 90 % interval | outside>=1/4 | " + " | ".join(g.split(' (')[0] for g in groups))
    res[part] = {}
    for v in variants:
        P = per[v]; h = np.array(P["hit"]); b = int((h & ~base).sum()); w = int((~h & base).sum())
        boot = []
        for _ in range(2000):
            pick = np.concatenate([idx_of[s] for s in rng.choice(ust, len(ust))]); boot.append(h[pick].mean() - base[pick].mean())
        lo, hi = np.percentile(boot, [5, 95]); ch = np.mean([q["changed"] for q in P["q"]]) if v != a.base else 0.0
        o = np.mean([q["outside"] >= 0.25 for q in P["q"]])
        g = " | ".join(f"{h[m].mean():.3f} ({int(m.sum())})" if m.sum() else "  -  " for m in groups.values())
        A = auc(np.concatenate(P["pos"]), np.concatenate(P["neg"]))
        print(f"    {v:11s} {h.mean():.3f}  {np.mean(P['top5']):.3f}  {A:.3f} | {ch:6.1%}           | {b:3d} / {w:3d}  p {sign_p(b, w):.3f}          | {h.mean() - base.mean():+.3f}  [{lo:+.3f}, {hi:+.3f}]         | {o:5.1%}        | {g}")
        res[part][v] = dict(n=len(h), first=float(h.mean()), top5=float(np.mean(P["top5"])), auc=A, better=b, worse=w, p=sign_p(b, w), lo=float(lo), hi=float(hi), changed=float(ch), outside=float(o),
                            groups={k: [float(h[m].mean()) if m.sum() else None, int(m.sum())] for k, m in groups.items()})
if a.json:
    a.json.write_text(json.dumps(res, indent=1))
