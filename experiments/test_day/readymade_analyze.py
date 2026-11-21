"""Round 2: score the ready-made segmentations (readymade.py output) against the by-eye truth with the rules fixed in
round2_rules.md, section 2 (class map, X = 0.5 % and X = 0, land subject >= 25 %, whale pixels = live AND S6h, veto).

    .venv/bin/python experiments/test_day/readymade_analyze.py --root ROOT --rm ROOT/round2/readymade --out ROOT/round2

-> OUT/readymade_photos.tsv (per photo: group shares per model, whale-pixel groups, veto numbers),
   OUT/readymade_measures.json (every number of the round-2 tables, sections 2a-2e). Nothing is fitted or tuned.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

GROUPS = ["person", "boat", "vessel", "land", "water", "sky", "other"]
MAP = {
    "coco": {"person": ["person"], "boat": ["fence-merged", "railroad", "platform", "floor-wood", "floor-other-merged", "stairs", "bridge"],
             "vessel": ["boat"], "land": ["mountain-merged", "rock-merged", "sand", "dirt-merged", "gravel", "tree-merged", "grass-merged", "building-other-merged",
                                          "house", "roof", "road", "pavement-merged", "playingfield", "snow"],
             "water": ["sea", "water-other", "river"], "sky": ["sky-other-merged"]},
    "ade": {"person": ["person"], "boat": ["railing", "fence", "bannister", "floor", "pole", "step", "stairs", "stairway", "column", "base"],
            "vessel": ["boat", "ship"], "land": ["mountain", "hill", "earth", "land", "rock", "sand", "tree", "palm", "plant", "grass", "field", "building", "house",
                                                 "skyscraper", "tower", "road", "path", "dirt track"],
            "water": ["sea", "water", "river", "lake", "swimming pool", "waterfall", "fountain"], "sky": ["sky"]},
}
HUB = {"coco": "facebook/mask2former-swin-large-coco-panoptic", "ade": "facebook/mask2former-swin-large-ade-semantic"}
VETO = ("person", "boat", "vessel")
NOWHALE = ["marker_rail", "bow", "crew_nowhale", "marker_land", "water"]


def class_names(model: str) -> list[str]:
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(HUB[model]).id2label
    return [c[i].strip() for i in range(len(c))]


def lut_for(model: str):
    names = class_names(model); g = np.full(256, GROUPS.index("other"), np.uint8)
    for grp, lst in MAP[model].items():
        for n in lst:
            assert n in names, (model, n)
            g[names.index(n)] = GROUPS.index(grp)
    return names, g


def prf(pred: list[bool], truth: list[bool]) -> dict:
    tp = sum(p and t for p, t in zip(pred, truth)); fp = sum(p and not t for p, t in zip(pred, truth)); fn = sum(t and not p for p, t in zip(pred, truth))
    return {"truth_n": tp + fn, "said_n": tp + fp, "recall": round(tp / max(tp + fn, 1), 3), "precision": round(tp / (tp + fp), 3) if tp + fp else None}


def auc(score: list[float], truth: list[bool]) -> float | None:
    s = np.array(score); t = np.array(truth, bool)
    if t.all() or not t.any():
        return None
    from scipy.stats import rankdata
    r = rankdata(s); n1 = t.sum(); n0 = (~t).sum()
    return round(float((r[t].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)), 3)


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True); p.add_argument("--rm", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(); R = a.root; D = R / "days" / R.name
    T = {r["seq"]: r for r in csv.DictReader(open(R / "truth.tsv"), delimiter="\t")}
    stems = {Path(json.loads(l)["image"]).stem.split("_")[-2]: Path(json.loads(l)["image"]).stem for l in open(D / "images" / "index.jsonl")}
    names, luts = {}, {}
    for m in ("coco", "ade"):
        names[m], luts[m] = lut_for(m)
    persons = json.loads((a.rm / "cocoinst" / "persons.json").read_text())
    rows = []; whale_cls = {m: Counter() for m in names}; whale_grp = {m: Counter() for m in list(names) + ["union"]}
    veto = defaultdict(lambda: defaultdict(lambda: np.zeros(6, np.int64)))   # [model][class] -> live px, live removed, false px, false removed, true px, true removed
    s6veto = defaultdict(lambda: defaultdict(lambda: np.zeros(2, np.int64)))
    for seq in sorted(stems):
        t = T[seq]; stem = stems[seq]
        lab = {m: np.array(Image.open(a.rm / m / f"{seq}.png")) for m in names}
        H, W = lab["coco"].shape
        live = cv2.resize(np.array(Image.open(D / "canon" / f"{stem}_mask.png")), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        s6 = np.array(Image.open(R / "cand_s6" / f"{stem}_s6.png")) > 0
        pin = np.array(Image.open(a.rm / "cocoinst" / f"{seq}.png")) > 0
        grp = {m: luts[m][lab[m]] for m in names}
        r = {"seq": seq, "truth_class": t["class"], "has_whale": t["has_whale"]}
        for m in names:
            sh = np.bincount(grp[m].ravel(), minlength=len(GROUPS)) / grp[m].size
            for k, g in enumerate(GROUPS):
                r[f"{m}_{g}"] = round(float(sh[k]), 5)
        r["inst_person_n05"] = sum(x["score"] >= 0.5 for x in persons.get(seq, [])); r["inst_person_max"] = max([x["score"] for x in persons.get(seq, [])] or [0.0])
        r["inst_person_px"] = round(float(pin.mean()), 5)
        vmask = {m: np.isin(grp[m], [GROUPS.index(v) for v in VETO]) for m in names}; vmask["union"] = vmask["coco"] | vmask["ade"]; vmask["inst_person"] = pin
        true = live & s6; false = live & ~s6
        r["live_px"] = int(live.sum()); r["s6_px"] = int(s6.sum()); r["true_px"] = int(true.sum())
        whale_photo = t["class"] in ("whale", "whale_crew")
        for m, v in vmask.items():
            x = veto[m][t["class"]]
            x += [live.sum(), (live & v).sum(), false.sum(), (false & v).sum(), true.sum(), (true & v).sum()]
            s6veto[m][t["class"]] += [s6.sum(), (s6 & v).sum()]
            r[f"veto_{m}_live_removed"] = round(float((live & v).sum() / max(live.sum(), 1)), 4)
            r[f"veto_{m}_true_removed"] = round(float((true & v).sum() / max(true.sum(), 1)), 4) if true.any() else ""
        if whale_photo and true.any():
            for m in names:
                c = np.bincount(lab[m][true], minlength=len(names[m])); whale_cls[m].update({names[m][k]: int(n) for k, n in enumerate(c) if n})
                gc = np.bincount(grp[m][true], minlength=len(GROUPS)); whale_grp[m].update({GROUPS[k]: int(n) for k, n in enumerate(gc) if n})
                r[f"whale_{m}_top_group"] = GROUPS[int(np.argmax(gc))]; r[f"whale_{m}_top_share"] = round(float(gc.max() / gc.sum()), 3)
                for g in ("person", "boat", "vessel", "land", "water", "other"):
                    r[f"whale_{m}_{g}"] = round(float(gc[GROUPS.index(g)] / gc.sum()), 3)
                cc = Counter({names[m][k]: int(n) for k, n in enumerate(c) if n}); top = cc.most_common(1)[0]
                r[f"whale_{m}_top_class"] = top[0]; r[f"whale_{m}_rock"] = round(cc.get("rock" if m == "ade" else "rock-merged", 0) / gc.sum(), 3)
        rows.append(r)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in rows[0], list(rows[0]).index(k) if k in rows[0] else 0, k))
    with open(a.out / "readymade_photos.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, delimiter="\t", restval=""); w.writeheader(); w.writerows(rows)

    out = {}
    tr = {"person": lambda t: t["person"] == "1", "rail_or_boat_parts": lambda t: t["rail"] == "1" or t["deck"] == "1", "rail_only": lambda t: t["rail"] == "1",
          "other_vessel": lambda t: t["vessel"] == "1", "land_subject": lambda t: t["land_subj"] == "1", "land_background": lambda t: t["land_bg"] == "1"}
    d = {}
    for X in (0.005, 0.0):
        for m in names:
            g = lambda r, k: r[f"{m}_{k}"]
            pred = {"person": lambda r: g(r, "person") > X if X == 0 else g(r, "person") >= X,
                    "rail_or_boat_parts": lambda r: (g(r, "boat") + g(r, "vessel")) > X if X == 0 else (g(r, "boat") + g(r, "vessel")) >= X,
                    "other_vessel": lambda r: g(r, "vessel") > X if X == 0 else g(r, "vessel") >= X,
                    "land_subject": lambda r: g(r, "land") >= 0.25,
                    "land_background": lambda r: (g(r, "land") > 0 if X == 0 else g(r, "land") >= X) and g(r, "land") < 0.25}
            pred["rail_only"] = pred["rail_or_boat_parts"]
            for thing, f in tr.items():
                truth = [f(T[r["seq"]]) for r in rows]; pr = [pred[thing](r) for r in rows]
                e = prf(pr, truth); e["missed"] = [r["seq"] for r, p_, t_ in zip(rows, pr, truth) if t_ and not p_][:60]; e["false"] = [r["seq"] for r, p_, t_ in zip(rows, pr, truth) if p_ and not t_][:60]
                d[f"{thing}|{m}|X={X}"] = e
    truth = [tr["person"](T[r["seq"]]) for r in rows]; pr = [r["inst_person_n05"] > 0 for r in rows]
    e = prf(pr, truth); e["missed"] = [r["seq"] for r, p_, t_ in zip(rows, pr, truth) if t_ and not p_]; e["false"] = [r["seq"] for r, p_, t_ in zip(rows, pr, truth) if p_ and not t_]
    d["person|cocoinst>=0.5"] = e
    out["2a_photo_level"] = d
    out["2a_auc_descriptive"] = {f"{thing}|{m}": auc([r[f"{m}_{grp}"] if grp != "boat+vessel" else r[f"{m}_boat"] + r[f"{m}_vessel"] for r in rows], [tr[thing](T[r["seq"]]) for r in rows])
                                 for m in names for thing, grp in (("person", "person"), ("rail_or_boat_parts", "boat+vessel"), ("other_vessel", "vessel"), ("land_subject", "land"), ("land_background", "land"))}
    out["2a_auc_descriptive"]["person|cocoinst_max_score"] = auc([r["inst_person_max"] for r in rows], [tr["person"](T[r["seq"]]) for r in rows])
    # by truth class: mean shares
    bc = {}
    for c in sorted({r["truth_class"] for r in rows}):
        sel = [r for r in rows if r["truth_class"] == c]
        bc[c] = {"n": len(sel), **{f"{m}_{g}": round(float(np.mean([r[f"{m}_{g}"] for r in sel])), 4) for m in names for g in GROUPS}}
    out["2b_mean_share_by_class"] = bc
    # what they call a whale
    W = {}
    for m in names:
        tot = sum(whale_cls[m].values())
        W[m] = {"whale_px": tot, "top_classes": [(k, round(v / tot, 4)) for k, v in whale_cls[m].most_common(12)],
                "groups": {g: round(whale_grp[m].get(g, 0) / tot, 4) for g in GROUPS}}
        wr = [r for r in rows if f"whale_{m}_top_group" in r]
        W[m]["photos"] = len(wr); W[m]["photo_top_group"] = dict(Counter(r[f"whale_{m}_top_group"] for r in wr))
        W[m]["photo_top_class"] = dict(Counter(r[f"whale_{m}_top_class"] for r in wr).most_common(10))
        for g in ("person", "boat", "vessel", "land"):
            W[m][f"photos_{g}_ge50"] = [r["seq"] for r in wr if r[f"whale_{m}_{g}"] >= 0.5]
        W[m]["photos_veto_ge50"] = [r["seq"] for r in wr if r[f"whale_{m}_person"] + r[f"whale_{m}_boat"] + r[f"whale_{m}_vessel"] >= 0.5]
        W[m]["photos_rock_ge50"] = [r["seq"] for r in wr if r[f"whale_{m}_rock"] >= 0.5]
        W[m]["by_class"] = {c: {"photos": sum(1 for r in wr if r["truth_class"] == c), "veto_ge50": sum(1 for r in wr if r["truth_class"] == c and r[f"whale_{m}_person"] + r[f"whale_{m}_boat"] + r[f"whale_{m}_vessel"] >= 0.5)} for c in ("whale", "whale_crew")}
    out["2c_whale_pixels"] = W
    # veto
    V = {}
    for m, per in veto.items():
        V[m] = {}
        for c, x in per.items():
            V[m][c] = {"photos": sum(1 for r in rows if r["truth_class"] == c), "live_px": int(x[0]), "live_removed": round(float(x[1] / max(x[0], 1)), 4),
                       "false_px": int(x[2]), "false_removed": round(float(x[3] / max(x[2], 1)), 4), "true_px": int(x[4]), "true_removed": round(float(x[5] / max(x[4], 1)), 4),
                       "s6_px": int(s6veto[m][c][0]), "s6_removed": round(float(s6veto[m][c][1] / max(s6veto[m][c][0], 1)), 4)}
            sel = [r for r in rows if r["truth_class"] == c and r["true_px"] > 0]
            V[m][c]["photos_true_ge50_removed"] = [r["seq"] for r in sel if r[f"veto_{m}_true_removed"] >= 0.5]
            sel = [r for r in rows if r["truth_class"] == c and r["live_px"] > 0]
            V[m][c]["photos_live_ge90_removed"] = sum(1 for r in sel if r[f"veto_{m}_live_removed"] >= 0.9); V[m][c]["photos_with_live"] = len(sel)
    out["2d_veto"] = V
    (a.out / "readymade_measures.json").write_text(json.dumps(out, indent=1)); print(json.dumps({k: out[k] for k in ("2a_auc_descriptive",)}, indent=0))


if __name__ == "__main__":
    main()
