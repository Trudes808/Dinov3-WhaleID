"""Measures of the live pipeline on an unsorted test day against the by-eye truth (all data in the untracked day folder).

    .venv/bin/python experiments/test_day/analyze.py --root /home/sat3737/NOAA_whales/moonshot_cache/test_days/2026-04-07 --keys INST_KEYS.json

Inputs (ROOT): truth.tsv, inst_eye.txt (by-eye label per instance, by index into INST_KEYS.json = instances sorted by
photo), days/<day>/{safe/photos.json, safe/not_whale.json, safe/clusters.json, safe/view_class.json, parts_hr/index.jsonl,
canon/*_mask.png}, openvocab.json and nw_probe.json when present. Outputs: ROOT/photos.tsv (pipeline answer per photo),
ROOT/instances.tsv, ROOT/measures.json (every number of tables.md). Nothing is fitted or tuned here.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

EYE = {"W": "whale", "P": "person", "R": "rail/boat", "G": "gear", "S": "water/splash", "F": "whale fused with crew/vessel", "L": "land", "V": "vessel"}


def read_eye(path: Path) -> dict[int, str]:
    lab = {}
    for l in open(path):
        if l.startswith("#") or not l.strip():
            continue
        r, c = l.split(); a, _, b = r.partition("-")
        for i in range(int(a), int(b or a) + 1):
            lab[i] = c
    return lab


def colour_rule(im: np.ndarray) -> tuple[str, dict]:
    """The rail-tape colour rule of prompts.json, fixed before looking at the markers' pixel values."""
    import cv2
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV); h, s, v = hsv[..., 0], hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    ok = (s >= 0.45) & (v >= 0.25); n = h.size
    frac = {"red": float((ok & ((h < 15) | (h >= 165))).sum() / n), "green": float((ok & (h >= 40) & (h <= 85)).sum() / n), "blue": float((ok & (h >= 95) & (h <= 130)).sum() / n)}
    best = sorted(frac.items(), key=lambda x: -x[1])
    return (best[0][0] if best[0][1] >= 0.005 and best[0][1] >= 2 * best[1][1] else "none"), frac


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True); p.add_argument("--keys", type=Path, required=True); a = p.parse_args()
    R = a.root; D = R / "days" / R.name
    T = {r["seq"]: r for r in csv.DictReader(open(R / "truth.tsv"), delimiter="\t")}
    P = json.loads((D / "safe" / "photos.json").read_text()); NW = json.loads((D / "safe" / "not_whale.json").read_text()); drop = set(NW["drop"])
    CL = json.loads((D / "safe" / "clusters.json").read_text())["clusters"]; VC = json.loads((D / "safe" / "view_class.json").read_text())
    parts = {it["key"]: it["parts"] for l in open(D / "parts_hr" / "index.jsonl") for it in json.loads(l)["instances"]}
    keys = json.loads(a.keys.read_text()); eye = read_eye(R / "inst_eye.txt"); ekey = {k: eye[i] for i, k in enumerate(keys)}
    ov = json.loads((R / "openvocab.json").read_text()) if (R / "openvocab.json").exists() else None
    pr = json.loads((R / "nw_probe.json").read_text()) if (R / "nw_probe.json").exists() else None
    seq = lambda stem: stem.split("_")[-2]
    cl_of = {k: c["id"] for c in CL for k in c["keys"]}; cl_size = {c["id"]: len(c["keys"]) for c in CL}
    cl_eye = {c["id"]: Counter(ekey.get(k, "?") for k in c["keys"]) for c in CL}
    thr = VC.get("threshold", 0.5)

    def view(k):
        v = VC["views"].get(k)
        if v is None:
            return ""
        return "fluke" if v[0] >= thr and v[0] >= v[1] else "dorsal" if v[1] >= thr else "other"

    # ---- instances
    irows = []
    for k in keys:
        stem = k.split("~")[0]; it = next(x for x in P[stem]["instances"] if x["key"] == k); c = cl_of.get(k)
        row = {"inst": keys.index(k), "seq": seq(stem), "key_suffix": k.split("~")[1], "eye": ekey[k], "truth_class": T[seq(stem)]["class"],
               "nw_score": NW["scores"].get(k), "flagged": int(k in drop), "area_px": it.get("area"), "bbox": ",".join(str(round(x)) for x in it["bbox"]),
               "chart_parts": "+".join(parts.get(k, [])), "view": view(k), "p_fluke": (VC["views"].get(k) or ["", ""])[0], "p_dorsal": (VC["views"].get(k) or ["", ""])[1],
               "cluster": "" if c is None else c, "cluster_size": "" if c is None else cl_size[c],
               "cluster_whales": "" if c is None else cl_eye[c].get("W", 0) + cl_eye[c].get("F", 0)}
        if ov:
            v = (ov["instances"].get(k) or {}).get("clip"); row["clip_name"] = ov["classes"][int(np.argmax(v))] if v else ""; row["clip_p"] = round(max(v), 3) if v else ""
        if pr:
            v = pr["instances"].get(k); row["probe_name"] = pr["classes"][int(np.argmax(v))] if v else ""; row["probe_p"] = round(max(v), 3) if v else ""
        irows.append(row)
    with open(R / "instances.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(irows[0]), delimiter="\t"); w.writeheader(); w.writerows(irows)

    # ---- photos
    prow = []; M = {}
    for stem in sorted(P, key=seq):
        s = seq(stem); t = T[s]; ins = P[stem]["instances"]; kept = [x for x in ins if x["key"] not in drop]
        m = np.array(Image.open(D / "canon" / f"{stem}_mask.png")) > 0
        im = np.array(Image.open(D / "images" / f"{stem}.jpg").convert("RGB"))
        col, frac = colour_rule(im)
        r = {"seq": s, "truth_class": t["class"], "has_whale": t["has_whale"], "n_inst": len(ins), "n_kept": len(kept),
             "answer": ("whale" if kept else ("no whale (filtered)" if ins else "no whale")), "mask_share": round(float(m.mean()), 5),
             "mask_share_kept": None, "rail_colour_rule": col, "red": round(frac["red"], 4), "green": round(frac["green"], 4), "blue": round(frac["blue"], 4)}
        if ov:
            ph = ov["photos"][stem]; r.update({f"clip_{c}": ph["clip"][j] for j, c in enumerate(ov["classes"])}); r.update({f"owl_{c}": v for c, v in ph["owl_best"].items()})
            r["clip_main"] = ov["classes"][int(np.argmax(ph["clip"]))]
        prow.append(r); M[s] = m
    with open(R / "photos.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(prow[0]), delimiter="\t"); w.writeheader(); w.writerows(prow)

    out = {}
    # (a) no whale per photo
    yes = [r for r in prow if r["has_whale"].rstrip("?") == "yes"]; no = [r for r in prow if r["has_whale"] == "no"]; und = [r for r in prow if r["has_whale"] == "under"]
    for name, sel in (("before_filter", lambda r: r["n_inst"] > 0), ("after_filter", lambda r: r["n_kept"] > 0)):
        out[f"a_{name}"] = {"truth_yes": len(yes), "yes_said_whale": sum(map(sel, yes)), "truth_no": len(no), "no_said_whale": sum(map(sel, no)),
                            "truth_under": len(und), "under_said_whale": sum(map(sel, und)),
                            "no_by_class": {c: [sum(1 for r in no if r["truth_class"] == c), sum(1 for r in no if r["truth_class"] == c and sel(r))] for c in sorted({r["truth_class"] for r in no})}}
    out["a_false_whale_photos"] = [r["seq"] for r in no if r["n_kept"] > 0]; out["a_missed_photos"] = [r["seq"] for r in yes if r["n_kept"] == 0]
    out["a_filtered_whale_photos"] = [r["seq"] for r in yes if r["n_inst"] > 0 and r["n_kept"] == 0]
    # (b) instances
    tab = defaultdict(lambda: [0, 0])
    for r in irows:
        tab[r["eye"]][0] += 1; tab[r["eye"]][1] += r["flagged"]
    out["b_eye_vs_flag"] = {EYE[k]: {"n": v[0], "flagged": v[1]} for k, v in sorted(tab.items())}
    kept_nw = [r for r in irows if r["eye"] != "W" and not r["flagged"]]
    out["b_kept_nonwhale"] = [{k: r[k] for k in ("inst", "seq", "eye", "nw_score", "view", "chart_parts", "cluster", "cluster_size", "cluster_whales")} for r in kept_nw]
    out["b_flagged_whale"] = [{k: r[k] for k in ("inst", "seq", "eye", "nw_score", "view")} for r in irows if r["eye"] == "W" and r["flagged"]]
    out["b_view_of_nonwhale"] = dict(Counter((EYE[r["eye"]], r["view"]) for r in irows if r["eye"] not in ("W",)).most_common()) if irows else {}
    out["b_view_of_nonwhale"] = {f"{a}|{b}": n for (a, b), n in Counter((EYE[r["eye"]], r["view"]) for r in irows if r["eye"] != "W").items()}
    out["b_chart_of_nonwhale"] = {f"{a}|{b}": n for (a, b), n in Counter((EYE[r["eye"]], r["chart_parts"]) for r in irows if r["eye"] != "W").items()}
    # (c) whale pixels on photos without a whale, by class
    seg = {}
    for c in sorted({r["truth_class"] for r in prow}):
        v = np.array([r["mask_share"] for r in prow if r["truth_class"] == c])
        seg[c] = {"n": int(len(v)), "mean_share": round(float(v.mean()), 4), "median_share": round(float(np.median(v)), 4), "photos_with_any": int((v > 0).sum()), "photos_over_1pct": int((v > 0.01).sum())}
    out["c_mask_share_by_class"] = seg
    # whale-mask share inside OWL boxes (score >= 0.2) of person / railing / land in photos whose truth has that thing
    if ov:
        inside = defaultdict(list)
        for stem in P:
            s = seq(stem); t = T[s]; m = M[s]; H, W = m.shape
            want = {"person": t["person"] == "1", "railing": t["rail"] == "1" or t["deck"] == "1", "land": t["land_subj"] == "1" or t["land_bg"] == "1", "vessel": t["vessel"] == "1"}
            for c, flag in want.items():
                if not flag:
                    continue
                bx = [d[3] for d in ov["photos"][stem]["owl_det"] if d[0] == c and d[2] >= 0.2]
                if not bx:
                    continue
                reg = np.zeros_like(m)
                for x0, y0, x1, y1 in bx:
                    reg[max(0, int(y0)):min(H, int(y1)), max(0, int(x0)):min(W, int(x1))] = True
                inside[c].append((s, float(m[reg].mean()), t["has_whale"]))
        out["c_mask_inside_owl_boxes"] = {c: {"photos": len(v), "mean_share": round(float(np.mean([x[1] for x in v])), 4), "median": round(float(np.median([x[1] for x in v])), 4),
                                              "over_10pct": sum(1 for x in v if x[1] > 0.1), "seqs_over_10pct": [x[0] for x in v if x[1] > 0.1]} for c, v in inside.items()}
    # (d) naming at photo level
    if ov:
        truth_has = {"person": lambda t: t["person"] == "1", "railing": lambda t: t["rail"] == "1", "land": lambda t: t["land_subj"] == "1",
                     "land_any": lambda t: t["land_subj"] == "1" or t["land_bg"] == "1", "vessel": lambda t: t["vessel"] == "1",
                     "whale": lambda t: t["has_whale"].rstrip("?") == "yes", "water": lambda t: t["class"] == "water"}
        d = {}
        for name, f in truth_has.items():
            c = "land" if name == "land_any" else name
            for model, pred in (("clip_p>=0.2", lambda r: r[f"clip_{c}"] >= 0.2), ("clip_main", lambda r: r["clip_main"] == c),
                                ("owl>=0.2", (lambda r: r[f"owl_{c}"] >= 0.2) if f"owl_{c}" in prow[0] else None)):
                if pred is None:
                    continue
                tp = sum(1 for r in prow if f(T[r["seq"]]) and pred(r)); fn = sum(1 for r in prow if f(T[r["seq"]]) and not pred(r)); fp = sum(1 for r in prow if not f(T[r["seq"]]) and pred(r))
                d[f"{name}|{model}"] = {"truth_n": tp + fn, "said_n": tp + fp, "recall": round(tp / max(tp + fn, 1), 3), "precision": round(tp / max(tp + fp, 1), 3),
                                        "missed": [r["seq"] for r in prow if f(T[r["seq"]]) and not pred(r)][:40], "false": [r["seq"] for r in prow if not f(T[r["seq"]]) and pred(r)][:40]}
        out["d_photo_naming"] = d
        cm = Counter((EYE[r["eye"]], r["clip_name"]) for r in irows); out["d_instance_clip"] = {f"{a}|{b}": n for (a, b), n in sorted(cm.items())}
    if pr:
        cm = Counter((EYE[r["eye"]], r["probe_name"]) for r in irows); out["d_instance_probe"] = {f"{a}|{b}": n for (a, b), n in sorted(cm.items())}
        out["d_probe_train_cv"] = pr["train_cv_by_day_group"]
    # rail colour rule
    rc = {}
    for r in prow:
        t = T[r["seq"]]; truth_col = t["marker"].replace("rail_", "").rstrip("?") if t["class"] == "marker_rail" else "none"
        rc.setdefault(f"truth {truth_col}", Counter())[r["rail_colour_rule"]] += 1
    out["d_rail_colour_rule"] = {k: dict(v) for k, v in rc.items()}
    out["d_rail_colour_rule_markers"] = {r["seq"]: [T[r["seq"]]["marker"], r["rail_colour_rule"], r["red"], r["green"], r["blue"]] for r in prow if T[r["seq"]]["class"] == "marker_rail"}
    # (f) removal candidates
    f = defaultdict(lambda: Counter())
    for r in prow:
        t = T[r["seq"]]; reason = t["cull_reason"] or ("keep" if t["cull_guess"] == "keep" else "keep?")
        key = f"{t['cull_guess']}:{reason}"
        f[key]["n"] += 1; f[key]["no_instance"] += r["n_inst"] == 0; f[key]["no_kept_instance"] += r["n_kept"] == 0
        f[key]["some_flagged"] += r["n_kept"] < r["n_inst"]
    out["f_cull"] = {k: dict(v) for k, v in sorted(f.items(), key=lambda x: -x[1]["n"])}
    (R / "measures.json").write_text(json.dumps(out, indent=1, default=str)); print(json.dumps(out, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
