"""The one read of the test day 2026-04-07 with SAM 3, at the settings frozen in rules_sam3.md part B (passed as a json:
--settings). Photo level (all 427), pixel level (the frozen pixel truth), instances (whale + crew fusion), extra classes,
markers, tape colour. Per-photo rows go to the untracked OUT folder only.

    PYTHONPATH=. .venv/bin/python experiments/sam3_compare/analyze_test.py --settings SETTINGS.json --out /home/sat3737/NOAA_whales/moonshot_cache/sam3_compare/test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import s3common as C  # noqa: E402

PT = C.TD / "pixel_truth"
TH, TW = 1835, 2748
TRUTH = ["water_sky", "whale", "person", "rail", "tape", "deck", "pontoon", "other_vessel", "land", "gear"]
CLS_T = {"whale": [1], "person": [2], "rail": [3], "tape": [4], "deck": [5], "pontoon": [6], "other_vessel": [7], "land": [8], "gear": [9],
         "rail_tape": [3, 4], "our_boat": [3, 4, 5, 6], "any_boat": [3, 4, 5, 6, 7]}
G = (427, 640)
NOWHALE = ["marker_rail", "bow", "crew_nowhale", "marker_land", "water"]
# truth.tsv tag -> SAM 3 class used for the photo-level table
TAGS = {"person": "person", "rail": "rail", "deck": "deck", "gear": "gear", "vessel": "other_vessel", "land_subj": "land", "land_bg": "land"}


def check_frozen() -> dict:
    m = json.loads((PT / "manifest.json").read_text())
    for f, h in m["frozen"]["sha256"].items():
        if hashlib.sha256((PT / f).read_bytes()).hexdigest() != h:
            raise SystemExit(f"truth file changed since freeze: {f}")
    return m


def prf(tp, fp, fn):
    f = lambda x, y: round(x / y, 4) if y else None
    return {"iou": f(tp, tp + fp + fn), "recall": f(tp, tp + fn), "precision": f(tp, tp + fp)}


def hue_class(h_deg: float) -> str:
    if h_deg < 20 or h_deg >= 320:
        return "red"
    if h_deg < 70:
        return "yellow"
    if h_deg < 170:
        return "green"
    if h_deg < 260:
        return "blue"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--settings", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(); S = json.loads(a.settings.read_text()); out = a.out; out.mkdir(parents=True, exist_ok=True)
    man = check_frozen(); truth = C.truth_tsv(); idx = C.prompt_index("test")
    cls_prompts = {c: [idx[p] for p in ps] for c, ps in S["class_prompts"].items()}   # class -> prompt indices (union)
    cls_thr = S["class_thr"]
    wj = cls_prompts["whale"]; wthr = cls_thr["whale"]
    alt = {c: [idx[p] for p in ps] for c, ps in C.PROMPTS["groups"].items()}
    seqs = sorted(truth)
    rows = []
    for seq in seqs:
        rec = C.load("test", seq); assert rec is not None, seq
        t = truth[seq]
        lv = C.test_live(seq, G); s6 = C.test_s6(seq, G)
        r = {"seq": seq, "class": t["class"], "marker": t["marker"]}
        sc, bx, ms = [], [], []
        for j in wj:
            s_, b_, m_ = C.inst(rec, j, wthr, G); sc += list(s_); ms += m_
        u = np.any(ms, axis=0) if ms else np.zeros(G, bool)
        r["sam3_n"] = len(ms); r["sam3_px_share"] = round(float(u.mean()), 5)
        r["sam3_on_ours"] = round(float((u & (lv | s6)).sum() / max(u.sum(), 1)), 4)
        r["sam3_inst_on_ours"] = [round(float((m & (lv | s6)).sum() / max(m.sum(), 1)), 3) for m in ms]
        r["sam3_scores"] = [round(float(x), 3) for x in sc]
        r["live_px_share"] = round(float(lv.mean()), 5); r["s6_px_share"] = round(float(s6.mean()), 5)
        r["s6_on_sam3"] = round(float((s6 & u).sum() / max(s6.sum(), 1)), 4)
        r["iou_sam3_s6"] = round(float((u & s6).sum() / max((u | s6).sum(), 1)), 4)
        # every phrasing at its class threshold: share of frame, number of instances, max score
        for c, js in alt.items():
            th = cls_thr.get(c, 0.5)
            for j in js:
                s_ = rec[f"s{j}"]; r[f"n:{C.PROMPTS['prompts'][j]}"] = int((s_ >= th).sum())
                r[f"a:{C.PROMPTS['prompts'][j]}"] = round(float(C.union(rec, j, th, G).mean()), 5)
                r[f"max:{C.PROMPTS['prompts'][j]}"] = round(float(s_.max()), 3) if len(s_) else 0.0
        for c, js in cls_prompts.items():
            r[f"cls:{c}"] = round(float(C.union(rec, js, cls_thr[c], G).mean()), 5)
        bj = [idx[p] for p in S["marker_rule"]["bollard_prompts"]]
        r["bollard_area"] = round(float(C.union(rec, bj, S["marker_rule"]["bollard_score"], G).mean()), 5)
        r["bollard_max"] = round(float(max([rec[f"s{j}"].max() for j in bj if len(rec[f"s{j}"])] or [0.0])), 3)
        rows.append(r)
    R = {r["seq"]: r for r in rows}
    res = {"settings": S, "pixel_truth_freeze": man["frozen"]["time"] if isinstance(man["frozen"], dict) else str(man["frozen"])[:60]}

    # 1a photo level
    groups = {"whale above surface": ["whale", "whale_crew"], "- whale": ["whale"], "- whale + crew": ["whale_crew"], "under water only": ["under"],
              "no whale": NOWHALE}
    groups.update({f"- {c}": [c] for c in NOWHALE})
    pa = {}
    for g, cs in groups.items():
        rr = [r for r in rows if r["class"] in cs]
        pa[g] = {"n": len(rr), "sam3_any_instance": sum(r["sam3_n"] > 0 for r in rr),
                 "sam3_instance_on_ours": sum(any(x >= 0.3 for x in r["sam3_inst_on_ours"]) for r in rr),
                 "live_any_px": sum(r["live_px_share"] > 0 for r in rr), "s6_any_px": sum(r["s6_px_share"] > 0 for r in rr),
                 "sam3_seqs_none": [r["seq"] for r in rr if r["sam3_n"] == 0] if len(rr) < 60 or cs[0] in ("whale", "whale_crew") else None,
                 "sam3_seqs_any": [r["seq"] for r in rr if r["sam3_n"] > 0] if cs[0] not in ("whale", "whale_crew") else None}
    res["photo_level"] = pa

    # pixel truth
    sample = man["sample"] if isinstance(man["sample"], list) else json.loads(man["sample"].replace("'", '"'))
    tclass = man["truth_class"] if isinstance(man["truth_class"], dict) else json.loads(man["truth_class"].replace("'", '"'))
    acc = defaultdict(lambda: defaultdict(int)); conf = defaultdict(lambda: np.zeros(11, np.int64)); fuse = []
    tape_rows = []; pix_rows = []
    for seq in sample:
        t = np.array(Image.open(PT / "labels" / f"{seq}.png")); valid = t != 255; tr = np.where(t == 255, 10, t)
        rec = C.load("test", seq); hw = (TH, TW)
        maps = {"live": C.test_live(seq, hw), "s6h": C.test_s6(seq, hw)}
        wsc, wms = [], []
        for j in wj:
            s_, _, m_ = C.inst(rec, j, wthr, hw); wsc += list(s_); wms += m_
        maps["sam3"] = np.any(wms, axis=0) if wms else np.zeros(hw, bool)
        for c, js in cls_prompts.items():
            if c != "whale":
                maps[f"sam3:{c}"] = C.union(rec, js, cls_thr[c], hw)
        for c, js in alt.items():
            for j in js:
                maps[f"alt:{C.PROMPTS['prompts'][j]}"] = C.union(rec, j, cls_thr.get(c, 0.5), hw)
        for name, pm in maps.items():
            if name in ("live", "s6h", "sam3"):
                conf[name] += np.bincount(tr[pm].ravel(), minlength=11)
                tm = (t == 1) & valid; pairs = [("whale", tm)]
            else:
                c = name.split(":", 1)[1]
                cc = c if name.startswith("sam3:") else next(k for k, v in C.PROMPTS["groups"].items() if c in v)
                if name.startswith("sam3:"):
                    conf[name] += np.bincount(tr[pm].ravel(), minlength=11)
                pairs = [(cc, np.isin(t, CLS_T[cc]) & valid)]
                if cc == "other_vessel":
                    pairs.append(("any_boat", np.isin(t, CLS_T["any_boat"]) & valid))
                if cc == "rail":
                    pairs.append(("rail_tape", np.isin(t, CLS_T["rail_tape"]) & valid))
                if cc in ("deck", "pontoon"):
                    pairs.append(("our_boat", np.isin(t, CLS_T["our_boat"]) & valid))
            for pn, tm in pairs:
                p2 = pm & valid
                A = acc[(name, pn)]
                tp, fp, fn = int((tm & p2).sum()), int((~tm & p2).sum()), int((tm & ~p2).sum())
                A["tp"] += tp; A["fp"] += fp; A["fn"] += fn; A["n_truth"] += int(tm.any()); A["n_pred"] += int(p2.any())
                if name in ("live", "s6h", "sam3"):
                    pix_rows.append((seq, tclass[seq], name, tp, fp, fn))
        # whale + crew fusion, per instance (SAM 3 instances de-overlapped; ours = 8-connected components)
        if tclass[seq] == "whale_crew":
            def fused(masks):
                n = 0
                for m in masks:
                    v = m & valid; tot = v.sum()
                    if tot == 0:
                        continue
                    w = (v & (t == 1)).sum(); p = (v & (t == 2)).sum()
                    n += (w >= 200 and p >= 200 and w >= 0.1 * tot and p >= 0.1 * tot)
                return int(n)
            def comps(m):
                k, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
                return [lab == i for i in range(1, k)]
            fuse.append({"seq": seq, "sam3_inst": len(wms), "sam3_fused": fused(C.deoverlap(wsc, wms)) if wms else 0,
                         "s6_fused_cc": fused(comps(maps["s6h"])), "live_fused_cc": fused(comps(maps["live"])),
                         "sam3_px_on_person": int((maps["sam3"] & (t == 2)).sum()), "s6_px_on_person": int((maps["s6h"] & (t == 2)).sum()),
                         "live_px_on_person": int((maps["live"] & (t == 2)).sum())})
    pix = {f"{n}|{p}": dict(prf(A["tp"], A["fp"], A["fn"]), n_truth=A["n_truth"], n_pred=A["n_pred"], truth_px=A["tp"] + A["fn"], pred_px=A["tp"] + A["fp"])
           for (n, p), A in acc.items()}
    res["pixel"] = pix
    res["confusion_rownorm_of_pred"] = {n: {TRUTH[i] if i < 10 else "unsure_255": round(float(c[i] / max(c.sum(), 1)), 4) for i in range(11)} for n, c in conf.items()}
    res["confusion_px"] = {n: [int(x) for x in c] for n, c in conf.items()}
    res["fusion_whale_crew"] = {"n_photos": len(fuse), "sam3_photos_fused": sum(f["sam3_fused"] > 0 for f in fuse),
                                "s6_photos_fused_cc": sum(f["s6_fused_cc"] > 0 for f in fuse), "live_photos_fused_cc": sum(f["live_fused_cc"] > 0 for f in fuse),
                                "sam3_px_on_person": sum(f["sam3_px_on_person"] for f in fuse), "s6_px_on_person": sum(f["s6_px_on_person"] for f in fuse),
                                "live_px_on_person": sum(f["live_px_on_person"] for f in fuse)}

    # photo level, extra classes (X = class pixels >= 0.5 % of the frame, round 2's rule)
    pl = {}
    for tag, c in TAGS.items():
        pos = {s for s in seqs if truth[s][tag] == "1"}
        key = f"cls:{c}"
        if tag == "land_subj":
            pred = {s for s in seqs if R[s][key] >= 0.25}
        else:
            pred = {s for s in seqs if R[s][key] >= 0.005}
        pl[tag] = {"n_truth": len(pos), "n_pred": len(pred), "recall": round(len(pos & pred) / max(len(pos), 1), 3),
                   "precision": round(len(pos & pred) / max(len(pred), 1), 3) if pred else None}
    res["photo_level_classes"] = pl

    # markers (rule of part B)
    M = S["marker_rule"]
    def marker_call(r, empty_by, variant="primary"):
        empty = (r["sam3_n"] == 0) if empty_by == "sam3" else (r["s6_px_share"] == 0)
        if not empty:
            return "whale"
        if r["cls:land"] >= M["land_min"]:
            return "land"
        if r["bollard_max"] >= M["bollard_score"] and r["bollard_area"] >= M["bollard_min"]:
            return "cleat"
        rt = r["cls:rail"] + r["cls:tape"]
        if (r["cls:tape"] >= M["tape_min"] and rt >= M["rail_tape_min"]) if variant == "primary" else rt >= M["rail_tape_min_variant"]:
            return "rail"
        if r["cls:person"] >= M["person_min"]:
            return "crew"
        return "water"
    mk = {}
    tm_ = {"marker_rail": "rail", "bow": "cleat", "marker_land": "land"}
    for empty_by, variant in (("sam3", "primary"), ("s6h", "primary"), ("sam3", "variant"), ("s6h", "variant")):
        calls = {s: marker_call(R[s], empty_by, variant) for s in seqs}
        tab = {}
        for tc in ("marker_rail", "bow", "marker_land", "crew_nowhale", "water", "under", "whale_crew", "whale"):
            ss = [s for s in seqs if truth[s]["class"] == tc]
            tab[tc] = {"n": len(ss), **dict(Counter(calls[s] for s in ss))}
        true_m = [s for s in seqs if truth[s]["class"] in tm_]
        other = [s for s in seqs if truth[s]["class"] not in tm_]
        tab["summary"] = {"markers_found_any_marker_call": sum(calls[s] in ("rail", "cleat", "land") for s in true_m), "n_markers": len(true_m),
                          "markers_right_kind": sum(calls[s] == tm_[truth[s]["class"]] for s in true_m),
                          "false_markers_on_other": sum(calls[s] in ("rail", "cleat", "land") for s in other), "n_other": len(other),
                          "false_marker_seqs": [s for s in other if calls[s] in ("rail", "cleat", "land")]}
        tab["calls_on_markers"] = {s: calls[s] for s in true_m}
        mk[f"{empty_by}_{variant}"] = tab
    res["markers"] = mk

    # tape colour (rule of part B): pixels under the tape mask, in the full-res photo
    tc_truth = {l.split("\t")[0]: l.split("\t")[1].strip() for l in open(PT / "tape_colours.tsv") if not l.startswith("seq")}
    T = S["tape_rule"]; tape_out = {}
    for seq in sorted(tc_truth) + [s for s in seqs if s not in tc_truth]:
        rec = C.load("test", seq)
        full = cv2.imread(str(C.TD / "fullres" / f"{C.test_stems()[seq]}.jpg"))
        hw = full.shape[:2]
        sc_, _, ms_ = [], [], []
        for j in cls_prompts["tape"]:
            s_, _, m_ = C.inst(rec, j, cls_thr["tape"], hw); ms_ += m_
        if not ms_:
            tape_out[seq] = "none"; continue
        m = np.any(ms_, axis=0)
        k = max(1, int(round(T["erode_frac"] * hw[1])))
        m = cv2.erode(m.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
        hsv = cv2.cvtColor(full, cv2.COLOR_BGR2HSV)
        H_ = hsv[..., 0][m].astype(np.float64) * 2; S_ = hsv[..., 1][m] / 255.0; V_ = hsv[..., 2][m] / 255.0
        col = (S_ >= T["s_min"]) & (V_ >= T["v_min"])
        if col.sum() < T["min_px"]:
            tape_out[seq] = "none"; continue
        ang = np.deg2rad(H_[col]); hm = (np.rad2deg(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) + 360) % 360
        tape_out[seq] = hue_class(hm)
    tt = {}
    for seq, c in tc_truth.items():
        tt[seq] = {"truth": c, "sam3": tape_out[seq], "right": c.split()[0] == tape_out[seq]}
    others = [s for s in seqs if s not in tc_truth]
    res["tape_colour"] = {"markers": tt, "right": sum(v["right"] for v in tt.values()), "n": len(tt),
                          "none_on_markers": sum(v["sam3"] == "none" for v in tt.values()),
                          "colour_calls_on_other_photos": sum(tape_out[s] != "none" for s in others), "n_other": len(others),
                          "colour_calls_on_other_by_class": dict(Counter(truth[s]["class"] for s in others if tape_out[s] != "none"))}
    (out / "test_results.json").write_text(json.dumps(res, indent=1, default=str))
    (out / "test_per_photo.json").write_text(json.dumps(rows))
    (out / "test_fusion.json").write_text(json.dumps(fuse))
    with open(out / "test_pixel_per_photo.tsv", "w") as f:
        f.write("seq\tclass\tmodel\ttp\tfp\tfn\n")
        for r in pix_rows:
            f.write("\t".join(map(str, r)) + "\n")
    print(json.dumps({"photo_level": {k: (v["n"], v["sam3_any_instance"], v["s6_any_px"], v["live_any_px"]) for k, v in pa.items()},
                      "whale_pixel": {k: v for k, v in pix.items() if k.split("|")[0] in ("live", "s6h", "sam3")},
                      "markers": {k: v["summary"] for k, v in mk.items()}, "tape": {k: res["tape_colour"][k] for k in ("right", "n", "none_on_markers", "colour_calls_on_other_photos")}}, indent=1))


if __name__ == "__main__":
    main()
