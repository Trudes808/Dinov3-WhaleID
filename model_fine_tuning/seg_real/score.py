"""Scores of cached predictions (seg_real.predict) on real photos: 3a labelled blobs, 3b whales that must not get worse.

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.seg_real.score --names r04,r05,s1,s2,s3
        -> experiments/seg_splash/score_blobs.json, score_whales.json; tables on stdout

3a  per category and split (val = headline, never trained on; train = the variants' own training blobs): share of
    the blob's pixels with P(whale) >= 0.5 (pixels pooled over blobs), and the share of blobs with >= 30 % of their
    pixels still called whale. Blob masks are the seg_r04_ft4 blobs taken to the 1280 scale (INTER_AREA, >= 0.5).
3b  (i) photos whose main whale is certain (view label F / D / O on its instance, which maps to ONE r04 blob): IoU of
    the component of the new mask that overlaps that blob most, against the blob; share of the blob kept.
    (ii) the researcher's clicks (2025-01-02, 2025-03-18, training days): share of clicked points (main whale; other
    whales) within 12.5 px of the 1280-px mask (= 25 px of the 2560 frame, the instances_v2 tolerance).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import cv2
import numpy as np

from . import common as C
from .data import blob_1280
from .predict import CLICK_DAYS, load_marks, load_prob, view_keys

GROUPS = ["whale", "whale_splash", "splash", "wake", "water", "shore", "boat", "person", "pole", "drone", "bird", "other"]


def blob_scores(name: str) -> dict:
    out = defaultdict(lambda: defaultdict(list))        # split -> label -> [(called, total)]
    by = defaultdict(list)
    for b in C.load_labels():
        by[(b["day"], b["stem"])].append(b)
    per = {}
    for (day, stem), bs in by.items():
        p = load_prob(name, day, stem) >= 0.5; L = C.blob_image(day, stem)
        for b in bs:
            m = blob_1280(L, b["b"], p.shape); n = int(m.sum())
            if n == 0:
                continue
            c = int(p[m].sum()); per[C.key_of(day, stem, b["b"])] = round(c / n, 4)
            out[b["split"]][b["label"]].append((c, n))
            if b["label"] == "whale" and b["sure"]:
                out[b["split"]]["whale (sure)"].append((c, n))
            if b["label"] in C.NEG:
                out[b["split"]]["ALL non-whale"].append((c, n))
    res = {}
    for sp, d in out.items():
        res[sp] = {}
        for lab, v in d.items():
            c = np.array([x[0] for x in v], float); n = np.array([x[1] for x in v], float)
            res[sp][lab] = {"blobs": len(v), "pixel_share_whale": round(float(c.sum() / n.sum()), 4), "blobs_ge30": round(float(np.mean(c / n >= 0.3)), 4),
                            "blobs_ge30_n": int(np.sum(c / n >= 0.3)),
                            "blobs_ge50": round(float(np.mean(c / n >= 0.5)), 4), "mean_blob_share": round(float(np.mean(c / n)), 4)}
    return {"summary": res, "per_blob": per}


def whale_scores(name: str, ref: str = "r04") -> dict:
    res = {"_per_photo": []}
    # (i) view-labelled main whales
    vk = view_keys(); rows = defaultdict(list)
    for k, (day, code) in vk.items():
        rows[day].append(k)
    for sp in ("train", "val"):
        iou, kept = [], []
        for day in [d for d in rows if C.split_of(d) == sp]:
            mp = json.loads((C.DAYS / day / "instances_v2" / "mapping.json").read_text())["old_to_new"]
            idx = {json.loads(l)["stem"]: json.loads(l) for l in open(C.DAYS / day / "instances_v2" / "index.jsonl")}
            for k in rows[day]:
                m = mp.get(k); stem = k.rsplit("~", 1)[0]
                if not m or not m["new"] or stem not in idx:
                    continue
                inst = [i for i in idx[stem]["instances"] if i["key"] == m["new"]]
                if not inst or len(inst[0]["blobs"]) != 1:
                    continue
                p = load_prob(name, day, stem) >= 0.5; old = blob_1280(C.blob_image(day, stem), inst[0]["blobs"][0], p.shape)
                if old.sum() < 50:
                    continue
                n, lab = cv2.connectedComponents(p.astype(np.uint8), connectivity=8)
                ids, cnt = np.unique(lab[old], return_counts=True); cnt = cnt[ids > 0]; ids = ids[ids > 0]
                if len(ids) == 0:
                    iou.append(0.0); kept.append(0.0)
                else:
                    comp = lab == ids[np.argmax(cnt)]
                    iou.append(float((comp & old).sum() / (comp | old).sum())); kept.append(float(p[old].mean()))
                res["_per_photo"].append({"day": day, "stem": stem, "b": inst[0]["blobs"][0], "key": k, "code": vk[k][1], "split": sp, "iou": round(iou[-1], 4), "kept": round(kept[-1], 4)})
        iou, kept = np.array(iou), np.array(kept)
        res[f"views_{sp}"] = {"photos": len(iou), "median_iou": round(float(np.median(iou)), 4), "mean_iou": round(float(iou.mean()), 4),
                              "iou_lt_0.8": round(float((iou < 0.8).mean()), 4), "iou_lt_0.5": round(float((iou < 0.5).mean()), 4),
                              "mean_kept": round(float(kept.mean()), 4), "kept_lt_0.9": round(float((kept < 0.9).mean()), 4)}
    # (ii) clicks
    tot = {"main": [0, 0], "other": [0, 0]}; whales_main, whales_other = [], []
    for day in CLICK_DAYS:
        photos, others = load_marks(day)
        for key, kp in photos.items():
            stem = key.rsplit("~", 1)[0]
            p = load_prob(name, day, stem) >= 0.5; H, W = p.shape
            dist = cv2.distanceTransform((~p).astype(np.uint8), cv2.DIST_L2, 3)
            for who, pts in [("main", kp)] + [("other", w) for w in (others.get(key) or []) if w]:
                hit = [dist[min(H - 1, max(0, int(q["y"] * H))), min(W - 1, max(0, int(q["x"] * W)))] <= 12.5 for q in pts.values()]
                tot[who][0] += sum(hit); tot[who][1] += len(hit)
                (whales_main if who == "main" else whales_other).append(all(hit))
    res["clicks_train_days"] = {"main_points": tot["main"][1], "main_points_in_mask": round(tot["main"][0] / max(1, tot["main"][1]), 4),
                                "main_whales_all_points_in": round(float(np.mean(whales_main)), 4), "other_whales": len(whales_other),
                                "other_points_in_mask": round(tot["other"][0] / max(1, tot["other"][1]), 4),
                                "other_whales_all_points_in": round(float(np.mean(whales_other)), 4) if whales_other else None}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", default="r04,r05"); a = ap.parse_args(argv)
    names = a.names.split(",")
    fb, fw = C.EXP / "score_blobs.json", C.EXP / "score_whales.json"
    B = json.loads(fb.read_text()) if fb.exists() else {}; Wd = json.loads(fw.read_text()) if fw.exists() else {}
    per = {}
    for n in names:
        r = blob_scores(n); B[n] = r["summary"]; per[n] = r["per_blob"]; Wd[n] = whale_scores(n)
    pp = C.CACHE / "per_photo_whales.json"; PP = json.loads(pp.read_text()) if pp.exists() else {}
    for n in names:
        PP[n] = Wd[n].pop("_per_photo", [])
    pp.write_text(json.dumps(PP))
    fb.write_text(json.dumps(B, indent=1)); fw.write_text(json.dumps(Wd, indent=1))
    (C.CACHE / "per_blob_scores.json").write_text(json.dumps({**(json.loads((C.CACHE / "per_blob_scores.json").read_text()) if (C.CACHE / "per_blob_scores.json").exists() else {}), **per}))
    for sp in ("val", "train"):
        print(f"\n3a {sp}: pixel share called whale / blobs with >= 30 % called whale")
        print(f"{'category':16s} {'n':>5s} " + " ".join(f"{n:>14s}" for n in names))
        for g in GROUPS + ["whale (sure)", "ALL non-whale"]:
            if g not in B[names[0]].get(sp, {}):
                continue
            print(f"{g:16s} {B[names[0]][sp][g]['blobs']:5d} " + " ".join(f"{B[n][sp][g]['pixel_share_whale']:6.3f}/{B[n][sp][g]['blobs_ge30']:6.3f}" for n in names))
    print("\n3b")
    for k in [k for k in Wd[names[0]] if not k.startswith("_")]:
        print(k)
        for n in names:
            print("   ", n, Wd[n][k])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
