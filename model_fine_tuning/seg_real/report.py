"""Tables of the seg_splash scores (markdown) -> stdout and experiments/seg_splash/tables.md.

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.seg_real.report --names r04,r05,s0,s1,s2,s3
"""
from __future__ import annotations

import argparse
import json

from . import common as C

SMALL = ["person", "drone", "boat", "pole", "shore", "bird"]
ORDER = ["whale (sure)", "whale", "whale_splash", "splash", "wake", "water", "shore", "boat", "person", "pole", "drone", "bird", "ALL non-whale"]
EXPDIR = {"s0": "seg_splash_s0", "s1": "seg_splash_s1", "s2": "seg_splash_s2", "s3": "seg_splash_s3", "s4": "seg_splash_s4", "s5": "seg_splash_s5", "s5h": "seg_splash_s5", "s5b": "seg_splash_s5b", "s6": "seg_splash_s6", "s6h": "seg_splash_s6", "r05": "seg_r05_ft4", "r04": "seg_r04_ft4"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--names", default="r04,r05,s0,s1,s2,s3"); ap.add_argument("--out", default="tables.md"); a = ap.parse_args(argv)
    B = json.loads((C.EXP / "score_blobs.json").read_text()); W = json.loads((C.EXP / "score_whales.json").read_text())
    names = [n for n in a.names.split(",") if n in B]; out = []
    for sp in ("val", "train"):
        out.append(f"\n### 3a {sp} days: share of blob pixels called whale / share of blobs with >= 30 % called whale\n")
        out.append("| category | n | " + " | ".join(names) + " |"); out.append("|---" * (len(names) + 2) + "|")
        for g in ORDER:
            if g in B[names[0]].get(sp, {}):
                out.append(f"| {g} | {B[names[0]][sp][g]['blobs']} | " + " | ".join(f"{B[n][sp][g]['pixel_share_whale']:.2f} / {B[n][sp][g]['blobs_ge30']:.2f}" for n in names) + " |")
    out.append("\n### 3a small groups, validation days: blobs with >= 30 % still called whale (count of n)\n")
    out.append("| category | n | " + " | ".join(names) + " |"); out.append("|---" * (len(names) + 2) + "|")
    for g in SMALL:
        if g in B[names[0]].get("val", {}):
            out.append(f"| {g} | {B[names[0]]['val'][g]['blobs']} | " + " | ".join(str(B[n]["val"][g].get("blobs_ge30_n", round(B[n]["val"][g]["blobs_ge30"] * B[n]["val"][g]["blobs"]))) for n in names) + " |")
    out.append("\n### 3b whales that must not get worse\n")
    out.append("| measure | " + " | ".join(names) + " |"); out.append("|---" * (len(names) + 1) + "|")
    for k, f in (("views_val", "median_iou"), ("views_val", "iou_lt_0.8"), ("views_val", "mean_kept"), ("views_val", "kept_lt_0.9"),
                 ("views_train", "median_iou"), ("views_train", "iou_lt_0.8"), ("views_train", "mean_kept"), ("views_train", "kept_lt_0.9"),
                 ("clicks_train_days", "main_points_in_mask"), ("clicks_train_days", "main_whales_all_points_in"), ("clicks_train_days", "other_points_in_mask")):
        out.append(f"| {k} {f} | " + " | ".join(f"{W[n][k][f]:.3f}" for n in names if n in W) + " |")
    out.append(f"\n(views_val photos {W[names[0]]['views_val']['photos']}, views_train {W[names[0]]['views_train']['photos']}, clicked main whales on 2 training days, "
               f"main points {W[names[0]]['clicks_train_days']['main_points']}, other whales {W[names[0]]['clicks_train_days']['other_whales']})")
    e = C.EXP / "e2e.json"
    if e.exists():
        E = json.loads(e.read_text()); en = ["ref"] + [n for n in names if n in E]
        out.append("\n### 3c end to end (instances_v2 unchanged; ref = seg_r04_ft4)\n")
        out.append("| measure | " + " | ".join(en) + " |"); out.append("|---" * (len(en) + 1) + "|")
        for grp in ("val", "click_days"):
            for f in ("photos", "instances_per_photo", "inst_0", "inst_1", "inst_2", "inst_3+", "rejected_nw", "rejected_gate"):
                out.append(f"| {grp} {f} | " + " | ".join(str(E[n][grp][f]) for n in en) + " |")
        for f in ("own", "fused", "none", "flukes_main_one_instance", "main_all_clicks_one_instance"):
            out.append(f"| clicks {f} | " + " | ".join(str(E[n]["clicks"][f]) for n in en) + " |")
        out.append(f"\n(17 other whales; {E['ref']['clicks']['flukes_main']} main-whale flukes with both tips clicked; {E['ref']['clicks']['main_whales']} clicked main whales)")
    nb = C.EXP / "new_blobs.json"
    if nb.exists():
        N = json.loads(nb.read_text()); nn = [n for n in names if n in N]
        out.append("\n### new blobs: >= 300 px, overlapping no r04 blob, 884 validation photos\n")
        out.append("| measure | " + " | ".join(nn) + " |"); out.append("|---" * (len(nn) + 1) + "|")
        out.append("| new blobs | " + " | ".join(str(N[n]["new_blobs"]) for n in nn) + " |")
        out.append("| per photo | " + " | ".join(f"{N[n]['per_photo']:.3f}" for n in nn) + " |")
        out.append("| sizes | " + " | ".join(str(N[n]["sizes"]) for n in nn) + " |")
        out.append("| sample labelled | " + " | ".join(str(N[n].get("labels", {}).get("labelled", "-")) for n in nn) + " |")
        out.append("| whale share of sample | " + " | ".join(str(N[n].get("labels", {}).get("whale_share", "-")) for n in nn) + " |")
        out.append("| est. non-whale new blobs / photo | " + " | ".join(str(N[n].get("labels", {}).get("nonwhale_new_per_photo", "-")) for n in nn) + " |")
        out.append("| sample labels | " + " | ".join(str(N[n].get("labels", {}).get("counts", "-")) for n in nn) + " |")
    out.append("\n### 3d synthetic check: run05 validation renders (1 379), dataset IoU\n")
    for n in names:
        d = C.REPO / "experiments" / EXPDIR[n]
        f = d / "last_val_metrics.json" if (d / "last_val_metrics.json").exists() else d / "val_metrics.json"
        if f.exists():
            m = json.loads(f.read_text()); out.append(f"- {n}: dataset IoU {m['dataset_iou']:.4f}, precision {m['precision']:.4f}, recall {m['recall']:.4f} ({f.relative_to(C.REPO)})")
    txt = "\n".join(out); print(txt); (C.EXP / a.out).write_text(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
