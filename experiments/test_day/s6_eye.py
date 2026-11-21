"""Round 2: by-eye labels for the instances of the candidate-mask arms (rules: round2_rules.md, section 1).

    .venv/bin/python experiments/test_day/s6_eye.py prep --root ROOT --sheets SCRATCH/eye
        S6-src instances (round2/days_s6/<day>/parts_hr) and S6-v2 instances (round2/days_s6/<day>/instances_v2) ->
        ROOT/round2/s6_inst.json (per instance: arm, key, seq, box, inherited label or None). An instance whose box has
        IoU >= 0.7 with a round-1 instance labelled W inherits W; the rest go on tile sheets SCRATCH/eye/eye_<arm>_NN.jpg
        (full photo, box yellow, mask outline cyan, tile number) for a by-eye pass; the filter score is not shown.
The labels are written by hand to ROOT/round2/s6_eye.txt as "<arm> <tile number> <letter>" (letters as inst_eye.txt).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

EYE_FINAL = None


def read_round1(R: Path):
    keys = json.loads((R / "inst_keys.json").read_text()); lab = {}
    for l in open(R / "inst_eye.txt"):
        if l.startswith("#") or not l.strip():
            continue
        r, c = l.split(); a, _, b = r.partition("-")
        for i in range(int(a), int(b or a) + 1):
            lab[i] = c                                    # later lines (M resolved) override
    box = {it["key"]: it["bbox_2560"] for l in open(R / "days" / R.name / "parts_hr" / "index.jsonl") for it in json.loads(l)["instances"]}
    return {k: (lab[i], box[k]) for i, k in enumerate(keys)}


def iou(a, b) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0); u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def tiles(R: Path, items: list, out: Path, tag: str, per: int = 24, w: int = 480) -> None:
    D = R / "round2" / "days_s6" / R.name
    for s in range(0, len(items), per):
        tl = []
        for it in items[s:s + per]:
            im = cv2.imread(str(D / "images" / f"{it['stem']}.jpg")); H, W = im.shape[:2]; f = w / W
            sm = cv2.resize(im, (w, int(H * f)), interpolation=cv2.INTER_AREA)
            if it.get("mask") is not None:
                m = cv2.resize(it["mask"].astype(np.uint8), (w, int(H * f)), interpolation=cv2.INTER_NEAREST)
                cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(sm, cs, -1, (255, 255, 0), 1)
            x0, y0, x1, y1 = (int(v * f) for v in it["box"]); cv2.rectangle(sm, (x0, y0), (x1, y1), (0, 255, 255), 2)
            cv2.rectangle(sm, (0, 0), (150, 22), (0, 0, 0), -1); cv2.putText(sm, f"#{it['tile']} s{it['seq']}", (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            tl.append(sm)
        h = max(t.shape[0] for t in tl); tl = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tl]
        while len(tl) % 4:
            tl.append(np.zeros_like(tl[0]))
        rows = [np.hstack(tl[i:i + 4]) for i in range(0, len(tl), 4)]
        cv2.imwrite(str(out / f"eye_{tag}_{s // per:02d}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def main() -> None:
    from model_fine_tuning.cluster.instances_v2 import old_instances
    from model_fine_tuning.common import geometry, paths
    p = argparse.ArgumentParser(); p.add_argument("cmd"); p.add_argument("--root", type=Path, required=True); p.add_argument("--sheets", type=Path, required=True); a = p.parse_args()
    R = a.root; D = R / "round2" / "days_s6" / R.name; a.sheets.mkdir(parents=True, exist_ok=True)
    r1 = read_round1(R); r1_by_stem = {}
    for k, v in r1.items():
        r1_by_stem.setdefault(k.split("~")[0], []).append((k, *v))
    table = geometry.load_folded_table(paths.FOLDED_TABLE); seq = lambda stem: stem.split("_")[-2]
    out, todo = [], {"src": [], "v2": []}
    for l in open(D / "parts_hr" / "index.jsonl"):
        r = json.loads(l); stem = r["stem"]
        if not r["instances"]:
            continue
        H, W = cv2.imread(str(D / "images" / f"{stem}.jpg"), cv2.IMREAD_REDUCED_GRAYSCALE_2).shape; H, W = H * 2, W * 2
        okeys, omasks = old_instances(D, stem, W, H, table); mk = dict(zip(okeys, omasks))
        for it in r["instances"]:
            best = max([(iou(it["bbox_2560"], b), k, lab) for k, lab, b in r1_by_stem.get(stem, [])], default=(0, None, None))
            rec = {"arm": "src", "key": it["key"], "seq": seq(stem), "stem": stem, "box": it["bbox_2560"], "r1_key": best[1], "r1_iou": round(best[0], 3), "r1_label": best[2],
                   "label": "W" if best[0] >= 0.7 and best[2] == "W" else None}
            out.append(rec)
            if rec["label"] is None:
                todo["src"].append({**rec, "mask": mk.get(it["key"])})
    for l in open(D / "instances_v2" / "index.jsonl"):
        r = json.loads(l); stem = r["stem"]
        if not r["instances"] and not r["rejected"]:
            continue
        m = cv2.imread(str(D / "instances_v2" / "masks" / f"{stem}.png"), cv2.IMREAD_UNCHANGED); bl = cv2.imread(str(D / "instances_v2" / "blobs" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        for it in r["instances"] + r["rejected"]:
            best = max([(iou(it["bbox"], b), k, lab) for k, lab, b in r1_by_stem.get(stem, [])], default=(0, None, None))
            key = it.get("key") or f"{stem}~rej{'+'.join(map(str, it['blobs']))}"
            rec = {"arm": "v2", "key": key, "seq": seq(stem), "stem": stem, "box": it["bbox"], "kept": "key" in it, "why": it.get("why"), "r1_key": best[1], "r1_iou": round(best[0], 3), "r1_label": best[2],
                   "label": "W" if best[0] >= 0.7 and best[2] == "W" else None}
            out.append(rec)
            if rec["label"] is None:
                todo["v2"].append({**rec, "mask": np.isin(bl, it["blobs"])})
    for arm, items in todo.items():
        for j, it in enumerate(items):
            it["tile"] = j
        tiles(R, items, a.sheets, arm)
        for it in items:
            next(o for o in out if o["arm"] == arm and o["key"] == it["key"])["tile"] = it["tile"]
    (R / "round2" / "s6_inst.json").write_text(json.dumps(out, indent=0))
    print({arm: (sum(o["arm"] == arm for o in out), len(v)) for arm, v in todo.items()}, "(instances, to label by eye)")


if __name__ == "__main__":
    main()
