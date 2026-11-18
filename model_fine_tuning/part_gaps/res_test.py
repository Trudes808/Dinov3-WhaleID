"""Inference resolution test on the labelled real instances: the per-instance window of predict.hr_window (1.5 x the
box) run at several output caps (the live cap is 768 px), the instance's part shares recomputed and called with the
model's hr rule. Shows whether the part call is limited by the pixels the model gets.

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.part_gaps.res_test [--model parts_seg_r05_a] [--caps 256 384 768 1024]

Writes experiments/part_gaps/res_test_<model>.json (per instance and cap: shares and call; summary by size bin).
"""

from __future__ import annotations

import argparse
import json

import cv2
import numpy as np
import torch

from ..common import paths
from ..common.data import to_tensor
from ..parts_seg.model import PartsSegModel
from ..parts_seg.predict import instances_of, run_model
from ..parts_seg import evaluate_real as ER
from . import common as C
from .report import load_labels, rules_for
from .sample import table


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="parts_seg_r05_a")
    ap.add_argument("--caps", type=int, nargs="+", default=[256, 384, 768, 1024])
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    lab = load_labels()
    rows = C.arm_rows(a.model, C.TRAIN_DAYS + C.VAL_DAYS)
    rule = rules_for(a.model, rows)["hr"]
    model, _ = PartsSegModel.from_checkpoint(paths.EXPERIMENTS_DIR / a.model / "checkpoints" / "best.pt", torch.device("cpu"))
    out_p = C.OUT / f"res_test_{a.model}.json"
    res = json.loads(out_p.read_text()) if out_p.exists() else {"model": a.model, "rule": [list(rule[0]), rule[1]], "instances": {}}
    for n, (k, v) in enumerate(sorted(lab.items())):
        if v["three"] is None or (k in res["instances"] and all(str(c) in res["instances"][k]["caps"] for c in a.caps)):
            continue
        stem, kk = k.rsplit("~", 1)
        C.check_day(v["day"])
        rgb = cv2.cvtColor(cv2.imread(str(C.DAYS_ALL / v["day"] / "images" / f"{stem}.jpg")), cv2.COLOR_BGR2RGB)
        masks, (x0, y0, s) = instances_of(v["day"], stem, table())
        if int(kk) >= len(masks):
            continue
        x1, y1, x2, y2 = v["bbox"]
        side = max(max(x2 - x1, y2 - y1) * 1.5, 192.0)
        hx, hy = (x1 + x2) / 2 - side / 2, (y1 + y2) / 2 - side / 2
        rec = {"cat": v["cat"], "three": v["three"], "L": v["L"], "split": v["split"], "weight": v["weight"], "caps": {}}
        for cap in a.caps:
            osz = int(min(cap, max(256, round(side * 2 / 16) * 16)))
            M = np.array([[osz / side, 0, -hx * osz / side], [0, osz / side, -hy * osz / side]])
            crop = cv2.warpAffine(rgb, M, (osz, osz), flags=cv2.INTER_AREA if side > osz else cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            cm = run_model(model, to_tensor(crop), torch.device("cpu"))
            st = __import__("model_fine_tuning.parts_seg.predict", fromlist=["region_stats"]).region_stats(cm, masks[int(kk)], x0, y0, s, osz / side, hx, hy)
            call = ER.rule_three(st, rule[0], rule[1])
            rec["caps"][str(cap)] = {"osz": osz, "call": call, "ok": int(call == v["three"]),
                                     "shares": {c: round(x, 4) for c, x in C.pred_shares(st).items()}}
        res["instances"][k] = rec
        if n % 25 == 0:
            out_p.write_text(json.dumps(res))
            print(n, flush=True)
    # summary: weighted error by size bin and cap
    summ = {}
    for sp in ("train", "val", "all"):
        for b in [x[2] for x in C.SIZE_BINS]:
            rs = [r for r in res["instances"].values() if (sp == "all" or r["split"] == sp) and C.size_bin(r["L"]) == b]
            summ[f"{sp}|{b}"] = {str(c): {"n": len(rs), "err": float(sum(r["weight"] * (1 - r["caps"][str(c)]["ok"]) for r in rs) / max(sum(r["weight"] for r in rs), 1e-9)) if rs else None}
                                 for c in a.caps}
    res["summary"] = summ
    out_p.write_text(json.dumps(res, indent=1))
    for k_, d in summ.items():
        if k_.startswith("all|"):
            print(k_, {c: (x["n"], None if x["err"] is None else round(x["err"], 3)) for c, x in d.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
