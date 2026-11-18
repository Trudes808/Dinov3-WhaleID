"""Does a part model read "pale = fluke"? Two CPU tests (the coordinator's hypothesis, 2026-11-17 night).

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.part_gaps.pale_test recolour --models parts_seg_r05_a ... [--threads 8]
    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.part_gaps.pale_test summary
    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.part_gaps.pale_test real

recolour: rendered whales with a labelled fluke (render category F_*) or >= 800 px of labelled pectoral fin, from run05
(validation shots scored by render_eval), r06r_test_r9 and r06r_test_r10. In CIELAB, the part's L is shifted so its
median equals the median L of the same whale's body pixels (darken; texture kept, a / b unchanged); on r10, whose parts are
already dark, the reverse: L shifted up to the median L of the r9 + run05 fluke pixels (lighten). Control: the same L
shift on an equal number of body pixels of the same whale (the body pixels nearest to the body pixel farthest from the
part). The per-whale window of render_eval (1.5 x box, <= 768 px) is run for original / part edit / control edit, with
each model's own hr three-way rule (re-chosen on the training days) and pixel shares of the part's true pixels.
real: real instances of the labelled sample (AI labels) called fluke views or pectorals; whiteness of the instance pixels
(median L*, share with L* > 75) in terciles; each model's error / "called fluke" share per tercile.
Outputs: /var/tmp/sat3737-parts-cache/part_gaps/pale/*.jsonl, experiments/part_gaps/pale_test.json (numbers only).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from ..common.data import to_tensor
from ..parts_seg.model import PartsSegModel
from ..parts_seg.predict import run_model
from ..parts_seg import evaluate_real as ER
from . import common as C
from .render_eval import truth_maps
from .report import render_category, rules_for

OUT = C.CACHE / "pale"
CKPT = {"parts_seg_r05_w2": Path("/var/tmp/sat3737-parts-cache/parts_weak/runs/parts_seg_r05_w2/checkpoints/best.pt"),
        "parts_seg_r05_w2s1": Path("/var/tmp/sat3737-parts-cache/parts_weak/runs/parts_seg_r05_w2s1/checkpoints/best.pt")}
FLUKE, PEC, BODY = 1, 3, 5


def items(seed=0, n_run05=50):
    out = []
    for tag in ("run05", "r06r_test_r9", "r06r_test_r10"):
        rs = [json.loads(l) for l in open(C.CACHE / "renders" / tag / "parts_seg_r05_a.jsonl")]
        fl = [r for r in rs if render_category(r).startswith("F_")]
        pe = [r for r in rs if r["truth"][PEC] >= 800]
        if tag == "run05":
            fl = random.Random(seed).sample(fl, min(n_run05, len(fl)))
        for r in fl:
            out.append((tag, r, FLUKE))
        for r in pe:
            out.append((tag, r, PEC))
    return out


def lab_L(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)


def edit(rgb, sel, dL):
    lab = lab_L(rgb).astype(np.float32)
    lab[..., 0][sel] = np.clip(lab[..., 0][sel] + dL, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


def control_region(body, part, n):
    ys, xs = np.nonzero(body)
    if len(ys) == 0:
        return None
    py, px = np.nonzero(part)
    cy, cx = py.mean(), px.mean()
    k = int(np.argmax((ys - cy) ** 2 + (xs - cx) ** 2))
    d = (ys - ys[k]) ** 2 + (xs - xs[k]) ** 2
    take = np.argsort(d)[: min(n, len(ys))]
    m = np.zeros_like(body)
    m[ys[take], xs[take]] = True
    return m


def window(box, W):
    k2560 = 2560.0 / W
    x1, y1, x2, y2 = box
    side = max(max(x2 - x1, y2 - y1) * 1.5, 192.0 / k2560)
    osz = int(min(768, max(256, round(side * k2560 * 2 / 16) * 16)))
    hx, hy = (x1 + x2) / 2 - side / 2, (y1 + y2) / 2 - side / 2
    return np.array([[osz / side, 0, -hx * osz / side], [0, osz / side, -hy * osz / side]]), osz


def recolour(a):
    torch.set_num_threads(a.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    its = items()
    # lighten target: median L of r9 + run05 fluke pixels
    tgt_p = OUT / "lighten_target.json"
    if tgt_p.exists():
        target = json.loads(tgt_p.read_text())["L"]
    else:
        vals = []
        for tag, r, part in its:
            if part != FLUKE or tag == "r06r_test_r10":
                continue
            ds = C.FACTORY / tag / "dataset"
            rgb = cv2.cvtColor(cv2.imread(str(ds / "images" / f"{r['shot']}.png")), cv2.COLOR_BGR2RGB)
            parts, inst = truth_maps(ds, r["shot"], *rgb.shape[:2])
            m = (inst == r["iid"]) & (parts == FLUKE)
            if m.sum():
                vals.append(float(np.median(lab_L(rgb)[..., 0][m])))
        target = float(np.median(vals))
        tgt_p.write_text(json.dumps({"L": target, "n": len(vals)}))
    models = {n: PartsSegModel.from_checkpoint(CKPT.get(n, C.DAYS_ALL.parent / n / "checkpoints" / "best.pt"), torch.device("cpu"))[0]
              for n in a.models}
    out_p = OUT / f"recolour_{'_'.join(x.replace('parts_seg_r05_', '') for x in a.models)}.jsonl"
    done = {(d["tag"], d["shot"], d["iid"], d["part"]) for d in map(json.loads, open(out_p))} if out_p.exists() else set()
    with open(out_p, "a") as f:
        for n_, (tag, r, part) in enumerate(its):
            key = (tag, r["shot"], r["iid"], part)
            if key in done:
                continue
            ds = C.FACTORY / tag / "dataset"
            rgb = cv2.cvtColor(cv2.imread(str(ds / "images" / f"{r['shot']}.png")), cv2.COLOR_BGR2RGB)
            H, W = rgb.shape[:2]
            parts, inst = truth_maps(ds, r["shot"], H, W)
            whale = inst == r["iid"]
            P = whale & (parts == part)
            B = whale & (parts == BODY)
            L = lab_L(rgb)[..., 0].astype(np.float32)
            Lp = float(np.median(L[P]))
            Lb = float(np.median(L[B])) if B.any() else None
            if tag == "r06r_test_r10":
                mode, dL = "lighten", target - Lp
            else:
                mode, dL = "darken", ((Lb - Lp) if Lb is not None else None)
            rec = {"tag": tag, "shot": r["shot"], "iid": r["iid"], "part": "fluke" if part == FLUKE else "pec", "mode": mode,
                   "L_part": Lp / 2.55, "L_body": None if Lb is None else Lb / 2.55, "white_share": float(np.mean(L[P] / 2.55 > 75)),
                   "dL": None if dL is None else dL / 2.55, "part_px": int(P.sum()), "truth": r["truth"], "res": {}}
            versions = {"orig": rgb}
            if dL is not None and ((mode == "darken" and dL < -5 * 2.55) or (mode == "lighten" and dL > 5 * 2.55)):
                versions["part"] = edit(rgb, P, dL)
                Cm = control_region(B, P, int(P.sum()))
                if Cm is not None:
                    versions["control"] = edit(rgb, Cm, dL)
                    rec["control_px"] = int(Cm.sum())
            M, osz = window(r["box"], W)
            Mi = cv2.invertAffineTransform(M)
            near = cv2.dilate(whale.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            for vn, img in versions.items():
                crop = cv2.warpAffine(img, M, (osz, osz), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                rec["res"][vn] = {}
                for mn, model in models.items():
                    cm = run_model(model, to_tensor(crop), torch.device("cpu"))
                    up = cv2.warpAffine(cm, Mi, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
                    cnt = np.bincount(up[near & (up != 255)], minlength=256)[:6].tolist()
                    onpart = np.bincount(up[P & (up != 255)], minlength=256)[:6].tolist()
                    rec["res"][vn][mn] = {"near": cnt, "on_part": onpart}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if n_ % 10 == 0:
                print(n_, len(its), flush=True)
    print("wrote", out_p)


def call(cnt, rule):
    return ER.rule_three({"pixels_near": dict(zip(C.CLASSES, cnt)), "pixels": dict(zip(C.CLASSES, cnt)), "covered": max(sum(cnt), 1)},
                         rule[0], rule[1])


def summary(a):
    recs = []
    for p in sorted(OUT.glob("recolour_*.jsonl")):
        for d in map(json.loads, open(p)):
            recs.append(d)
    by = {}
    for d in recs:                                   # merge model results of the same whale from several files
        k = (d["tag"], d["shot"], d["iid"], d["part"])
        if k in by:
            for v, x in d["res"].items():
                by[k]["res"].setdefault(v, {}).update(x)
        else:
            by[k] = d
    models = sorted({m for d in by.values() for v in d["res"].values() for m in v})
    rules = {m: rules_for(m, C.arm_rows(m, C.TRAIN_DAYS + C.VAL_DAYS))["hr"] for m in models}
    res = {"models": models, "n_whales": len(by), "groups": {}}
    print(f"{'group':28s} {'model':10s} {'n':>3s}  fluke-called orig->part edit | orig->control | on-part px painted fluke orig / part / control")
    for part in ("fluke", "pec"):
        for mode in ("darken", "lighten"):
            ds = [d for d in by.values() if d["part"] == part and d["mode"] == mode and "part" in d["res"]]
            if not ds:
                continue
            g = res["groups"].setdefault(f"{part}|{mode}", {"n": len(ds), "n_with_control": sum("control" in d["res"] for d in ds),
                                                          "L_part_median": float(np.median([d["L_part"] for d in ds])),
                                                          "L_body_median": float(np.median([d["L_body"] for d in ds if d["L_body"] is not None])) if any(d["L_body"] is not None for d in ds) else None,
                                                          "dL_median": float(np.median([d["dL"] for d in ds])), "models": {}})
            for m in models:
                rows = [d for d in ds if m in d["res"]["orig"]]
                fl = lambda d, v: int(call(d["res"][v][m]["near"], rules[m]) == "fluke")  # noqa: E731
                pf = lambda d, v: d["res"][v][m]["on_part"][FLUKE] / max(sum(d["res"][v][m]["on_part"]), 1)  # noqa: E731
                orig = np.array([fl(d, "orig") for d in rows]); ed = np.array([fl(d, "part") for d in rows])
                cr = [d for d in rows if "control" in d["res"]]
                oc = np.array([fl(d, "orig") for d in cr]); cc = np.array([fl(d, "control") for d in cr])
                x = {"n": len(rows), "fluke_called_orig": float(orig.mean()), "fluke_called_part_edit": float(ed.mean()),
                     "flip_part": float(np.mean(orig != ed)), "n_control": len(cr),
                     "fluke_called_orig_ctrlset": float(oc.mean()) if len(cr) else None, "fluke_called_control_edit": float(cc.mean()) if len(cr) else None,
                     "flip_control": float(np.mean(oc != cc)) if len(cr) else None,
                     "part_px_fluke_orig": float(np.median([pf(d, "orig") for d in rows])), "part_px_fluke_part_edit": float(np.median([pf(d, "part") for d in rows])),
                     "part_px_fluke_control_edit": float(np.median([pf(d, "control") for d in cr])) if cr else None,
                     "part_px_fluke_mean_change_part": float(np.mean([pf(d, "part") - pf(d, "orig") for d in rows])),
                     "part_px_fluke_mean_change_control": float(np.mean([pf(d, "control") - pf(d, "orig") for d in cr])) if cr else None}
                g["models"][m] = x
                f2 = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
                print(f"{part + ' ' + mode:28s} {m.replace('parts_seg_r05_', ''):10s} {len(rows):3d}  {x['fluke_called_orig']:.2f}->{x['fluke_called_part_edit']:.2f} (flip {x['flip_part']:.2f}) | "
                      f"{f2(x['fluke_called_orig_ctrlset'])}->{f2(x['fluke_called_control_edit'])} (flip {f2(x['flip_control'])}, n {len(cr)}) | "
                      f"mean change of part px painted fluke: part edit {x['part_px_fluke_mean_change_part']:+.2f}, control {f2(x['part_px_fluke_mean_change_control'])}")
    (C.OUT / "pale_test.json").write_text(json.dumps({**json.loads((C.OUT / "pale_test.json").read_text()), "recolour": res}
                                                     if (C.OUT / "pale_test.json").exists() else {"recolour": res}, indent=1))


def real(a):
    from .report import load_labels
    from .sample import instance_mask_2560
    lab = load_labels()
    models = a.models
    rows = {m: {r["key"]: r for r in C.arm_rows(m, C.TRAIN_DAYS + C.VAL_DAYS)} for m in models}
    rules = {m: rules_for(m, list(rows[m].values()))["hr"] for m in models}
    feats = {}
    cache = OUT / "real_whiteness.json"
    if cache.exists():
        feats = json.loads(cache.read_text())
    for k, v in lab.items():
        if k in feats or not (v["cat"].startswith("F_") or v["cat"] == "P_pec"):
            continue
        stem = k.rsplit("~", 1)[0]
        C.check_day(v["day"])
        bgr = cv2.imread(str(C.DAYS_ALL / v["day"] / "images" / f"{stem}.jpg"))
        H, W = bgr.shape[:2]
        sc = W / 2560.0
        m = instance_mask_2560(v["day"], stem, int(k.rsplit("~", 1)[1]), 2560, int(round(H / sc)))
        if m is None:
            continue
        if sc != 1.0:
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        L = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[..., 0][m] / 2.55
        feats[k] = {"L_median": float(np.median(L)), "white_share": float(np.mean(L > 75))}
    OUT.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(feats))
    res = {}
    for group, sel in (("fluke views", lambda v: v["cat"].startswith("F_")), ("pectorals", lambda v: v["cat"] == "P_pec")):
        ks = [k for k, v in lab.items() if sel(v) and k in feats]
        Ls = np.array([feats[k]["L_median"] for k in ks])
        t1, t2 = np.percentile(Ls, [33.3, 66.7])
        terc = {k: ("dark" if feats[k]["L_median"] <= t1 else "mid" if feats[k]["L_median"] <= t2 else "pale") for k in ks}
        g = res[group] = {"n": len(ks), "tercile_L_bounds": [float(t1), float(t2)], "white_share_median_by_tercile": {}, "models": {}}
        for t in ("dark", "mid", "pale"):
            g["white_share_median_by_tercile"][t] = float(np.median([feats[k]["white_share"] for k in ks if terc[k] == t]))
        for m in models:
            g["models"][m] = {}
            for t in ("dark", "mid", "pale"):
                kk = [k for k in ks if terc[k] == t and k in rows[m] and "hr" in rows[m][k]["pred"]]
                calls = [ER.rule_three(rows[m][k]["pred"]["hr"], rules[m][0], rules[m][1]) for k in kk]
                w = np.array([lab[k]["weight"] for k in kk])
                fl = np.array([c == "fluke" for c in calls], float)
                truth = "fluke" if group == "fluke views" else "other"
                err = np.array([c != truth for c in calls], float)
                g["models"][m][t] = {"n": len(kk), "called_fluke": float((w * fl).sum() / w.sum()), "error": float((w * err).sum() / w.sum()),
                                     "k_fluke": int(fl.sum()), "k_err": int(err.sum())}
        print(f"== {group} (n {len(ks)}; tercile bounds L* {t1:.0f} / {t2:.0f}; median white share dark/mid/pale "
              f"{g['white_share_median_by_tercile']['dark']:.2f}/{g['white_share_median_by_tercile']['mid']:.2f}/{g['white_share_median_by_tercile']['pale']:.2f})")
        for m in models:
            print("   " + m.replace("parts_seg_r05_", "").ljust(6) + "  ".join(
                f"{t}: called fluke {x['called_fluke']:.2f} ({x['k_fluke']}/{x['n']}), error {x['error']:.2f}" for t, x in g["models"][m].items()))
    p = C.OUT / "pale_test.json"
    d = json.loads(p.read_text()) if p.exists() else {}
    d["real"] = res
    p.write_text(json.dumps(d, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["recolour", "summary", "real"])
    ap.add_argument("--models", nargs="+", default=["parts_seg_r05_a", "parts_seg_r05_s1", "parts_seg_r05_w2", "parts_seg_r05_w2s1"])
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    {"recolour": recolour, "summary": summary, "real": real}[a.what](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
