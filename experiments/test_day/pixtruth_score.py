"""Score segmentation models on the frozen pixel truth of the test day 2026-04-07 (rules: pixtruth_rules.md, section 4).

    R=/home/sat3737/NOAA_whales/moonshot_cache/test_days/2026-04-07
    .venv/bin/python experiments/test_day/pixtruth_score.py --root $R                      # every model we have -> $R/pixel_truth/scores/
    .venv/bin/python experiments/test_day/pixtruth_score.py --root $R --ours NAME=DIR      # a model of ours, added to the same tables

A model of ours: DIR/<seq>.png per sampled photo (3-digit sequence number), uint8 label map in the truth's classes
(0 water / sky, 1 whale, 2 person, 3 rail, 4 tape, 5 deck, 6 pontoon, 7 other vessel, 8 land, 9 gear; 255 = no answer),
any resolution with the photo's aspect (resized nearest to the truth grid). It is scored per fine class and at the
grouped levels (our boat = 3-6, any boat = 3-7). The script refuses to run unless the truth files match the manifest's
frozen hashes.
Outputs: scores.json (pooled, per pair), per_photo.tsv, confusion_<model>.tsv, tape.tsv.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

H, W = 1835, 2748
TRUTH = ["water_sky", "whale", "person", "rail", "tape", "deck", "pontoon", "other_vessel", "land", "gear"]
GROUPS = ["person", "boat", "vessel", "land", "water", "sky", "other"]
# truth side of each scored pair: name -> truth classes
T_SETS = {"whale": [1], "person": [2], "rail": [3], "tape": [4], "deck": [5], "pontoon": [6], "other_vessel": [7], "land": [8], "gear": [9],
          "water_sky": [0], "our_boat": [3, 4, 5, 6], "any_boat": [3, 4, 5, 6, 7], "rail_tape": [3, 4]}
# ready-made pairs (rules 4): pair -> (truth set, model groups)
RM_PAIRS = {"person": ("person", ["person"]), "our_boat_strict": ("our_boat", ["boat"]), "our_boat": ("our_boat", ["boat", "vessel"]),
            "other_vessel": ("other_vessel", ["vessel"]), "any_boat": ("any_boat", ["boat", "vessel"]), "land": ("land", ["land"]),
            "water_sky": ("water_sky", ["water", "sky"]), "rail_tape": ("rail_tape", ["boat"])}
OURS_PAIRS = {k: (k, T_SETS[k]) for k in T_SETS}
NEAR = 250   # rules said 40 px; at 40 px the rail + tape patch is nearly square and the tube axis is unstable (deviation, see tables.md)


def check_frozen(P: Path) -> dict:
    m = json.loads((P / "manifest.json").read_text())
    for f, h in m["frozen"]["sha256"].items():
        if hashlib.sha256((P / f).read_bytes()).hexdigest() != h:
            raise SystemExit(f"truth file changed since freeze: {f} (re-freeze with a new manifest entry first)")
    return m


def rs(a: np.ndarray) -> np.ndarray:
    return cv2.resize(a, (W, H), interpolation=cv2.INTER_NEAREST) if a.shape != (H, W) else a


def band_mask(t: np.ndarray, r: int = 2) -> np.ndarray:
    k = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
    return cv2.dilate(t, k) != cv2.erode(t, k)


def m2f_luts():
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from readymade_analyze import lut_for
    return {m: lut_for(m)[1] for m in ("ade", "coco")}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--root", type=Path, required=True); ap.add_argument("--ours", action="append", default=[])
    ap.add_argument("--out", type=Path); a = ap.parse_args()
    R = a.root; P = R / "pixel_truth"; m = check_frozen(P); out = a.out or P / "scores"; out.mkdir(parents=True, exist_ok=True)
    D = R / "days" / "2026-04-07"
    stems = {p.name.split("_")[-3]: p.name[: -len("_mask.png")] for p in (D / "canon").glob("*_mask.png")}
    luts = m2f_luts()
    ours = dict(s.split("=", 1) for s in a.ours)

    def load_models(seq: str) -> dict:
        st = stems[seq]; mm = {}
        for k in ("ade", "coco"):
            mm[k] = rs(luts[k][np.array(Image.open(R / "round2" / "readymade" / k / f"{seq}.png"))])
        mm["live"] = rs((np.array(Image.open(D / "canon" / f"{st}_mask.png")) > 0).astype(np.uint8))
        mm["s6h"] = rs((np.array(Image.open(R / "cand_s6" / f"{st}_s6.png")) > 0).astype(np.uint8))
        f = R / "round2" / "w3c_live" / "2026-04-07" / "w3c" / "full" / f"{st}.png"
        mm["w3c_anypart"] = rs(((np.array(Image.open(f)) >= 1) & (np.array(Image.open(f)) <= 5)).astype(np.uint8))
        for n, d in ours.items():
            mm[n] = rs(np.array(Image.open(Path(d) / f"{seq}.png")))
        return mm

    # pair definitions per model: name -> (truth set, predicate on the model map)
    pairs = {}
    for k in ("ade", "coco"):
        pairs[k] = {p: (T_SETS[ts], [GROUPS.index(g) for g in gs]) for p, (ts, gs) in RM_PAIRS.items()}
    for k in ("live", "s6h", "w3c_anypart"):
        pairs[k] = {"whale": ([1], [1])}
    for n in ours:
        pairs[n] = {p: (ts, ts) for p, (_, ts) in OURS_PAIRS.items()}
    ncols = {"ade": GROUPS, "coco": GROUPS, "live": ["not", "whale"], "s6h": ["not", "whale"], "w3c_anypart": ["not", "part"]}
    ncols.update({n: TRUTH + ["no_answer"] for n in ours})

    acc = {mdl: {p: defaultdict(int) for p in pp} for mdl, pp in pairs.items()}
    conf = {mdl: np.zeros((11, len(c)), np.int64) for mdl, c in ncols.items()}
    rows = []; tape_rows = []
    for seq in m["sample"]:
        t = np.array(Image.open(P / "labels" / f"{seq}.png")); valid = t != 255; band = band_mask(t) | ~valid
        tr = np.where(t == 255, 10, t)
        mm = load_models(seq)
        for mdl, pp in pairs.items():
            pred = mm[mdl]
            pc = pred if mdl in ("ade", "coco", "live", "s6h", "w3c_anypart") else np.where(pred == 255, 10, pred).clip(0, 10)
            conf[mdl] += np.bincount((tr.astype(np.int64) * len(ncols[mdl]) + pc.astype(np.int64)).ravel(), minlength=11 * len(ncols[mdl])).reshape(11, -1)
            for p, (ts, ps) in pp.items():
                tm = np.isin(t, ts) & valid; pm = np.isin(pred, ps) & valid
                if p == "person":
                    acc[mdl][p]["fp_on_vessel"] += int((pm & (t == 7)).sum())
                tp, fp, fn = int((tm & pm).sum()), int((~tm & pm).sum()), int((tm & ~pm).sum())
                nb = ~band; tpb, fpb, fnb = int((tm & pm & nb).sum()), int((~tm & pm & nb).sum()), int((tm & ~pm & nb).sum())
                A = acc[mdl][p]
                for key, v in (("tp", tp), ("fp", fp), ("fn", fn), ("tp_b", tpb), ("fp_b", fpb), ("fn_b", fnb)):
                    A[key] += v
                A["n_truth"] += int(tm.any()); A["n_pred"] += int(pm.any()); A["n_both"] += int(tm.any() and pm.any())
                iou = tp / (tp + fp + fn) if tp + fp + fn else ""
                rows.append((seq, m["truth_class"][seq], mdl, p, int(tm.sum()), int(pm.sum()), tp, fp, fn, round(iou, 4) if iou != "" else ""))
        if (t == 4).any():
            n, cc = cv2.connectedComponents((t == 4).astype(np.uint8))
            for k in range(1, n):
                comp = cc == k
                if comp.sum() < 50:
                    continue
                near = cv2.dilate(comp.astype(np.uint8), np.ones((2 * NEAR + 1, 2 * NEAR + 1), np.uint8)).astype(bool) & np.isin(t, [3, 4])
                ys, xs = np.nonzero(near); pts = np.stack([xs, ys], 1).astype(np.float64); pts -= pts.mean(0)
                axis = np.linalg.eigh(pts.T @ pts)[1][:, -1]; across = np.array([-axis[1], axis[0]])
                cy, cx = np.nonzero(comp); q = np.stack([cx, cy], 1).astype(np.float64)
                along_ext = np.ptp(q @ axis); across_ext = np.ptp(q @ across)
                tape_rows.append((seq, m["truth_class"][seq], int(comp.sum()), int(comp.sum()) * 4, round(comp.sum() / (H * W), 5),
                                  round(float(along_ext), 1), round(float(across_ext), 1), round(float(along_ext / max(across_ext, 1)), 3)))
    res = {}
    for mdl, pp in acc.items():
        res[mdl] = {}
        for p, A in pp.items():
            tp, fp, fn = A["tp"], A["fp"], A["fn"]; tpb, fpb, fnb = A["tp_b"], A["fp_b"], A["fn_b"]
            f = lambda x, y: round(x / y, 4) if y else None
            res[mdl][p] = {"iou": f(tp, tp + fp + fn), "recall": f(tp, tp + fn), "precision": f(tp, tp + fp), "iou_band": f(tpb, tpb + fpb + fnb),
                           "recall_band": f(tpb, tpb + fnb), "precision_band": f(tpb, tpb + fpb), "n_truth": A["n_truth"], "n_pred": A["n_pred"],
                           "truth_px": tp + fn, "pred_px": tp + fp}
            if "fp_on_vessel" in A:
                res[mdl][p]["precision_without_vessel_px"] = f(tp, tp + fp - A["fp_on_vessel"])
    (out / "scores.json").write_text(json.dumps(res, indent=1))
    with open(out / "per_photo.tsv", "w") as f:
        f.write("seq\tphoto_class\tmodel\tpair\ttruth_px\tpred_px\ttp\tfp\tfn\tiou\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")
    for mdl, c in conf.items():
        with open(out / f"confusion_{mdl}.tsv", "w") as f:
            f.write("truth\\model\t" + "\t".join(ncols[mdl]) + "\t" + "\t".join(f"{x}_rownorm" for x in ncols[mdl]) + "\n")
            for i, name in enumerate(TRUTH + ["unsure_255"]):
                s = c[i].sum()
                f.write(name + "\t" + "\t".join(str(int(v)) for v in c[i]) + "\t" + "\t".join(f"{v / s:.4f}" if s else "" for v in c[i]) + "\n")
    with open(out / "tape.tsv", "w") as f:
        f.write("seq\tphoto_class\ttape_px_half\ttape_px_full_equiv\tshare_of_frame\textent_along_tube\textent_across_tube\tband_height_in_tube_widths\n")
        for r in tape_rows:
            f.write("\t".join(map(str, r)) + "\n")
    print(json.dumps({k: {p: (v["iou"], v["recall"], v["precision"], v["n_truth"]) for p, v in d.items()} for k, d in res.items()}, indent=0))


if __name__ == "__main__":
    main()
