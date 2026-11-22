"""C1: our scene model's whale against S6h on the val days' catalogue photos (both at 1280, S6h = S6 P(whale) with
hysteresis 0.6 / 0.1, computed by s6_predict.py), plus where S6h's pixels go in our map, and the extra classes' shares
against the labeller's maps on the same photos (pseudo_val; a check of fit to the labeller, not a truth).

    .venv/bin/python experiments/sam3_labeller/score_val.py --names seg_scene_sam3_s0,seg_scene_sam3_s1 --out OUT/score_val.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import lab_common as L  # noqa: E402

CAT = json.loads((L.REPO / "model_fine_tuning" / "splits" / "labels_all_days.json").read_text())["days"]


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--names", required=True); ap.add_argument("--out", type=Path, required=True); a = ap.parse_args()
    res = {}
    for name in a.names.split(","):
        n = 0; s6_empty = []; our_empty = []; I = U = 0; ious = []; where = np.zeros(7); conf = np.zeros((7, 7))
        for day in L.VAL:
            for stem in L.day_stems(day):
                if not CAT[day]["photos"].get(stem.split("__")[-1]):
                    continue
                n += 1
                s6 = L.hyst(L.s6_prob(day, stem)); ours = cv2.imread(str(L.OUT / "pred" / name / day / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
                w = ours == 1
                if not s6.any(): s6_empty.append(stem)
                if not w.any(): our_empty.append(stem)
                i, u = int((s6 & w).sum()), int((s6 | w).sum()); I += i; U += u
                if u: ious.append(i / u)
                where += np.bincount(ours[s6], minlength=7)[:7]
                pv = cv2.imread(str(L.OUT / "pseudo_val" / day / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
                if pv is not None:
                    ok = pv != 255
                    conf += np.bincount(pv[ok].astype(np.int64) * 7 + ours[ok], minlength=49).reshape(7, 7)
        ious = np.array(ious)
        lab_iou = {c: float(conf[k, k] / max(1, conf[k].sum() + conf[:, k].sum() - conf[k, k])) for k, c in enumerate(L.CLS)}
        res[name] = {"catalogue_photos": n, "s6h_empty": len(s6_empty), "ours_empty": len(our_empty),
                     "ours_empty_not_s6h_empty": sorted(set(our_empty) - set(s6_empty)), "s6h_empty_not_ours": sorted(set(s6_empty) - set(our_empty)),
                     "pooled_iou_vs_s6h": I / max(1, U), "per_photo_iou_median": float(np.median(ious)), "share_photos_iou_lt_0.8": float((ious < 0.8).mean()),
                     "n_photos_iou": int(len(ious)), "s6h_pixels_to_our_class": {c: float(where[k] / where.sum()) for k, c in enumerate(L.CLS)},
                     "iou_vs_labeller_maps_val": lab_iou}
        print(name, json.dumps({k: v for k, v in res[name].items() if not isinstance(v, list)}, indent=1))
    a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
