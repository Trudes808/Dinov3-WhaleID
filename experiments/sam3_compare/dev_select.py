"""Dev measures for choosing SAM 3's phrasing and threshold (rules_sam3.md part A), on val days and training-day
non-whale instances only. Never reads the test day or the eval days.

    PYTHONPATH=. .venv/bin/python experiments/sam3_compare/dev_select.py --out /home/sat3737/NOAA_whales/moonshot_cache/sam3_compare/dev.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import s3common as C  # noqa: E402

G = (427, 640)
THRS = [0.3, 0.4, 0.5]


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); a = ap.parse_args()
    from model_fine_tuning.cluster.instances_v2 import old_instances
    from model_fine_tuning.common import geometry, paths
    table = geometry.load_folded_table(paths.FOLDED_TABLE)
    cat = C.catalogue()
    P = C.PROMPTS["prompts"]; groups = C.PROMPTS["groups"]
    gear_union = [P.index(p) for p in groups["gear"]]
    names = P + ["gear_union"]
    idx = C.prompt_index("val")
    assert [idx[p] for p in P] == list(range(len(P)))

    def masks_of(rec, thr):
        out = {n: C.union(rec, i, thr, G) for i, n in enumerate(P)}
        out["gear_union"] = np.any([out[P[i]] for i in gear_union], axis=0)
        return out

    acc = {t: defaultdict(lambda: defaultdict(float)) for t in THRS}
    n_cat = 0; per_photo = []
    for day in C.VAL:
        labels = cat[day]["photos"]
        for stem in C.day_stems(C.LIVE / day):
            rec = C.load("val", stem)
            if rec is None:
                continue
            lv = C.live_mask(day, stem, G); s6 = C.s6_mask_val(day, stem, G)
            sure = lv & s6; ours = lv | s6; is_cat = bool(labels.get(stem.split("__")[-1]))
            n_cat += is_cat
            row = {"day": day, "stem": stem, "cat": is_cat, "sure_px": int(sure.sum())}
            for t in THRS:
                mm = masks_of(rec, t)
                for n, u in mm.items():
                    A = acc[t][n]
                    j = P.index(n) if n in P else None
                    ninst = int((rec[f"s{j}"] >= t).sum()) if j is not None else int(u.any())
                    if is_cat:
                        A["photos_with_inst"] += ninst > 0
                    A["sure_px"] += sure.sum(); A["sure_cov"] += (u & sure).sum()
                    A["pred_px"] += u.sum(); A["pred_outside_ours"] += (u & ~ours).sum()
                    if sure.sum() >= 500:
                        A["photos_sure500"] += 1; A["photos_cover25_sure"] += (u & sure).sum() >= 0.25 * sure.sum()
                    A["photos_ge1pct"] += u.mean() >= 0.01
                    if n == "whale" and t == 0.5:
                        row["whale_n"] = ninst; row["whale_px"] = int(u.sum()); row["whale_on_sure"] = int((u & sure).sum())
                        row["whale_outside_ours"] = int((u & ~ours).sum())
            per_photo.append(row)
    # non-whale instances
    nw = json.loads((C.REPO / "experiments" / "stage2_linker" / "nw_labels.json").read_text())
    sub = json.loads((C.TD / "nw_subclass_train.json").read_text())["labels"]
    inst_rows = []
    for day, v in nw.items():
        assert day not in C.EVAL
        for key in v["pos"]:
            stem, k = key.rsplit("~", 1)
            rec = C.load("val" if day in C.VAL else "nw", stem)
            if rec is None:
                continue
            H0, W0 = rec["hw"]
            keys, ms = old_instances(C.LIVE / day, stem, int(W0), int(H0), table)
            if key not in keys:
                continue
            import cv2
            m = cv2.resize(ms[keys.index(key)].astype(np.uint8), (G[1], G[0]), interpolation=cv2.INTER_NEAREST) > 0
            if m.sum() < 5:
                continue
            r = {"day": day, "key": key, "sub": sub.get(key, "val_unlabelled" if day in C.VAL else "unlabelled"), "px": int(m.sum())}
            for t in THRS:
                mm = masks_of(rec, t)
                r[str(t)] = {n: round(float((u & m).sum() / m.sum()), 4) for n, u in mm.items()}
            inst_rows.append(r)
    res = {"n_catalogue": n_cat, "n_val_photos": len(per_photo), "n_nw_instances": len(inst_rows), "by_thr": {}}
    for t in THRS:
        res["by_thr"][str(t)] = {}
        for n in names:
            A = acc[t][n]
            d = {"R_photos_with_inst": round(A["photos_with_inst"] / n_cat, 4), "photos_with_inst": int(A["photos_with_inst"]),
                 "sure_recall": round(A["sure_cov"] / A["sure_px"], 4), "outside_ours_share": round(A["pred_outside_ours"] / max(A["pred_px"], 1), 4),
                 "pred_px": int(A["pred_px"]), "Fc_cover25_sure": round(A["photos_cover25_sure"] / A["photos_sure500"], 4),
                 "photos_cover25_sure": int(A["photos_cover25_sure"]), "photos_sure500": int(A["photos_sure500"]),
                 "photos_ge1pct": int(A["photos_ge1pct"])}
            allnw = [r[str(t)][n] for r in inst_rows]
            d["F_nw_painted25"] = round(float(np.mean([x >= 0.25 for x in allnw])), 4)
            for s_ in ("person", "gear", "boat", "other", "val_unlabelled"):
                xs = [r[str(t)][n] for r in inst_rows if r["sub"] == s_]
                d[f"Rc_{s_}"] = round(float(np.mean([x >= 0.25 for x in xs])), 4) if xs else None
                d[f"n_{s_}"] = len(xs)
            res["by_thr"][str(t)][n] = d
    a.out.write_text(json.dumps(res, indent=1))
    (a.out.parent / "dev_per_photo.json").write_text(json.dumps(per_photo))
    (a.out.parent / "dev_nw_instances.json").write_text(json.dumps(inst_rows))
    print("catalogue", n_cat, "nw instances", len(inst_rows), {s_: sum(r["sub"] == s_ for r in inst_rows) for s_ in ("person", "gear", "boat", "other", "val_unlabelled", "unlabelled")})


if __name__ == "__main__":
    main()
