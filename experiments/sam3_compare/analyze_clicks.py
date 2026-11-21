"""SAM 3 instances on the researcher's clicks (2025-01-02, 2025-03-18; training days of ours), scored as
experiments/test_day/switch_eval.py click_score scores our arms: flukes with both tips in one instance, main whales with
all clicks in one instance, other whales own / fused / none; plus the part phrasings on the clicked fins and flukes.

    PYTHONPATH=. .venv/bin/python experiments/sam3_compare/analyze_clicks.py --settings SETTINGS.json --out OUT.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "instances_v2"))
import s3common as C  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--settings", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(); S = json.loads(a.settings.read_text())
    import common as IC
    from model_fine_tuning.cluster.instances_v2 import old_instances
    from model_fine_tuning.common import geometry, paths
    table = geometry.load_folded_table(paths.FOLDED_TABLE)
    idx = C.prompt_index("click")
    wj = [idx[p] for p in S["class_prompts"]["whale"]]; wthr = S["class_thr"]["whale"]
    part_thr = S.get("part_thr", 0.5)
    parts = {"dorsal fin": idx["dorsal fin"], "whale tail": idx["whale tail"], "whale fluke": idx["whale fluke"]}
    res = {}; fin_rows = []; fluke_rows = []
    for day in C.CLICK:
        photos, oth = IC.load_marks(day)
        others, flukes_main, flukes_other, main_one, no_inst = [], [], [], [], 0
        n_inst = []
        for key, kp in photos.items():
            stem = key.rsplit("~", 1)[0]
            rec = C.load("click", stem)
            if rec is None:
                continue
            H, W = (int(x) for x in rec["hw"])
            lk, _ = old_instances(C.LIVE / day, stem, W, H, table)
            if key not in lk:
                continue
            sc, ms = [], []
            for j in wj:
                s_, _, m_ = C.inst(rec, j, wthr, (H, W)); sc += list(s_); ms += m_
            masks = C.deoverlap(sc, ms) if ms else []
            n_inst.append(len(masks))
            if not masks:
                no_inst += 1

            def whale(pts):
                P = {n: (p["x"] * W, p["y"] * H) for n, p in pts.items()}
                nw = {n: IC.which(masks, p, 25.0) for n, p in P.items()}
                v = [x for x in nw.values() if x is not None]
                major = Counter(v).most_common(1)[0][0] if v else None
                fl = None
                if "fluke_tip_left" in nw and "fluke_tip_right" in nw:
                    fl = None if None in (nw["fluke_tip_left"], nw["fluke_tip_right"]) else nw["fluke_tip_left"] == nw["fluke_tip_right"]
                return major, nw, fl, "fluke_tip_left" in P and "fluke_tip_right" in P
            mj, nw, fl, has = whale(kp)
            main_one.append(None not in nw.values() and len(set(nw.values())) == 1)
            if has:
                flukes_main.append(fl)
            for w in (oth.get(key) or []):
                if not w:
                    continue
                m2, _, fl2, has2 = whale(w)
                others.append("none" if m2 is None else "fused" if m2 == mj else "own")
                if has2:
                    flukes_other.append(fl2)
            # parts on the main whale's clicks
            P = {n: (p["x"] * W, p["y"] * H) for n, p in kp.items()}
            if all(k in P for k in ("dorsal_fin_tip", "dorsal_fin_front", "dorsal_fin_back")):
                tri = np.zeros((H, W), np.uint8)
                cv2.fillPoly(tri, [np.array([P["dorsal_fin_tip"], P["dorsal_fin_front"], P["dorsal_fin_back"]], np.int32)], 1)
                tri = tri.astype(bool)
                if tri.sum() >= 50:
                    s_, _, m_ = C.inst(rec, parts["dorsal fin"], part_thr, (H, W))
                    u = np.any(m_, axis=0) if m_ else np.zeros((H, W), bool)
                    best = max(m_, key=lambda m: (m & tri).sum()) if m_ else np.zeros((H, W), bool)
                    whale_u = np.any(ms, axis=0) if ms else np.zeros((H, W), bool)
                    fin_rows.append({"day": day, "key": key, "tri_px": int(tri.sum()), "covered": round(float((u & tri).sum() / tri.sum()), 4),
                                     "best_px": int(best.sum()), "best_outside_tri": round(float((best & ~tri).sum() / max(best.sum(), 1)), 4),
                                     "best_outside_over_tri": round(float((best & ~tri).sum() / tri.sum()), 4),
                                     "tip_in_fin": bool(IC.which(m_, P["dorsal_fin_tip"], 25.0) is not None) if m_ else False,
                                     "n_fin_inst": len(m_), "whale_covers_tri": round(float((whale_u & tri).sum() / tri.sum()), 4)})
            if all(k in P for k in ("fluke_tip_left", "fluke_tip_right")):
                r = {"day": day, "key": key, "has_notch": "fluke_notch" in P}
                for nm, j in (("whale tail", parts["whale tail"]), ("whale fluke", parts["whale fluke"])):
                    s_, _, m_ = C.inst(rec, j, part_thr, (H, W))
                    pts = [P["fluke_tip_left"], P["fluke_tip_right"]] + ([P["fluke_notch"]] if "fluke_notch" in P else [])
                    hits = [IC.which(m_, p, 25.0) for p in pts] if m_ else [None] * len(pts)
                    r[nm] = {"n": len(m_), "all_points_one_instance": None not in hits and len(set(hits)) == 1,
                             "any_point": any(h is not None for h in hits)}
                fluke_rows.append(r)
        c = Counter(others)
        res[day] = {"other_whales": len(others), "own": c["own"], "fused": c["fused"], "none": c["none"], "flukes_main": len(flukes_main),
                    "flukes_main_one_instance": sum(x is True for x in flukes_main), "flukes_all": len(flukes_main) + len(flukes_other),
                    "flukes_all_one_instance": sum(x is True for x in flukes_main + flukes_other), "main_whales": len(main_one),
                    "main_all_clicks_one_instance": sum(main_one), "clicked_photos_without_instance": no_inst,
                    "instances_per_photo": round(float(np.mean(n_inst)), 3)}
    both = {k: sum(res[d][k] for d in C.CLICK) for k in res[C.CLICK[0]] if k != "instances_per_photo"}
    out = {"per_day": res, "both": both}
    if fin_rows:
        fr = fin_rows
        out["dorsal_fin"] = {"n": len(fr), "tri_covered_median": float(np.median([r["covered"] for r in fr])),
                             "tri_covered_ge_0.5": sum(r["covered"] >= 0.5 for r in fr), "tri_covered_ge_0.8": sum(r["covered"] >= 0.8 for r in fr),
                             "no_fin_instance": sum(r["n_fin_inst"] == 0 for r in fr), "tip_in_a_fin_instance": sum(r["tip_in_fin"] for r in fr),
                             "best_outside_share_median": float(np.median([r["best_outside_tri"] for r in fr if r["best_px"]])) if any(r["best_px"] for r in fr) else None,
                             "best_outside_over_tri_median": float(np.median([r["best_outside_over_tri"] for r in fr if r["best_px"]])) if any(r["best_px"] for r in fr) else None,
                             "whale_mask_covers_tri_median": float(np.median([r["whale_covers_tri"] for r in fr]))}
    if fluke_rows:
        out["fluke"] = {nm: {"n": len(fluke_rows), "all_points_one_instance": sum(r[nm]["all_points_one_instance"] for r in fluke_rows),
                             "no_instance": sum(r[nm]["n"] == 0 for r in fluke_rows), "any_point": sum(r[nm]["any_point"] for r in fluke_rows)}
                        for nm in ("whale tail", "whale fluke")}
    a.out.write_text(json.dumps(out, indent=1)); (a.out.parent / "click_fin_rows.json").write_text(json.dumps(fin_rows)); (a.out.parent / "click_fluke_rows.json").write_text(json.dumps(fluke_rows))
    print(json.dumps({k: v for k, v in out.items() if k != "per_day"}, indent=1))


if __name__ == "__main__":
    main()
