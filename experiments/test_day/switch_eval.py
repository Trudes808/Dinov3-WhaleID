"""Switch prep: the live chain (production day folders) against S6-src (switch_val.sh shadow root) on the val days, with the
catalogue; rules: moonshot_cache/test_days/switch_prep/rules_switch.md (written before the runs). Read only on every day
folder; eval days refused.

    PYTHONPATH=. .venv/bin/python experiments/test_day/switch_eval.py days  --s6 S6ROOT --out OUT.json [--arm-name S6] [--days ...]
    PYTHONPATH=. .venv/bin/python experiments/test_day/switch_eval.py clicks --s6 S6ROOT --out OUT.json
days: per day and pooled - catalogue photos without a kept instance, instances per photo (before / after the filter),
strict-tier clusters with the catalogue (wrong links, impure photos, clusters per animal, audit), view classes, and the
per-photo rows (mask px, CSE use px, instances) of every photo whose kept-instance state differs between the arms.
clicks: flukes with both tips in one instance, main whales with all clicks in one instance, other whales own / fused /
none, on the click days (2025-01-02, 2025-03-18) with the instance masks of each arm's parts_hr (kept after the filter
where the arm has safe/not_whale.json).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
LIVE = REPO / "experiments" / "days_all"
LABELS = REPO / "model_fine_tuning" / "splits" / "labels_all_days.json"
MODEL = REPO / "experiments" / "safe_clusters" / "safe_model.pkl"
VAL = ["2025-01-15", "2025-02-27", "2025-03-14"]
CLICK = ["2025-01-02", "2025-03-18"]
EVAL = {"2024-12-18", "2024-12-26", "2025-02-19", "2025-03-21"}


def kept_instances(d: Path) -> tuple[dict, dict, set]:
    """stem -> all instance keys (parts_hr), stem -> kept keys, the drop set."""
    nwf = d / "safe" / "not_whale.json"
    drop = set(json.loads(nwf.read_text()).get("drop", [])) if nwf.exists() else set()
    allk, kept = {}, {}
    for l in open(d / "parts_hr" / "index.jsonl"):
        r = json.loads(l); ks = [i["key"] for i in r.get("instances", [])]
        allk[r["stem"]] = ks; kept[r["stem"]] = [k for k in ks if k not in drop]
    return allk, kept, drop


def mask_px(d: Path, stem: str) -> int:
    m = cv2.imread(str(d / "canon" / f"{stem}_mask.png"), cv2.IMREAD_GRAYSCALE)
    return int((m > 0).sum()) if m is not None else 0


def use_px(d: Path, stem: str) -> int:
    f = d / "cse" / f"{stem}_canon.npz"
    return int(np.load(f)["use"].sum()) if f.exists() else 0


def clusters(root: Path, day: str) -> dict:
    from model_fine_tuning.cluster.safe_clusters import load_model
    from model_fine_tuning.cluster.safe_tracklets import audit, day_data, evaluate
    dd = day_data(root, day, LABELS, 1.5, 1.5, 300.0, 3, use_filter=True)
    rep, cl = evaluate(dd, load_model(MODEL), 0.995, 300.0, strict=True)
    au = audit(dd, cl)
    return {"metrics": rep["metrics"], "stage1": rep["stage1"], "wrong_links": rep["wrong_links"], "links": rep["links"], "tracklets": rep["tracklets"],
            "pair_auc": rep.get("pair_auc"), "audit": {k: au[k] for k in ("multi_whale_bursts", "multi_whale_bursts_split_pure", "multi_whale_photos", "impure_clusters")},
            "impure": au["impure"], "n_clusters": len(cl), "n_multi": sum(1 for c in cl if len(c["photos"]) > 1)}


def view_counts(d: Path) -> dict:
    f = d / "safe" / "view_class.json"
    return json.loads(f.read_text()).get("counts") if f.exists() else None


def cmd_days(a) -> None:
    cat = json.loads(LABELS.read_text())["days"]; roots = {"L": LIVE, a.arm_name: Path(a.s6)}
    out = {"days": {}, "pooled": {}}
    for day in a.days.split(","):
        assert day not in EVAL
        labels = cat[day]["photos"]; R = {}
        stems = [Path(json.loads(l)["image"]).stem for l in open(LIVE / day / "images" / "index.jsonl")]
        per = {}
        for arm, root in roots.items():
            d = root / day; allk, kept, drop = kept_instances(d)
            no_kept = [s for s in stems if labels.get(s.split("__")[-1]) and not kept.get(s)]
            R[arm] = {"photos": len(stems), "catalogue_photos": sum(1 for s in stems if labels.get(s.split("__")[-1])),
                      "no_kept": len(no_kept), "no_instance_before_filter": sum(1 for s in stems if not allk.get(s)),
                      "instances": sum(len(v) for v in allk.values()), "kept": sum(len(v) for v in kept.values()), "dropped": len(drop),
                      "photos_2plus_kept": sum(1 for v in kept.values() if len(v) >= 2), "view": view_counts(d), "clusters": clusters(root, day)}
            R[arm]["inst_per_photo"] = round(R[arm]["instances"] / len(stems), 3); R[arm]["kept_per_photo"] = round(R[arm]["kept"] / len(stems), 3)
            per[arm] = (allk, kept, drop, d)
        # photos whose state differs, and every photo without a kept instance in either arm
        rows = []
        for s in stems:
            st = {arm: len(per[arm][1].get(s, [])) for arm in roots}
            if all(st.values()) and len(set(st.values())) == 1:
                continue
            r = {"stem": s, "cat": labels.get(s.split("__")[-1])}
            for arm in roots:
                allk, kept, drop, d = per[arm]
                r[arm] = {"n": len(allk.get(s, [])), "kept": len(kept.get(s, [])), "mask_px": mask_px(d, s), "use_px": use_px(d, s)}
            rows.append(r)
        R["rows"] = rows
        out["days"][day] = R
    arms = list(roots)
    for arm in arms:
        P = [out["days"][d][arm] for d in out["days"]]
        s = {k: sum(p[k] for p in P) for k in ("photos", "catalogue_photos", "no_kept", "no_instance_before_filter", "instances", "kept", "dropped", "photos_2plus_kept")}
        s["inst_per_photo"] = round(s["instances"] / s["photos"], 3); s["kept_per_photo"] = round(s["kept"] / s["photos"], 3)
        M = [p["clusters"]["metrics"] for p in P]
        s["wrong_links"] = sum(len(p["clusters"]["wrong_links"]) for p in P); s["impure_photos"] = sum(m["photos_in_impure_clusters"] for m in M)
        s["clustered_photos"] = sum(m["photos"] for m in M); s["clusters"] = sum(m["clusters"] for m in M); s["animals"] = sum(m["animals"] for m in M)
        s["clusters_per_animal_mean_of_days"] = round(float(np.mean([m["clusters_per_animal"] for m in M])), 2)
        s["clusters_per_animal_pooled"] = round(s["clusters"] / s["animals"], 2)
        s["multi_whale_bursts"] = sum(p["clusters"]["audit"]["multi_whale_bursts"] for p in P); s["split_pure"] = sum(p["clusters"]["audit"]["multi_whale_bursts_split_pure"] for p in P)
        out["pooled"][arm] = s
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    for arm in arms:
        print(arm, out["pooled"][arm])


def click_score(root: Path, day: str, drop_known: bool) -> dict:
    import sys
    sys.path.insert(0, str(REPO / "experiments" / "instances_v2"))
    import common as IC
    from model_fine_tuning.cluster.instances_v2 import old_instances
    from model_fine_tuning.common import geometry, paths
    table = geometry.load_folded_table(paths.FOLDED_TABLE)
    d = root / day; _, _, drop = kept_instances(d)
    photos, oth = IC.load_marks(day)
    others, flukes_main, flukes_other, main_one, no_inst = [], [], [], [], 0
    for key, kp in photos.items():
        stem = key.rsplit("~", 1)[0]
        m = cv2.imread(str(LIVE / day / "canon" / f"{stem}_mask.png"), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        H, W = m.shape
        lk, _ = old_instances(LIVE / day, stem, W, H, table)          # the click record must name a live instance key (as e2e.py)
        if key not in lk:
            continue
        keys, masks = old_instances(d, stem, W, H, table)
        if drop_known:
            masks = [mk for k, mk in zip(keys, masks) if k not in drop]
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
    c = Counter(others)
    return {"other_whales": len(others), "own": c["own"], "fused": c["fused"], "none": c["none"], "flukes_main": len(flukes_main),
            "flukes_main_one_instance": sum(x is True for x in flukes_main), "flukes_all": len(flukes_main) + len(flukes_other),
            "flukes_all_one_instance": sum(x is True for x in flukes_main + flukes_other), "main_whales": len(main_one),
            "main_all_clicks_one_instance": sum(main_one), "clicked_photos_without_instance": no_inst}


def cmd_clicks(a) -> None:
    out = {}
    for arm, root in (("L", LIVE), (a.arm_name, Path(a.s6))):
        for filt in (False, True):
            name = f"{arm}{'_filtered' if filt else ''}"
            per = {day: click_score(root, day, filt) for day in CLICK}
            out[name] = {"per_day": per, "both": {k: sum(per[d][k] for d in CLICK) for k in per[CLICK[0]]}}
            print(name, out[name]["both"])
    Path(a.out).write_text(json.dumps(out, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("days"); s.add_argument("--s6", required=True); s.add_argument("--out", required=True); s.add_argument("--days", default=",".join(VAL)); s.add_argument("--arm-name", default="S6")
    s = sub.add_parser("clicks"); s.add_argument("--s6", required=True); s.add_argument("--out", required=True); s.add_argument("--arm-name", default="S6")
    a = p.parse_args()
    {"days": cmd_days, "clicks": cmd_clicks}[a.cmd](a)


if __name__ == "__main__":
    main()
