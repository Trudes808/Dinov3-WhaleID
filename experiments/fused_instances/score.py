"""Step 3: score the arms of rules_fused.md against the plain shadow chain P, per day and pooled.

    PYTHONPATH=. .venv/bin/python experiments/fused_instances/score.py --arms P,A_G,... --days D,.. --out OUT.json

Arms: P = shadow root; A_G / A_S = split arms in FI/days/<arm>; B_* = P's instances with a guard; C_SG = A_S + guard G;
C_GS = A_G + guard S. The strict clustering is safe_tracklets.day_data + evaluate exactly as switch_eval.py calls it
(safe_model.pkl, P 0.995, strict, filter); a guard is applied by wrapping safe_tracklets.link_tracklets (an extra
allow mask: no link whose end instances include a flagged one) and, for B_Gc, make_tracklets (tracklets cut where the
flag changes). Nothing is refitted. Saved keys are read from Whale-Review-Web/data/<day>/ (read-only).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fcommon as F  # noqa: E402
import flags as FL  # noqa: E402
from model_fine_tuning.cluster import safe_tracklets as ST  # noqa: E402

MODEL = F.REPO / "experiments" / "safe_clusters" / "safe_model.pkl"
WEB = F.FI / "state_snapshot"          # read-only copy of Whale-Review-Web/data/<day>/{labeling,pair_marks,review_extra}.json,
                                        # taken 2026-11-23 00:2x, after the switch go-live migrated the keys to the S6h root
OLD_LIVE = Path("/var/tmp/sat3737-whale-days")   # the r04 live root until the switch went live (2026-11-22 22:16)
ARMS = {"L": ("L", ""), "P": ("P", ""), "A_G": ("A_G", ""), "A_S": ("A_S", ""), "B_G": ("P", "G"), "B_Gc": ("P", "G+cut"), "B_GS": ("P", "GS"),
        "C_SG": ("A_S", "G"), "C_GS": ("A_G", "S")}


def root_of(name: str) -> Path:
    return F.SR if name == "P" else OLD_LIVE if name == "L" else F.FI / "days" / name


def get_flags(rootname: str, day: str) -> dict:
    f = F.FI / "flags" / rootname / f"{day}.json"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(FL.day_flags(root_of(rootname), day)))
    return json.loads(f.read_text())


def flagged_set(rootname: str, day: str, guard: str) -> set:
    if not guard:
        return set()
    fl = get_flags(rootname, day)["flags"]
    use_g = "G" in guard; use_s = "S" in guard.replace("+cut", "")
    return {k for k, v in fl.items() if (use_g and v["G"]) or (use_s and v["S"])}


def cluster(root: Path, day: str, flagged: set, cut: bool) -> dict:
    from model_fine_tuning.cluster.safe_clusters import load_model
    orig_lt, orig_mt = ST.link_tracklets, ST.make_tracklets

    def lt(tracks, pairs, prob, threshold, allow=None):
        by = {t["id"]: t for t in tracks}
        g = np.array([not ({by[p["a"]]["keys"][-1], by[p["b"]]["keys"][0]} & flagged) for p in pairs], bool) if len(pairs) else None
        if g is not None:
            allow = g if allow is None else (allow & g)
        return orig_lt(tracks, pairs, prob, threshold, allow)

    def mt(ts, inst, gap, app, cost):
        out = []
        for t in orig_mt(ts, inst, gap, app, cost):
            cur = None
            for k, p in zip(t["keys"], t["photos"]):
                f = k in flagged
                if cur is None or f != cur[1]:
                    cur = ({"id": len(out), "burst": t["burst"], "camera": t["camera"], "keys": [], "photos": [], "start": float(ts[p]["time"]), "end": float(ts[p]["time"])}, f)
                    out.append(cur[0])
                cur[0]["keys"].append(k); cur[0]["photos"].append(p); cur[0]["end"] = float(ts[p]["time"])
        return out
    ST.link_tracklets = lt
    if cut:
        ST.make_tracklets = mt
    try:
        dd = ST.day_data(root, day, F.LABELS, 1.5, 1.5, 300.0, 3, use_filter=True)
        rep, cl = ST.evaluate(dd, load_model(MODEL), 0.995, 300.0, strict=True)
        au = ST.audit(dd, cl)
    finally:
        ST.link_tracklets, ST.make_tracklets = orig_lt, orig_mt
    return {"links": rep["links"], "wrong_links": rep["wrong_links"], "clusters_per_animal": rep["metrics"]["clusters_per_animal"],
            "impure_photos": rep["metrics"]["photos_in_impure_clusters"], "clusters": rep["metrics"]["clusters"], "animals": rep["metrics"]["animals"],
            "multi_whale_bursts": au["multi_whale_bursts"], "split_pure": au["multi_whale_bursts_split_pure"],
            "impure": [{k: v for k, v in i.items() if k in ("photos", "catalog", "links")} for i in au["impure"]],
            "link_keys": [(dd["tracks"][a]["keys"][-1], dd["tracks"][b]["keys"][0]) for c in cl for a, b, _ in c["links"]]}


def kept(root: Path, day: str) -> dict:
    nwf = root / day / "safe" / "not_whale.json"
    drop = set(json.loads(nwf.read_text()).get("drop", [])) if nwf.exists() else set()
    out = {}
    for l in open(root / day / "parts_hr" / "index.jsonl"):
        r = json.loads(l)
        out[r["stem"]] = {i["key"]: i["bbox_2560"] for i in r.get("instances", []) if i["key"] not in drop}
    return out


def masks_full(D: Path, stem: str, W: int, H: int):
    insts, z = FL.arm_instances(D, stem)
    if not insts:
        return [], []
    x0, y0, s = z["x0y0s"]
    Minv = np.array([[1 / s, 0, x0], [0, 1 / s, y0]], np.float64)
    return [f"{stem}~{k}" for k in range(len(insts))], [cv2.warpAffine(m.astype(np.uint8), Minv, (W, H), flags=cv2.INTER_NEAREST) > 0 for m in insts]


def clicks(root: Path, day: str) -> dict:
    """switch_eval.click_score with the arm's instances (filter applied)."""
    sys.path.insert(0, str(F.REPO / "experiments" / "instances_v2"))
    import common as IC
    IC.LAB = WEB
    from model_fine_tuning.cluster.instances_v2 import old_instances
    D = root / day
    nwf = D / "safe" / "not_whale.json"; drop = set(json.loads(nwf.read_text()).get("drop", [])) if nwf.exists() else set()
    photos, oth = IC.load_marks(day)
    others, flukes, main_one, no_inst = [], [], [], 0
    for key, kp in photos.items():
        stem = key.rsplit("~", 1)[0]
        m = cv2.imread(str(F.REPO / "experiments" / "days_all" / day / "canon" / f"{stem}_mask.png"), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        H, W = m.shape
        lk, _ = old_instances(F.REPO / "experiments" / "days_all" / day, stem, W, H, F.table())
        if key not in lk:
            continue
        keys, masks = masks_full(D, stem, W, H)
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
            flukes.append(fl)
        for w in (oth.get(key) or []):
            if not w:
                continue
            m2, _, fl2, has2 = whale(w)
            others.append("none" if m2 is None else "fused" if m2 == mj else "own")
            if has2:
                flukes.append(fl2)
    c = Counter(others)
    return {"other_whales": len(others), "own": c["own"], "fused": c["fused"], "none": c["none"], "flukes": len(flukes),
            "flukes_one_instance": sum(x is True for x in flukes), "main_whales": len(main_one), "main_all_clicks_one_instance": sum(main_one),
            "clicked_photos_without_instance": no_inst}


KEY_RE = re.compile(r"[A-Za-z0-9_.\-]+__[A-Za-z0-9_.\-]+~\d+")


def saved_keys(day: str) -> list:
    keys = set()
    for name in ("labeling.json", "pair_marks.json", "review_extra.json"):
        f = WEB / day / name
        if f.exists():
            keys |= set(KEY_RE.findall(f.read_text()))
    return sorted(keys)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def attach(rootP: Path, root: Path, day: str) -> dict:
    keys = saved_keys(day)
    if not keys:
        return {"keys": 0}
    def boxes(r):
        out = {}
        for l in open(r / day / "parts_hr" / "index.jsonl"):
            for i in json.loads(l).get("instances", []):
                out[i["key"]] = i["bbox_2560"]
        return out
    bP, bA = boxes(rootP), boxes(root)
    inP = [k for k in keys if k in bP]
    same = sum(1 for k in inP if k in bA and iou(bP[k], bA[k]) >= 0.5)
    gone = sum(1 for k in inP if k not in bA)
    return {"keys": len(keys), "in_P": len(inP), "same_box": same, "changed_box": len(inP) - same - gone, "gone": gone}


def no_kept(root: Path, day: str) -> int:
    lab = F.catalogue(day); K = kept(root, day)
    return sum(1 for s in F.stems(root / day) if lab.get(s.split("__")[-1]) and not K.get(s))


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--arms", default=",".join(ARMS)); ap.add_argument("--days", default=",".join(F.CAND_DAYS))
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--no-clicks", action="store_true"); ap.add_argument("--no-attach", action="store_true")
    a = ap.parse_args()
    days = a.days.split(","); [F.check_day(d) for d in days]
    out = {"days": {}, "pooled": {}}
    click_cache = {}
    for day in days:
        out["days"][day] = {}
        for arm in a.arms.split(","):
            rootname, guard = ARMS[arm]; root = root_of(rootname)
            if not (root / day / "safe" / "not_whale.json").exists():
                print("missing", arm, day); continue
            fl = flagged_set(rootname, day, guard)
            r = cluster(root, day, fl, "cut" in guard)
            r["flagged"] = len(fl); r["no_kept"] = no_kept(root, day)
            r["instances"] = sum(len(v) for v in kept(root, day).values())
            if day in F.CLICK and not a.no_clicks:
                if rootname not in click_cache.setdefault(day, {}):
                    click_cache[day][rootname] = clicks(root, day)
                r["clicks"] = click_cache[day][rootname]
            if not a.no_attach:
                r["attach"] = attach(F.SR, root, day)
            out["days"][day][arm] = r
            print(day, arm, {k: (len(v) if isinstance(v, list) else v) for k, v in r.items() if k not in ("impure", "clicks", "attach", "link_keys")}, flush=True)
    for arm in a.arms.split(","):
        R = [out["days"][d][arm] for d in days if arm in out["days"][d]]
        if len(R) != len(days):
            continue
        s = {"days": len(R), "wrong_links": sum(len(r["wrong_links"]) for r in R), "links": sum(r["links"] for r in R),
             "clusters_per_animal_mean": round(float(np.mean([r["clusters_per_animal"] for r in R])), 3), "clusters": sum(r["clusters"] for r in R),
             "animals": sum(r["animals"] for r in R), "impure_photos": sum(r["impure_photos"] for r in R), "no_kept": sum(r["no_kept"] for r in R),
             "multi_whale_bursts": sum(r["multi_whale_bursts"] for r in R), "split_pure": sum(r["split_pure"] for r in R), "flagged": sum(r["flagged"] for r in R),
             "instances": sum(r["instances"] for r in R)}
        C = [r["clicks"] for r in R if "clicks" in r]
        if C:
            s["clicks"] = {k: sum(c[k] for c in C) for k in C[0]}
        A = [r["attach"] for r in R if r.get("attach", {}).get("keys")]
        if A:
            s["attach"] = {k: sum(x.get(k, 0) for x in A) for k in ("keys", "in_P", "same_box", "changed_box", "gone")}
        out["pooled"][arm] = s
        print("POOLED", arm, s, flush=True)
    a.out.write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
