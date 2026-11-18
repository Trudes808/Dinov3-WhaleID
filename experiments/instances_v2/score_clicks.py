"""Scoring a: the old instances (DAY/cse + parts.split_instances, minus safe/not_whale.json) against the new ones
(DAY/instances_v2) on the researcher's keypoint clicks (Whale-Review-Web data/<day>/pair_marks.json, READ ONLY).
A click belongs to the kept instance that holds it, else the nearest within --tol px (2560-px frame), else none.
A whale (the main one of a record, or one of its other_whales) belongs to the instance most of its clicks fall in.

    PYTHONPATH=. .venv/bin/python experiments/instances_v2/score_clicks.py [--days 2025-01-02,2025-03-18] [--tol 25] [--sheet]

-> experiments/instances_v2/clicks_result.json (per whale: old / new instance and why none); with --sheet the other
whales' photos, old instances (left) against new (right), in the scratchpad (clicks: white = main whale, yellow = others).
"""
import argparse, json, sys
from collections import Counter
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import common as C

ap = argparse.ArgumentParser(); ap.add_argument("--days", default="2025-01-02,2025-03-18"); ap.add_argument("--tol", type=float, default=25.0); ap.add_argument("--sheet", action="store_true")
a = ap.parse_args()
res, tiles, legend = [], [], []
for day in a.days.split(","):
    photos, others = C.load_marks(day); idx = C.new_index(day); brows = C.blob_rows(day)
    for key, kp in photos.items():
        stem = key.rsplit("~", 1)[0]
        if stem not in idx:
            continue
        r = idx[stem]; W, H = r["W"], r["H"]
        okeys, omasks = C.old_masks(day, stem, W, H); drop = C.old_drop(day)
        if key not in okeys:
            continue
        ok = [i for i, k in enumerate(okeys) if k not in drop]; om = [omasks[i] for i in ok]
        nkeys, nmasks, nrec = C.new_masks(day, stem); blobs = C.blob_image(day, stem); by = {b["b"]: b for b in brows[stem]["blobs"]}
        rejected_blobs = {b: x["nw"] for x in r["rejected"] for b in x["blobs"]}
        largest = max([b["area"] for b in by.values()] or [1])

        def whale(pts: dict):
            P = {n: (p["x"] * W, p["y"] * H) for n, p in pts.items()}
            o = {n: C.which(om, p, a.tol) for n, p in P.items()}; nw = {n: C.which(nmasks, p, a.tol) for n, p in P.items()}
            def major(d):
                v = [x for x in d.values() if x is not None]
                return Counter(v).most_common(1)[0][0] if v else None
            out = {"pts": P, "old": o, "new": nw, "old_major": major(o), "new_major": major(nw)}
            # why there is no instance: look at the raw blobs under the clicks
            bl = [b for b in (C.which([blobs == i for i in by], p, a.tol) for p in P.values()) if b is not None]
            if bl:
                bid = list(by)[Counter(bl).most_common(1)[0][0]]; b = by[bid]
                out["raw_blob"] = {"b": bid, "area": b["area"], "share_of_largest": round(b["area"] / largest, 3), "nw": rejected_blobs.get(bid)}
            if "fluke_tip_left" in P and "fluke_tip_right" in P:
                out["fluke"] = {"old": None if None in (o["fluke_tip_left"], o["fluke_tip_right"]) else o["fluke_tip_left"] == o["fluke_tip_right"],
                                "new": None if None in (nw["fluke_tip_left"], nw["fluke_tip_right"]) else nw["fluke_tip_left"] == nw["fluke_tip_right"]}
            return out

        main = whale(kp); oth = [whale(w) for w in (others.get(key) or []) if w]
        def single(w, side):
            v = [x for x in w[side].values()]
            return None if None in v else len(set(v)) == 1
        rec = {"day": day, "key": key, "n_old": len(om), "n_new": len(nmasks), "main": {"old_one": single(main, "old"), "new_one": single(main, "new"),
               "old_hit": main["old_major"] is not None, "new_hit": main["new_major"] is not None, "fluke": main.get("fluke")}, "others": []}
        for w in oth:
            e = {}
            for side, mj in (("old", main["old_major"]), ("new", main["new_major"])):
                m = w[f"{side}_major"]; e[side] = "none" if m is None else "fused" if m == mj else "own"
            e["raw_blob"] = w.get("raw_blob"); e["fluke"] = w.get("fluke"); e["n_pts"] = len(w["pts"])
            rec["others"].append(e)
        res.append(rec)
        if a.sheet and oth:
            img = C.photo(day, stem); allp = np.array([p for w in [main] + oth for p in w["pts"].values()])
            win = C.region([allp[:, 0].min(), allp[:, 1].min(), allp[:, 0].max(), allp[:, 1].max()], W, H, pad=0.6, min_side=500)
            panels = []
            for side, masks, keys in (("old", om, [okeys[i] for i in ok]), ("new", nmasks, nkeys)):
                t = img.copy()
                for i, m in enumerate(masks):
                    C.outline(t, m, C.COLOURS[i % len(C.COLOURS)], 4)
                for w, col in [(main, (255, 255, 255))] + [(w, (0, 255, 255)) for w in oth]:
                    for p in w["pts"].values():
                        cv2.circle(t, (int(p[0]), int(p[1])), 9, (0, 0, 0), -1); cv2.circle(t, (int(p[0]), int(p[1])), 6, col, -1)
                t = C.fit(t[win[1]:win[3], win[0]:win[2]])
                C.label(t, f"{len(tiles) + 1} {side}: " + " ".join(e[side] for e in rec["others"]))
                panels.append(t)
            tiles.append(np.hstack(panels)); legend.append({"tile": len(tiles), "day": day, "stem": stem, "key": key, "others": [{k: e[k] for k in ("old", "new")} for e in rec["others"]]})
(Path(__file__).parent / "clicks_result.json").write_text(json.dumps(res, indent=0))

def pct(n, d):
    return f"{n} / {d} ({100 * n / max(d, 1):.0f} %)"
O = [e for r in res for e in r["others"]]
print(f"photos (old keys) with clicks: {len(res)}; other whales marked: {len(O)} on {sum(1 for r in res if r['others'])} photos (tolerance {a.tol:.0f} px)")
for side in ("old", "new"):
    c = Counter(e[side] for e in O)
    print(f"  {side}: own kept instance {c['own']}, fused with the main whale {c['fused']}, no kept instance {c['none']}")
for side in ("old", "new"):
    nn = [e for e in O if e[side] == "none"]
    why = Counter("no raw blob >= 100 px" if not e["raw_blob"] else f"raw blob below floor" if e["raw_blob"]["area"] < 300 else "rejected by the not-a-whale filter" if e["raw_blob"]["nw"] is not None
                  else ("blob < 25 % of the largest" if e["raw_blob"]["share_of_largest"] < 0.25 else "other (old: split / not-whale drop)") for e in nn)
    print(f"    {side} none, why: {dict(why)}")
for who, sel in (("main whales", [r["main"].get("fluke") for r in res]), ("other whales", [e.get("fluke") for e in O])):
    sel = [f for f in sel if f]
    for side in ("old", "new"):
        v = [f[side] for f in sel]
        print(f"  flukes with both tips clicked ({who}): {len(sel)}; {side}: both tips in one instance {pct(sum(x is True for x in v), len(v))}, in two {sum(x is False for x in v)}, a tip without instance {sum(x is None for x in v)}")
for side in ("old", "new"):
    one = [r["main"][f"{side}_one"] for r in res]
    print(f"  main whales: {side} all clicks in one kept instance {pct(sum(x is True for x in one), len(one))}; a click outside every kept instance or split {sum(x is not True for x in one)}; main whale without instance {sum(not r['main'][f'{side}_hit'] for r in res)}")
worse = [r["key"] for r in res if r["main"]["old_one"] and not r["main"]["new_one"]]; better = [r["key"] for r in res if not r["main"]["old_one"] and r["main"]["new_one"]]
print(f"  main whale: one instance before but not now {len(worse)}; now but not before {len(better)}")
for k in worse:
    print("     worse:", k)
if a.sheet:
    print(C.write_sheets(tiles, legend, "s1_other_whales", cols=3, per=24))
