"""The fluke-blade join rule from data. Every close blob pair (DAY/instances_v2/blobs.jsonl, both blobs >= the floor)
with its geometry (instances_v2.pair_geometry: dy, xover, gap_w, ratio) and the view classifier's P(fluke) on the
union box. Truth from the researcher's clicks (READ ONLY, 2025-01-02 and 2025-03-18): a pair is ONE whale when the two
fluke tips of one whale lie in its two blobs, TWO whales when one blob holds the main whale's clicks and the other an
other whale's. With --sheet: random joined pairs and near misses (pairs that fail exactly one condition) as tiles.

    PYTHONPATH=. .venv/bin/python experiments/instances_v2/join_study.py [--sheet]
"""
import argparse, json, random, sys
from collections import Counter
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import common as C
from model_fine_tuning.cluster import instances_v2 as V

ap = argparse.ArgumentParser(); ap.add_argument("--sheet", action="store_true"); ap.add_argument("--tol", type=float, default=25.0); a = ap.parse_args()
R = dict(V.DEFAULTS)
rows = []
for day in C.TRAINVAL:
    S = V.load_scores(C.DAYS / day)
    for stem, r in C.blob_rows(day).items():
        by = {b["b"]: b for b in r["blobs"]}
        for p in r["pairs"]:
            i, j = p["i"], p["j"]
            if by[i]["area"] < R["floor"] or by[j]["area"] < R["floor"]:
                continue
            g = V.pair_geometry(by[i], by[j], p["gap"]); su = S[f"{stem}|{i}+{j}"]
            rows.append({"day": day, "stem": stem, "i": i, "j": j, **g, "gap": p["gap"], "pu": su[1], "pi": S[f"{stem}|{i}"][1], "pj": S[f"{stem}|{j}"][1],
                         "nwu": su[0], "area_i": by[i]["area"], "area_j": by[j]["area"], "join": V.is_fluke_pair(g, su[1], R)})
print(f"{len(rows)} close pairs of blobs >= {R['floor']} px on {len(C.TRAINVAL)} days; joined by the rule {sum(r['join'] for r in rows)}")
# truth from clicks
truth = {}
for day in ("2025-01-02", "2025-03-18"):
    photos, others = C.load_marks(day); br = C.blob_rows(day)
    for key, kp in photos.items():
        stem = key.rsplit("~", 1)[0]
        if stem not in br:
            continue
        r = br[stem]; W, H = r["W"], r["H"]; lab = C.blob_image(day, stem); ids = [b["b"] for b in r["blobs"] if b["area"] >= R["floor"]]; ms = [lab == i for i in ids]
        def blobs_of(pts):
            return {n: (None if (k := C.which(ms, (p["x"] * W, p["y"] * H), a.tol)) is None else ids[k]) for n, p in pts.items()}
        mb = blobs_of(kp); wh = [mb] + [blobs_of(o) for o in (others.get(key) or []) if o]
        for w in wh:
            l, rr = w.get("fluke_tip_left"), w.get("fluke_tip_right")
            if l and rr and l != rr:
                truth[(day, stem, min(l, rr), max(l, rr))] = "one"
        mset = {v for v in mb.values() if v}
        for w in wh[1:]:
            for x in {v for v in w.values() if v} - mset:
                for y in mset:
                    truth.setdefault((day, stem, min(x, y), max(x, y)), "two")
lk = {(r["day"], r["stem"], r["i"], r["j"]): r for r in rows}
print(f"click truth: {Counter(truth.values())} pairs; of them close pairs >= floor: {Counter(v for k, v in truth.items() if k in lk)}")
for k, v in sorted(truth.items(), key=lambda kv: kv[1]):
    r = lk.get(k)
    if r:
        print(f"   {v:3s} joined={r['join']!s:5s} pu {r['pu']:.2f} pi {r['pi']:.2f} pj {r['pj']:.2f} dy {r['dy']:.2f} xover {r['xover']:+.2f} gap_w {r['gap_w']:.2f} ratio {r['ratio']:.1f}  {k[1][-22:]}")
    else:
        print(f"   {v:3s} (not a close pair >= floor) {k}")
F = [r for r in rows if r["pu"] >= R["join_pfluke"]]
print(f"\npairs with union P(fluke) >= {R['join_pfluke']}: {len(F)}")
for nm, edges in (("dy", [0, .1, .2, .3, .45, .6, .8, 1.2, 9]), ("xover", [-9, -.5, -.25, 0, .25, .5, .75, 1.01]), ("gap_w", [0, .02, .05, .1, .15, .25, .4, .6, 9]), ("ratio", [1, 1.5, 2, 3, 4, 6, 10, 1e9])):
    h = np.histogram([r[nm] for r in F], edges)[0]
    print(f"   {nm:6s}", " | ".join(f"{edges[k]:g}-{edges[k + 1]:g}: {h[k]}" for k in range(len(h))))
cond = {"pfluke": lambda r: r["pu"] >= R["join_pfluke"], "dy": lambda r: r["dy"] <= R["join_dy"], "xover": lambda r: r["xover"] <= R["join_xover"],
        "gap": lambda r: r["gap_w"] <= R["join_gap"], "ratio": lambda r: r["ratio"] <= R["join_ratio"]}
for r in rows:
    r["fails"] = [n for n, f in cond.items() if not f(r)]
near = [r for r in rows if len(r["fails"]) == 1 and (r["pu"] >= 0.25)]
print(f"near misses (exactly one condition failed, union P(fluke) >= 0.25): {len(near)} {dict(Counter(r['fails'][0] for r in near))}")
print(f"joined per day: {dict(Counter(r['day'] for r in rows if r['join']))}")
(Path(__file__).parent / "join_pairs.json").write_text(json.dumps([{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k not in ("nwu",)} for r in rows if r["pu"] >= 0.25]))   # the fluke-like pairs only (size)

if a.sheet:
    rng = random.Random(1); J = [r for r in rows if r["join"]]; rng.shuffle(J); rng.shuffle(near)
    tiles, legend = [], []
    for kind, sel in (("joined", J[:36]), ("near", near[:24])):
        for r in sel:
            img = C.photo(r["day"], r["stem"]); H, W = img.shape[:2]; lab = C.blob_image(r["day"], r["stem"]); by = {b["b"]: b for b in C.blob_rows(r["day"])[r["stem"]]["blobs"]} if False else None
            mi, mj = lab == r["i"], lab == r["j"]; ys, xs = np.nonzero(mi | mj)
            win = C.region([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], W, H, pad=0.35, min_side=240)
            t = img.copy(); th = max(2, int((win[2] - win[0]) / 250)); C.outline(t, cv2.dilate(mi.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=th), (0, 0, 255), th); C.outline(t, cv2.dilate(mj.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=th), (0, 255, 0), th)
            t = C.cut(t, win, (480, 320))
            n = len(tiles) + 1
            C.label(t, f"{n} {kind}" + ("" if kind == "joined" else f" fails {r['fails'][0]}"), (5, 20), 0.55)
            C.label(t, f"pu {r['pu']:.2f} dy {r['dy']:.2f} xo {r['xover']:+.2f} gap {r['gap_w']:.2f} ratio {r['ratio']:.1f}", (5, 312), 0.45)
            tiles.append(t); legend.append({"tile": n, "kind": kind, "day": r["day"], "stem": r["stem"], "blobs": [r["i"], r["j"]], **{k: r[k] for k in ("pu", "dy", "xover", "gap_w", "ratio", "fails")}})
    print(C.write_sheets(tiles, legend, "s4_fluke_joins", cols=4, per=24))
