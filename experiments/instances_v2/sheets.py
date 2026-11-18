"""Sheets for review (scratchpad only, never the repo): (2) random newly kept instances (no old key overlaps them) with
the filter's score, (3) random candidates the filter rejects, (5) photos where the old pipeline's main instance (~0)
is not a whale (dropped by safe/not_whale.json, or rejected by the filter now), old instances (left) against new (right).

    PYTHONPATH=. .venv/bin/python experiments/instances_v2/sheets.py [--n 48] [--n5 24]
"""
import argparse, json, random, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import common as C

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=48); ap.add_argument("--n5", type=int, default=24); a = ap.parse_args()
rng = random.Random(3); newk, rej, main_bad = [], [], []
for day in C.TRAINVAL:
    idx = C.new_index(day); mp = json.loads((C.DAYS / day / "instances_v2" / "mapping.json").read_text()); only = set(mp["new_only"]); drop = C.old_drop(day)
    for stem, r in idx.items():
        for i in r["instances"]:
            if i["key"] in only:
                newk.append({"day": day, "stem": stem, **i})
        for x in r["rejected"]:
            rej.append({"day": day, "stem": stem, **x})
        k0 = f"{stem}~0"
        if k0 in mp["old_to_new"]:
            m = mp["old_to_new"][k0]
            if k0 in drop or m["new"] is None:
                main_bad.append({"day": day, "stem": stem, "old_dropped": k0 in drop, "new_for_old0": m["new"], "n_new": len(r["instances"])})
print(f"newly kept instances {len(newk)}, rejected candidates {len(rej)}, photos whose old ~0 is not a whale (dropped, or no kept new instance on it) {len(main_bad)}"
      f" -> of these, photos with a kept new instance elsewhere {sum(1 for m in main_bad if m['n_new'] > 0)}")
(Path(__file__).parent / "main_not_whale.json").write_text(json.dumps(main_bad, indent=0))
for tag, pool in (("s2_newly_kept", newk), ("s3_rejected", rej)):
    rng.shuffle(pool); sel = pool[:a.n]; tiles, legend = [], []
    for n, s in enumerate(sel, 1):
        s = dict(s, tile=n, lines=[f"nw {s['nw']:+.1f} gate {'-' if s.get('gate') is None else format(s['gate'], '+.1f')}", f"fl {s['p_fluke']:.2f} df {s['p_dorsal']:.2f}" + ("  joined" if s.get("joined") else "") + (f"  {s['why']}" if s.get("why") else "")])
        tiles.append(C.blob_tile(s, zoom=240, ctx=(240, 160))); legend.append({k: s[k] for k in ("tile", "day", "stem", "blobs", "area", "nw", "gate", "why", "p_fluke", "p_dorsal") if k in s})
    print(C.write_sheets(tiles, legend, tag, cols=4, per=24))
rng.shuffle(main_bad); tiles, legend = [], []
for n, m in enumerate(main_bad[:a.n5], 1):
    img = C.photo(m["day"], m["stem"]); H, W = img.shape[:2]; okeys, om = C.old_masks(m["day"], m["stem"], W, H); drop = C.old_drop(m["day"])
    nkeys, nm, nrec = C.new_masks(m["day"], m["stem"]); r = C.new_index(m["day"])[m["stem"]]; lab = C.blob_image(m["day"], m["stem"])
    panels = []
    t = img.copy()
    for i, (k, mk) in enumerate(zip(okeys, om)):
        C.outline(t, mk, (0, 0, 255) if k in drop else C.COLOURS[1 + i % 6], 6)
    t = C.fit(t); C.label(t, f"{n} old (red = dropped as not whale)", (5, 18), 0.45); panels.append(t)
    t = img.copy()
    for x in r["rejected"]:
        C.outline(t, np.isin(lab, x["blobs"]), (0, 0, 255), 6)
    for i, mk in enumerate(nm):
        C.outline(t, mk, C.COLOURS[1 + i % 6], 6)
    t = C.fit(t); C.label(t, f"new: {len(nm)} kept, red = rejected", (5, 18), 0.45); panels.append(t)
    tiles.append(np.hstack(panels)); legend.append({"tile": n, **m})
print(C.write_sheets(tiles, legend, "s5_old_main_not_whale", cols=3, per=24))
