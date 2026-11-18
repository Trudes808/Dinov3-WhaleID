"""Nothing got worse? Every OLD kept instance (parts_hr minus safe/not_whale.json) that overlaps no NEW kept instance:
why (the new stage rejected it as not a whale / by the gate, its blob is below the floor, no blob), and a sheet of a
random sample of them (left the old instance in red, right the new kept instances in green and the rejected in red).

    PYTHONPATH=. .venv/bin/python experiments/instances_v2/lost_old.py [--n 24]
"""
import argparse, json, random, sys
from collections import Counter
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import common as C

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=24); a = ap.parse_args()
lost = []
for day in C.TRAINVAL:
    mp = json.loads((C.DAYS / day / "instances_v2" / "mapping.json").read_text())["old_to_new"]; idx = C.new_index(day)
    for k, v in mp.items():
        if v["new"] is None and not v["old_dropped_not_whale"]:
            lost.append({"day": day, "key": k, "stem": k.rsplit("~", 1)[0], "old_px": v["old_px"]})
why = Counter()
for x in lost:
    r = C.new_index(x["day"])[x["stem"]]; img_lab = C.blob_image(x["day"], x["stem"])
    okeys, om = C.old_masks(x["day"], x["stem"], r["W"], r["H"]); m = om[okeys.index(x["key"])]
    ids, cnt = np.unique(img_lab[m], return_counts=True); ov = {int(i): int(c) for i, c in zip(ids, cnt) if i > 0}
    if not ov:
        x["why"] = "no raw blob >= 100 px under it"
    else:
        b = max(ov, key=ov.get); rj = [y for y in r["rejected"] if b in y["blobs"]]
        area = next(bb["area"] for bb in C.blob_rows(x["day"])[x["stem"]]["blobs"] if bb["b"] == b) if False else None
        x["why"] = f"rejected: {rj[0]['why']}" if rj else "blob below the floor"
        x["blob"] = b; x["nw"] = rj[0]["nw"] if rj else None; x["gate"] = rj[0].get("gate") if rj else None
    why[x["why"]] += 1
print(f"old kept instances without a new kept instance: {len(lost)} -> {dict(why)}")
(Path(__file__).parent / "lost_old.json").write_text(json.dumps(lost, indent=0))
rng = random.Random(5); sel = lost[:]; rng.shuffle(sel); tiles, legend = [], []
for n, x in enumerate(sel[:a.n], 1):
    r = C.new_index(x["day"])[x["stem"]]; img = C.photo(x["day"], x["stem"]); H, W = img.shape[:2]
    okeys, om = C.old_masks(x["day"], x["stem"], W, H); m = om[okeys.index(x["key"])]
    t1 = img.copy(); C.outline(t1, m, (0, 0, 255), 6)
    t2 = img.copy(); lab = C.blob_image(x["day"], x["stem"]); nk, nm, _ = C.new_masks(x["day"], x["stem"])
    for y in r["rejected"]:
        C.outline(t2, np.isin(lab, y["blobs"]), (0, 0, 255), 6)
    for q in nm:
        C.outline(t2, q, (0, 220, 0), 6)
    ps = [C.fit(t) for t in (t1, t2)]
    C.label(ps[0], f"{n} old {x['key'][-2:]}"); C.label(ps[1], x["why"][:34], scale=0.42)
    tiles.append(np.hstack(ps)); legend.append({"tile": n, **x})
print(C.write_sheets(tiles, legend, "s6_old_lost", cols=3, per=24))
