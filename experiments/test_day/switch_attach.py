"""Switch prep: do the researcher's saved Labeling Assistant labels still attach after the switch? Reads two switch_load.py
outputs of one day (live load, candidate load; both with the same scratch copy of the saved state) and compares per photo
the animal the export gives (photo_rows), the instance boxes behind the saved keys, and the analyze.py style numbers
against the catalogue (photos in an animal whose majority catalogue animal is theirs, pair precision / recall).

    .venv/bin/python experiments/test_day/switch_attach.py LIVE.json CAND.json [STATE.json]
"""
from __future__ import annotations

import itertools
import json
import sys
from collections import Counter


def iou(a, b):
    if not a or not b:
        return 0.0
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u else 0.0


def scores(rows):
    sh = lambda x: (x or "-").split("_")[-1]
    cat = [sh(r["catalog"]) for r in rows]; mine = [(sh(r["id"]) if r["id"] else ("x" if r["rejected"] else "open")) for r in rows]
    M = set(mine) - {"x", "open"}; C = [c for c in set(cat) if c != "-"]
    tab = Counter(zip(mine, cat)); maj = {m: max(C, key=lambda c: tab[(m, c)]) for m in M}
    pure = sum(tab[(m, maj[m])] for m in M); n = sum(tab[(m, c)] for m in M for c in C)
    idx = [i for i in range(len(rows)) if cat[i] != "-" and mine[i] not in ("x", "open")]; tp = fp = fn = 0
    for i, j in itertools.combinations(idx, 2):
        a, b = mine[i] == mine[j], cat[i] == cat[j]; tp += a and b; fp += a and not b; fn += b and not a
    return {"sorted_photos": n, "in_majority_animal": pure, "open": mine.count("open"), "no_whale": mine.count("x"), "animals": len(M),
            "pair_precision": round(tp / max(1, tp + fp), 3), "pair_recall": round(tp / max(1, tp + fn), 3)}


def main() -> None:
    A, B = (json.load(open(f)) for f in sys.argv[1:3])
    ra = {r["stem"]: r for r in A["rows"]}; rb = {r["stem"]: r for r in B["rows"]}
    pa = {p["stem"]: p for p in A["photos"]}; pb = {p["stem"]: p for p in B["photos"]}
    lost = [s for s in ra if ra[s]["aid"] and not rb[s]["aid"]]; changed = [s for s in ra if ra[s]["aid"] and rb[s]["aid"] and ra[s]["aid"] != rb[s]["aid"]]
    gained = [s for s in ra if not ra[s]["aid"] and rb[s]["aid"]]
    # the main instance (~0) behind a photo's saved key: the same box in both loads?
    moved = []
    for s in ra:
        if ra[s]["aid"] and pa[s]["inst"] and pb[s]["inst"]:
            v = iou(pa[s]["inst"][0], pb[s]["inst"][0])
            if v < 0.5:
                moved.append((s, round(v, 2)))
    # every saved instance key that exists in both loads: is it the same whale (box IoU >= 0.5), another instance of the
    # live load (the numbering moved: a swap), or a box the live load did not have?
    keymap = Counter()
    sa = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else None
    if sa:
        for key in sa.get("assign", {}):
            stem, _, k = key.rpartition("~")
            if k == "x" or stem not in pa:
                continue
            k = int(k); la, lb = pa[stem]["inst"], pb[stem]["inst"]
            if k >= len(la) or k >= len(lb) or not la[k] or not lb[k]:
                keymap["not in one load"] += 1; continue
            if iou(la[k], lb[k]) >= 0.5:
                keymap["same box"] += 1
            elif any(iou(la[j], lb[k]) >= 0.5 for j in range(len(la)) if j != k and la[j]):
                keymap["swapped: now another live instance"] += 1
            else:
                keymap["changed box"] += 1
    why = Counter()
    for s in lost:
        why[f"live {'~0' if pa[s]['inst'] else '~x'} -> cand {'~0' if pb[s]['inst'] else '~x'}"] += 1
    out = {"day": A["day"], "photos": len(ra), "saved_on_live": sum(1 for s in ra if ra[s]["aid"]), "still_attached": sum(1 for s in ra if ra[s]["aid"] and rb[s]["aid"] == ra[s]["aid"]),
           "lost": len(lost), "lost_why": dict(why), "changed_animal": len(changed), "gained": len(gained), "main_box_moved": moved, "saved_keys_vs_boxes": dict(keymap),
           "state_keys_live": A["state_keys"], "state_keys_cand": B["state_keys"], "scores_live": scores(A["rows"]), "scores_cand": scores(B["rows"]),
           "lost_photos": [(s.split("__")[-1], ra[s]["catalog"]) for s in lost]}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
