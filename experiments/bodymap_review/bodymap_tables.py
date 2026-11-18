"""Tables from bodymap_stats.py: photos with one whale instance against photos with several."""
import json, sys
from collections import Counter, defaultdict
import numpy as np
rows = [r for r in json.load(open(sys.argv[1])) if "inst" in r and r.get("blobs", 0) > 0]
print(f"{len(rows)} photos with a whale mask")
def kept(r): return [x for x in r["inst"] if not x.get("not_whale")]
for r in rows: r["n"] = len(kept(r))
c = Counter(min(r["n"], 4) for r in rows); print("whale instances per photo (after the not-a-whale filter):", {k: f"{v} ({v / len(rows):.1%})" for k, v in sorted(c.items())})
c = Counter((min(r["blobs"], 4), min(r["n_inst"], 4)) for r in rows); print("mask blobs (>= 10 % of the largest) x instances after the merge rule:", dict(sorted(c.items())))
print("\nby the catalogue's group type: photos, share with 2+ instances, mean instances")
by = defaultdict(list)
for r in rows: by[r["gtype"] or "not in catalogue"].append(r["n"])
for g, v in sorted(by.items(), key=lambda kv: -len(kv[1])): print(f"   {g:18s} {len(v):5d}  {np.mean(np.array(v) >= 2):.1%}  {np.mean(v):.2f}")

def med(v): v = [x for x in v if x is not None]; return float(np.median(v)) if v else float("nan")
print("\nthe map's own mirror test per photo (points.jsonl), by instances in the photo")
for name, sel in (("1 instance", lambda r: r["n"] == 1), ("2 instances", lambda r: r["n"] == 2), ("3+ instances", lambda r: r["n"] >= 3)):
    s = [r for r in rows if sel(r)]
    print(f"   {name:13s} n {len(s):5d} | mirror disagreement median {med([r['flip_m'] for r in s]):.2f} m | share over 1 m {np.mean([r['flip_m'] > 1 for r in s if r['flip_m'] is not None]):.1%} | side agreement under mirroring median {med([r['side_flip'] for r in s]):.2f} | rotation head flips, median deg {med([r['rot_flip_deg'] for r in s]):.0f}")

print("\nper instance: where the head points against the flank shown, how the body fraction runs along the blob, mirror test inside the instance")
def inst_rows(sel):
    out = []
    for r in rows:
        for x in kept(r):
            if sel(r, x): out.append((r, x))
    return out
G = (("one whale in the photo", lambda r, x: r["n"] == 1), ("several: the largest", lambda r, x: r["n"] >= 2 and x["k"] == 0), ("several: the others", lambda r, x: r["n"] >= 2 and x["k"] >= 1))
for name, sel in G:
    s = inst_rows(sel); t = np.abs(np.array([x["turn"] for _, x in s])); uc = np.array([x["u_corr"] for _, x in s]); el = np.array([x["elong"] for _, x in s]); lp = np.array([x["long_px"] for _, x in s])
    fl = np.array([x["flip_m"] for _, x in s if x["flip_m"] is not None]); sp = np.array([x["side_purity"] for _, x in s if x["side_purity"] is not None])
    longish = el >= 2.5
    print(f"   {name:24s} n {len(s):5d} | long side in the 512 crop median {np.median(lp):.0f} px | head against flank (turn over 120 deg) {np.mean(t > 120):.1%}, 30-120 {np.mean((t > 30) & (t <= 120)):.1%} | on elongated blobs only: over 120 {np.mean(t[longish] > 120):.1%} (n {longish.sum()})"
          f" | body fraction follows the long axis (|r| >= 0.7) {np.mean(uc >= 0.7):.1%}, elongated only {np.mean(uc[longish] >= 0.7):.1%} | mirror disagreement median {np.median(fl):.2f} m | one flank for >= 90 % of pixels {np.mean(sp >= 0.9):.1%}")

print("\ncut-out flags against the view classifier (confident answers only: P >= 0.9 or <= 0.1)")
for name, sel in G:
    s = [(r, x) for r, x in inst_rows(sel) if x.get("vc")]
    for part, j in (("fluke", 0), ("fin", 1)):
        yes = [(part in x["parts"]) for _, x in s if x["vc"][j] >= 0.9]; no = [(part in x["parts"]) for _, x in s if x["vc"][j] <= 0.1]
        print(f"   {name:24s} {part:5s}: found when the whale shows one {np.mean(yes):.1%} (n {len(yes)}) | cut out when it shows none {np.mean(no):.1%} (n {len(no)})")
print("\nthree-way call (fluke / dorsal fin / other) from the cut-out flags against the classifier's (all instances with an answer)")
for name, sel in G:
    s = [(r, x) for r, x in inst_rows(sel) if x.get("vc")]
    def three(f, d): return "fluke" if f else "fin" if d else "other"
    a = [three("fluke" in x["parts"], "fin" in x["parts"]) == three(x["vc"][0] >= 0.5, x["vc"][1] >= 0.5) for _, x in s]
    print(f"   {name:24s} agree {np.mean(a):.1%} (n {len(a)})")
print("\ninstances with two separate dorsal-fin patches in the map (a sign of two whales in one instance), and instances made of several blobs")
for name, sel in G:
    s = inst_rows(sel); print(f"   {name:24s} two or more fin patches {np.mean([x['fin_patches'] >= 2 for _, x in s]):.1%} | made of 2+ blobs {np.mean([x['pieces'] >= 2 for _, x in s]):.1%}")
print("\nnot-a-whale instances by rank:", Counter((min(x["k"], 2), bool(x.get("not_whale"))) for r in rows for x in r["inst"]))
