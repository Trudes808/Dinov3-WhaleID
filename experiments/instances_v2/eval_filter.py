"""Scoring b, step 2: the not-a-whale filter (experiments/safe_clusters/not_whale.pkl, threshold 2.0, unchanged) on the
blobs the 25 % clean-up deletes, against the AI's by-eye labels (blob_labels_ai.json), by blob size; and the per-day
counts of instances per photo, old against new.

    PYTHONPATH=. .venv/bin/python experiments/instances_v2/eval_filter.py [--thresholds 2.0,0,-2]
"""
import argparse, json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import common as C
from model_fine_tuning.cluster import instances_v2 as V

HERE = Path(__file__).parent
ap = argparse.ArgumentParser(); ap.add_argument("--thresholds", default="2.0"); ap.add_argument("--counts", action="store_true")
ap.add_argument("--sample", default="blob_sample.json"); ap.add_argument("--labels", default="blob_labels_ai.json"); a = ap.parse_args()
L = json.loads((HERE / a.labels).read_text())["labels"]; sample = json.loads((HERE / a.sample).read_text())
WHALE = {"whale_part", "second_whale"}
scores = {}
for day in sorted({s["day"] for s in sample}):
    scores[day] = V.load_scores(C.DAYS / day)
rows = []
for s in sample:
    lab = L.get(str(s["tile"]))
    if lab is None or lab == "unclear":
        continue
    nw, pf, pd, gt = scores[s["day"]][f"{s['stem']}|{s['b']}"]
    rows.append({**s, "label": lab, "whale": lab in WHALE, "nw": nw, "pf": pf, "pd": pd, "gate": gt})
print(f"labelled blobs: {len(rows)} of {len(sample)} (unclear left out: {sum(1 for v in L.values() if v == 'unclear')}); days {len({r['day'] for r in rows})}")
print("labels:", dict(Counter(r["label"] for r in rows)))
bins = sorted({r["bin"] for r in rows}, key=lambda b: int(b.split("-")[0]))
GATED = [("filter", float(x), False) for x in a.thresholds.split(",")] + [("filter + gate", 2.0, True)]
for name, thr, use_gate in GATED:
    keep = lambda r: r["nw"] <= thr and (not use_gate or r["gate"] >= 0)
    print(f"\n{name}: keeps a blob when its score <= {thr:g}" + (" and the blob gate says whale" if use_gate else "") + "; 'keep' is right when the blob is whale (part of a whale or a second whale)")
    print(f"  {'size px':12s} {'n':>4s} {'whale':>6s} {'kept':>5s} {'precision':>10s} {'recall':>8s}   labels of the kept non-whales")
    for b in bins + ["all >= 300", "all"]:
        sel = [r for r in rows if (r["bin"] == b) or (b == "all") or (b == "all >= 300" and r["area"] >= 300)]
        k = [r for r in sel if keep(r)]; tp = sum(r["whale"] for r in k); pos = sum(r["whale"] for r in sel)
        fp = Counter(r["label"] for r in k if not r["whale"])
        print(f"  {b:12s} {len(sel):4d} {pos:6d} {len(k):5d} {tp / max(len(k), 1):10.2f} {tp / max(pos, 1):8.2f}   {dict(fp)}")
# score separation
from sklearn.metrics import roc_auc_score
for b in bins:
    sel = [r for r in rows if r["bin"] == b]; y = [r["whale"] for r in sel]
    if 0 < sum(y) < len(y):
        print(f"  AUC of the score in bin {b}: {roc_auc_score(y, [-r['nw'] for r in sel]):.3f} (whale median score {np.median([r['nw'] for r in sel if r['whale']]):.1f}, non-whale {np.median([r['nw'] for r in sel if not r['whale']]):.1f})")
if a.sample == "blob_sample.json":
  (HERE / "filter_rows.json").write_text(json.dumps([{k: r[k] for k in ("tile", "day", "stem", "b", "area", "bin", "label", "nw", "pf", "pd")} for r in rows], indent=0))

if a.counts:
    print("\nper day: instances per photo, old (parts_hr minus not_whale.json) against new (instances_v2)")
    print(f"  {'day':10s} {'photos':>6s} | {'old 0/1/2/3+':>16s} {'old n':>6s} | {'new 0/1/2/3+':>16s} {'new n':>6s} | {'new only':>8s} {'joined':>6s} {'reject':>6s} {'old->none':>9s} {'old split':>9s}")
    T = defaultdict(int)
    for day in C.TRAINVAL:
        idx = C.new_index(day); mp = json.loads((C.DAYS / day / "instances_v2" / "mapping.json").read_text()); drop = C.old_drop(day)
        old = {json.loads(l)["stem"]: json.loads(l) for l in open(C.DAYS / day / "parts_hr" / "index.jsonl")}
        on = Counter(); nn = Counter()
        for stem, r in idx.items():
            k = len([i for i in (old.get(stem) or {}).get("instances", []) if i["key"] not in drop]); on[min(k, 3)] += 1; nn[min(len(r["instances"]), 3)] += 1
            T["old_n"] += k; T["new_n"] += len(r["instances"])
        o2n = mp["old_to_new"]; kept_old = {k: v for k, v in o2n.items() if not v["old_dropped_not_whale"]}
        none = sum(v["new"] is None for v in kept_old.values()); split = sum(len(v["all"]) > 1 for v in kept_old.values())
        jn = sum(i["joined"] for r in idx.values() for i in r["instances"]); rj = sum(len(r["rejected"]) for r in idx.values())
        oc = "/".join(str(on[i]) for i in range(4)); nc = "/".join(str(nn[i]) for i in range(4))
        print(f"  {day:10s} {len(idx):6d} | {oc:>16s} {sum(i * c for i, c in on.items()):6d} | {nc:>16s} {sum(len(r['instances']) for r in idx.values()):6d} | {len(mp['new_only']):8d} {jn:6d} {rj:6d} {none:9d} {split:9d}")
        for i in range(4):
            T[f"o{i}"] += on[i]; T[f"n{i}"] += nn[i]
        T["photos"] += len(idx); T["new_only"] += len(mp["new_only"]); T["joined"] += jn; T["rej"] += rj; T["none"] += none; T["split"] += split; T["old_kept"] += len(kept_old)
    print(f"  {'all':10s} {T['photos']:6d} | {'/'.join(str(T[f'o{i}']) for i in range(4)):>16s} {T['old_n']:6d} | {'/'.join(str(T[f'n{i}']) for i in range(4)):>16s} {T['new_n']:6d} | {T['new_only']:8d} {T['joined']:6d} {T['rej']:6d} {T['none']:9d} {T['split']:9d}")
    print(f"  old kept instances {T['old_kept']}: with no new instance {T['none']}, overlapping two or more new instances {T['split']}")
