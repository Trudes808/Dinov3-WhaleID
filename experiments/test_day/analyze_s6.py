"""Round 2, task 1: the live chain against the candidate mask end to end on the test day (rules: round2_rules.md, section 1).
Tables (a) whale in photo, (b) instances by what they sit on, (f) removal candidates, clusters, Labeling Assistant load.

    .venv/bin/python experiments/test_day/analyze_s6.py --root ROOT
Inputs: round 1 (ROOT/photos.tsv, instances.tsv, labeling_day.json, days/<day>), round 2 (ROOT/round2/days_s6/<day>,
s6_inst.json + s6_labels.json (by-eye), labeling_day_s6.json). -> ROOT/round2/s6_photos.tsv, s6_instances.tsv, s6_measures.json
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

EYE = {"W": "whale", "P": "person", "R": "rail/boat", "G": "gear", "S": "water/splash", "F": "whale fused with crew/vessel", "L": "land", "V": "vessel", "?": "unlabelled"}
WHALEY = {"W", "F"}


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True); a = p.parse_args()
    R = a.root; R2 = R / "round2"; day = R.name; D6 = R2 / "days_s6" / day; DL = R / "days" / day
    T = {r["seq"]: r for r in csv.DictReader(open(R / "truth.tsv"), delimiter="\t")}
    seq = lambda stem: stem.split("_")[-2]
    # live
    I1 = list(csv.DictReader(open(R / "instances.tsv"), delimiter="\t"))
    live = defaultdict(list)
    for r in I1:
        live[r["seq"]].append({"key": r["key_suffix"], "eye": r["eye"], "kept": r["flagged"] == "0", "cluster": r["cluster"]})
    # S6-src
    lab = json.loads((R2 / "s6_labels.json").read_text())
    P6 = json.loads((D6 / "safe" / "photos.json").read_text()); NW6 = json.loads((D6 / "safe" / "not_whale.json").read_text()); drop6 = set(NW6["drop"])
    CL6 = json.loads((D6 / "safe" / "clusters.json").read_text())["clusters"]; cl6 = {k: c["id"] for c in CL6 for k in c["keys"]}
    VC6 = json.loads((D6 / "safe" / "view_class.json").read_text()); thr = VC6.get("threshold", 0.5)
    view = lambda v: "" if v is None else "fluke" if v[0] >= thr and v[0] >= v[1] else "dorsal" if v[1] >= thr else "other"
    src = defaultdict(list); irows = []
    for stem, ph in P6.items():
        for it in ph["instances"]:
            k = it["key"]; e = lab["src"].get(k, "?")
            rec = {"arm": "S6-src", "seq": seq(stem), "key_suffix": k.split("~")[1], "eye": e, "kept": k not in drop6, "nw_score": NW6["scores"].get(k), "cluster": cl6.get(k, ""),
                   "view": view(VC6["views"].get(k)), "bbox": ",".join(str(round(x)) for x in it["bbox"])}
            src[seq(stem)].append(rec); irows.append(rec)
    # S6-v2
    v2 = defaultdict(list)
    for l in open(D6 / "instances_v2" / "index.jsonl"):
        r = json.loads(l)
        for it in r["instances"] + r["rejected"]:
            k = it.get("key") or f"{r['stem']}~rej{'+'.join(map(str, it['blobs']))}"
            rec = {"arm": "S6-v2", "seq": seq(r["stem"]), "key_suffix": k.split("~")[1], "eye": lab["v2"].get(k, "?"), "kept": "key" in it, "nw_score": it["nw"], "why": it.get("why") or "",
                   "joined": it["joined"], "bbox": ",".join(str(round(x)) for x in it["bbox"])}
            v2[seq(r["stem"])].append(rec); irows.append(rec)
    with open(R2 / "s6_instances.tsv", "w", newline="") as f:
        keys = ["arm", "seq", "key_suffix", "eye", "kept", "nw_score", "why", "joined", "cluster", "view", "bbox"]
        w = csv.DictWriter(f, fieldnames=keys, delimiter="\t", restval=""); w.writeheader(); w.writerows(irows)
    # labeling loads
    def nowhale_photos(path):
        d = json.loads(path.read_text()); idx = {i: seq(ph["stem"]) for i, ph in enumerate(d["photos"])}
        nw = {idx[i[0]] for s in d["stacks"] if s["kind"] == "nowhale" for i in s["items"]}
        wh = {idx[i[0]] for s in d["stacks"] if s["kind"] != "nowhale" for i in s["items"]}
        return nw, wh, Counter(s["kind"] for s in d["stacks"]), len(d["sightings"]), d.get("boundaries")
    LD = {"L": nowhale_photos(R / "labeling_day.json"), "S6-src": nowhale_photos(R2 / "labeling_day_s6.json")}
    arms = {"L": live, "S6-src": src, "S6-v2": v2}
    prow = []
    for s in sorted(T):
        t = T[s]; r = {"seq": s, "truth_class": t["class"], "has_whale": t["has_whale"]}
        for an, A in arms.items():
            ins = A.get(s, []); kept = [x for x in ins if x["kept"]]
            r[f"{an}_n"] = len(ins); r[f"{an}_kept"] = len(kept); r[f"{an}_kept_whale_eye"] = sum(x["eye"] in WHALEY for x in kept)
            r[f"{an}_answer"] = "whale" if kept else ("no whale (filtered)" if ins else "no whale")
        for an, (nw, wh, *_rest) in LD.items():
            r[f"{an}_page"] = "nowhale" + ("+whale" if s in wh else "") if s in nw else ("whale" if s in wh else "absent")
        prow.append(r)
    with open(R2 / "s6_photos.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(prow[0]), delimiter="\t"); w.writeheader(); w.writerows(prow)
    out = {}
    groups = [("whale above surface", lambda t: t["has_whale"].rstrip("?") == "yes"), ("whale", lambda t: t["class"] == "whale"), ("whale + crew", lambda t: t["class"] == "whale_crew"),
              ("under water only", lambda t: t["class"] == "under"), ("no whale", lambda t: t["has_whale"] == "no")] + \
             [(f"- {c}", (lambda c: lambda t: t["class"] == c)(c)) for c in ("marker_rail", "bow", "crew_nowhale", "marker_land", "water")]
    A_ = {}
    for name, f in groups:
        sel = [r for r in prow if f(T[r["seq"]])]
        A_[name] = {"n": len(sel), **{f"{an}_{k}": sum(1 for r in sel if r[f"{an}_{k2}"] > 0) for an in arms for k, k2 in (("any", "n"), ("kept", "kept"), ("kept_whale_eye", "kept_whale_eye"))},
                    **{f"{an}_page_nowhale": sum(1 for r in sel if r[f"{an}_page"].startswith("nowhale")) for an in LD}}
    out["a"] = A_
    yes = [r for r in prow if T[r["seq"]]["has_whale"].rstrip("?") == "yes"]
    lost23 = [r["seq"] for r in yes if r["L_n"] > 0 and r["L_kept"] == 0]
    out["lost23"] = {"seqs": lost23, **{an: {"kept_any": sum(1 for r in prow if r["seq"] in lost23 and r[f"{an}_kept"] > 0), "kept_whale_eye": sum(1 for r in prow if r["seq"] in lost23 and r[f"{an}_kept_whale_eye"] > 0),
                                             "still_lost": [r["seq"] for r in prow if r["seq"] in lost23 and r[f"{an}_kept"] == 0]} for an in arms},
                     **{f"{an}_page_nowhale": [r["seq"] for r in prow if r["seq"] in lost23 and r[f"{an}_page"].startswith("nowhale")] for an in LD}}
    out["missed_whale_photos"] = {an: [r["seq"] for r in yes if r[f"{an}_kept"] == 0] for an in arms}
    out["false_whale_photos"] = {an: [r["seq"] for r in prow if T[r["seq"]]["has_whale"] == "no" and r[f"{an}_kept"] > 0] for an in arms}
    out["page_nowhale_whale_photos"] = {an: [r["seq"] for r in yes if r[f"{an}_page"].startswith("nowhale")] for an in LD}
    out["page_nowhale_under_photos"] = {an: [r["seq"] for r in prow if T[r["seq"]]["class"] == "under" and r[f"{an}_page"].startswith("nowhale")] for an in LD}
    out["page"] = {an: {"stacks": dict(v[2]), "sightings": v[3], "photos_in_nowhale": len(v[0]), "photos_in_both": len(v[0] & v[1])} for an, v in LD.items()}
    out["s6_empty8"] = {s: {an: (next(r for r in prow if r["seq"] == s)[f"{an}_kept"], next(r for r in prow if r["seq"] == s).get(f"{an}_page")) for an in arms} for s in ("118", "246", "260", "261", "262", "277", "350", "416")}
    # (b)
    B = {}
    for an, A in arms.items():
        tab = defaultdict(lambda: [0, 0])
        for ins in A.values():
            for x in ins:
                tab[x["eye"]][0] += 1; tab[x["eye"]][1] += (not x["kept"])
        B[an] = {EYE.get(k, k): {"n": v[0], "filtered": v[1]} for k, v in sorted(tab.items())}
        B[an]["_total"] = {"n": sum(v[0] for v in tab.values()), "filtered": sum(v[1] for v in tab.values())}
    out["b"] = B
    out["v2_rejected_why"] = dict(Counter(f"{x['eye']}|{x['why']}" for ins in v2.values() for x in ins if not x["kept"]).most_common())
    # clusters
    eyekey = {(r["seq"], r["key_suffix"]): r["eye"] for r in irows if r["arm"] == "S6-src"}
    mixed = []
    for c in CL6:
        es = Counter(lab["src"].get(k, "?") for k in c["keys"])
        if any(e not in WHALEY for e in es) and len(c["keys"]) > 1:
            mixed.append({"id": c["id"], "size": len(c["keys"]), "eye": dict(es), "seqs": sorted({seq(k.split("~")[0]) for k in c["keys"]})})
    lone_nw = [{"id": c["id"], "eye": dict(Counter(lab["src"].get(k, "?") for k in c["keys"]))} for c in CL6 if len(c["keys"]) == 1 and lab["src"].get(c["keys"][0], "?") not in WHALEY]
    CL1 = json.loads((DL / "safe" / "clusters.json").read_text())["clusters"]
    out["clusters"] = {"L": {"clusters": len(CL1), "multi": sum(len(c["keys"]) > 1 for c in CL1)}, "S6-src": {"clusters": len(CL6), "multi": sum(len(c["keys"]) > 1 for c in CL6), "with_nonwhale_and_others": mixed, "single_nonwhale": lone_nw}}
    out["view_of_nonwhale_S6src"] = dict(Counter(f"{r['eye']}|{r['view']}" for r in irows if r["arm"] == "S6-src" and r["eye"] not in WHALEY))
    # (f)
    F = {}
    for c in ("marker_rail", "bow", "marker_land", "crew_nowhale", "water", "under", "whale_crew", "whale"):
        sel = [r for r in prow if r["truth_class"] == c]
        F[c] = {"n": len(sel), **{f"{an}_no_kept": sum(1 for r in sel if r[f"{an}_kept"] == 0) for an in arms}, **{f"{an}_page_nowhale": sum(1 for r in sel if r[f"{an}_page"].startswith("nowhale")) for an in LD}}
    out["f"] = F
    (R2 / "s6_measures.json").write_text(json.dumps(out, indent=1)); print(json.dumps({k: out[k] for k in ("a", "lost23", "page", "clusters")}, indent=1)[:6000])


if __name__ == "__main__":
    main()
