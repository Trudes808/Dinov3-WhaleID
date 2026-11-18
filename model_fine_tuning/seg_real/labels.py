"""Merge the by-eye sheet labels (scratchpad labels_raw/lab_NNN.json) and the reusable labels into
experiments/seg_splash/blob_labels.json.

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.seg_real.labels merge

Training use (data.py): category whale with confidence 1 -> whale (eroded core); whale with confidence 0,
whale_splash, unclear and unsure water (faint dark strips that may be a whale under the surface) -> ignore; every
other category, sure or unsure (the doubt was nearly always splash vs wake, both background) -> background.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

from . import common as C
from .sample import SAMPLE

RAW = C.SHEETS / "labels_raw"


def train_role(b: dict) -> str:
    if b["label"] == "whale":
        return "whale" if b["sure"] else "ignore"
    if b["label"] in C.IGN:
        return "ignore"
    if b["label"] == "water" and not b["sure"]:
        return "ignore"                 # unsure "water" = faint dark strips that may be a whale just under the surface
    return "background"


def cmd_merge(a) -> None:
    S = json.loads(SAMPLE.read_text()); raw = {}
    for f in sorted(RAW.glob("lab_*.json")):
        raw.update(json.loads(f.read_text()))
    out, missing = [], []
    for t in S["tiles"]:
        v = raw.get(str(t["tile"]))
        if v is None:
            missing.append(t["tile"]); continue
        cat, conf = v
        assert cat in C.CATS, cat
        src = "by eye, sheet tile %d (lab_%03d)" % (t["tile"], (t["tile"] - 1) // 28 + 1)
        if t.get("reuse"):
            src += "; re-look of " + ", ".join(r["src"] + (f" [{r.get('ai') or r.get('nw') or r.get('view')}]") for r in t["reuse"])
        out.append({**{k: t[k] for k in ("day", "stem", "b", "area", "bbox", "split", "stratum", "size")}, "label": cat, "sure": bool(conf), "source": src,
                    "who": "sheet" if 28 < t["tile"] <= 1467 else "sheet (coordinator)"})
    for k in S["known"]:
        out.append({**{x: k[x] for x in ("day", "stem", "b", "area", "bbox", "split", "stratum", "size")}, "label": k["label"], "sure": True, "source": k["source"], "who": "reused"})
    for b in out:
        b["role"] = train_role(b)
    # white-water subtype (second by-eye pass on the splash / wake / whale_splash blobs, scratchpad subtype/sub_NNN.jpg)
    leg = C.SHEETS / "subtype" / "sub_legend.json"
    if leg.exists():
        sub = {}
        for f in sorted((C.SHEETS / "subtype_raw").glob("sub_*.json")):
            sub.update(json.loads(f.read_text()))
        pos = {(b["day"], b["stem"], b["b"]): b for b in out}
        for t in json.loads(leg.read_text()):
            v = sub.get(str(t["tile"]))
            if v:
                pos[(t["day"], t["stem"], t["b"])]["subtype"] = v[0]; pos[(t["day"], t["stem"], t["b"])]["subtype_sure"] = bool(v[1])
    seen = Counter((b["day"], b["stem"], b["b"]) for b in out); assert max(seen.values()) == 1
    sheet = [b for b in out if b["who"] != "reused"]
    note = ("AI by-eye labels (Claude, 2026-11-17), NOT the researcher's. Units are blobs of DAY/instances_v2/blobs (seg_r04_ft4 raw mask, "
            "8-connected, 2560-px frame), i.e. every label names a pixel region the live segmentation calls whale. Sources: (1) blind contact sheets "
            "(model_fine_tuning/seg_real/sample.py, tiles show no score; 1 467 tiles, sheet 1 by the coordinating agent, sheets 2-53 by six "
            "labelling sub-agents with one written guide, scratchpad LABEL_GUIDE.md), (2) reused without a new look: instances_v2 blob_labels_ai*.json "
            "(whale_part / second_whale -> whale, person, drone, boat, shore), view_class/view_labels.json codes F/D/O and stage2_linker/nw_labels.json "
            "pos -> whale, attached through DAY/instances_v2/mapping.json when the old instance maps to one blob holding >= half of it. AI labels "
            "'splash' / 'other' / 'unclear' and nw 'neg' were drawn again on the sheets. 'sure' = the labeller's confidence; "
            f"unsure share on the sheets {sum(not b['sure'] for b in sheet) / max(len(sheet), 1):.0%} (mostly splash vs wake and faint dark strips "
            "whale vs water). role = how training uses it (whale only when sure). subtype (splash / wake / whale_splash blobs only) = a second "
            "by-eye pass on the kind of white water (breach_slap, back_wash, travel_wake, blow, fluke_stream, whitecap, boat_wake, foam_other, "
            "not_white_water) by two sub-agents, 87 % of them marked unsure (mostly back_wash vs travel_wake).")
    C.EXP.mkdir(parents=True, exist_ok=True)
    (C.EXP / "blob_labels.json").write_text(json.dumps({"note": note, "categories": C.CAT_NAMES, "blobs": out}, indent=0))
    print("missing tiles:", missing)
    for sp in ("train", "val"):
        B = [b for b in out if b["split"] == sp]
        print(sp, len(B), dict(Counter(b["label"] for b in B).most_common()), "roles", dict(Counter(b["role"] for b in B)),
              "unsure (sheets)", sum(not b["sure"] for b in B if b["who"] != "reused"), "/", sum(b["who"] != "reused" for b in B))
    per = defaultdict(Counter)
    for b in out:
        per[b["label"]][b["sure"]] += 1
    print({k: f"{v[True]} sure / {v[False]} unsure" for k, v in per.items()})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("merge").set_defaults(fn=cmd_merge)
    a = ap.parse_args(argv); a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
