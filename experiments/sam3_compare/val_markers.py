"""The part B marker rule on the 884 val photos (no markers there: every call is a false marker), both gates.

    PYTHONPATH=. .venv/bin/python experiments/sam3_compare/val_markers.py --settings SETTINGS.json
"""
import argparse, json, sys
from collections import Counter
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import s3common as C

G = (427, 640)
ap = argparse.ArgumentParser(); ap.add_argument("--settings", type=Path, required=True); a = ap.parse_args()
S = json.loads(a.settings.read_text()); M = S["marker_rule"]; idx = C.prompt_index("val")
cp = {c: [idx[p] for p in ps] for c, ps in S["class_prompts"].items()}; th = S["class_thr"]
out = {}
for gate in ("sam3", "s6h"):
    calls = Counter(); items = []
    for day in C.VAL:
        for stem in C.day_stems(C.LIVE / day):
            rec = C.load("val", stem)
            empty = not any((rec[f"s{j}"] >= th["whale"]).any() for j in cp["whale"]) if gate == "sam3" else not C.s6_mask_val(day, stem, G).any()
            if not empty:
                calls["whale"] += 1; continue
            a_ = {c: C.union(rec, cp[c], th[c], G).mean() for c in ("land", "rail", "tape", "person")}
            bj = [idx[p] for p in M["bollard_prompts"]]
            bmax = max([rec[f"s{j}"].max() for j in bj if len(rec[f"s{j}"])] or [0.0]); barea = C.union(rec, bj, M["bollard_score"], G).mean()
            if a_["land"] >= M["land_min"]: c = "land"
            elif bmax >= M["bollard_score"] and barea >= M["bollard_min"]: c = "cleat"
            elif a_["tape"] >= M["tape_min"] and a_["rail"] + a_["tape"] >= M["rail_tape_min"]: c = "rail"
            elif a_["person"] >= M["person_min"]: c = "crew"
            else: c = "water"
            calls[c] += 1; items.append((day, stem.split("-")[-1], c))
    out[gate] = {"calls": dict(calls), "gated": items}
print(json.dumps(out, indent=1))
