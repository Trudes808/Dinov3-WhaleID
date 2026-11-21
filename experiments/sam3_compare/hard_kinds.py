"""Hard kinds on the test day (rules_sam3.md B5): selection from round 1's truth only, measured per kind for S6h, live
and SAM 3, and the sheet sample (6 per kind, random.Random(20261121)).

    PYTHONPATH=. .venv/bin/python experiments/sam3_compare/hard_kinds.py --settings SETTINGS.json --out OUT.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import s3common as C  # noqa: E402

G = (427, 640)


def kinds(truth: dict) -> dict:
    w = {s: t for s, t in truth.items() if t["class"] in ("whale", "whale_crew", "under")}
    return {"thin back awash": sorted(s for s, t in w.items() if re.search(r"awash|wash|thin", t["note"], re.I)),
            "splash or blow": sorted(s for s, t in w.items() if t["blow"] == "1" or t["splash"] == "1"),
            "just under the surface": sorted(s for s, t in w.items() if t["class"] == "under"),
            "far small": sorted(s for s, t in w.items() if t["far"] == "1"),
            "white fluke underside": sorted(s for s, t in w.items() if re.search(r"underside", t["note"], re.I)),
            "two whales": sorted(s for s, t in w.items() if t["n_whales"] in ("2", "2?"))}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--settings", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(); S = json.loads(a.settings.read_text()); idx = C.prompt_index("test")
    wj = [idx[p] for p in S["class_prompts"]["whale"]]; wthr = S["class_thr"]["whale"]
    truth = C.truth_tsv(); K = kinds(truth); res = {}
    for k, ss in K.items():
        n3 = n6 = nl = 0; ious = []; share3 = []; share6 = []
        for s in ss:
            rec = C.load("test", s); u = np.zeros(G, bool)
            for j in wj:
                u |= C.union(rec, j, wthr, G)
            s6 = C.test_s6(s, G); lv = C.test_live(s, G)
            n3 += u.any(); n6 += s6.any(); nl += lv.any()
            if u.any() or s6.any():
                ious.append((u & s6).sum() / (u | s6).sum())
            share3.append(u.mean()); share6.append(s6.mean())
        res[k] = {"n": len(ss), "sam3_any": int(n3), "s6h_any": int(n6), "live_any": int(nl),
                  "median_iou_sam3_s6h": round(float(np.median(ious)), 3) if ious else None,
                  "mean_share_sam3": round(float(np.mean(share3)), 4), "mean_share_s6h": round(float(np.mean(share6)), 4),
                  "sheet": sorted(random.Random(20261121).sample(ss, min(6, len(ss))))}
    a.out.write_text(json.dumps(res, indent=1)); print(json.dumps({k: {x: v[x] for x in v if x != "sheet"} for k, v in res.items()}, indent=1))
    print(",".join(s for v in res.values() for s in v["sheet"]))


if __name__ == "__main__":
    main()
