"""Val days (1c): catalogue photos SAM 3 leaves without a whale, and SAM 3 whale instances off our masks (live OR S6h),
for one phrasing / threshold. Rows to OUT (untracked).

    PYTHONPATH=. .venv/bin/python experiments/sam3_compare/val_whale.py --prompt whale --thr 0.5 --out OUT.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import s3common as C

G = (427, 640)
ap = argparse.ArgumentParser(); ap.add_argument("--prompt", default="whale"); ap.add_argument("--thr", type=float, default=0.5); ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args(); j = C.prompt_index("val")[a.prompt]; cat = C.catalogue()
empty, off, n_inst, n_cat, live_empty, s6_empty = [], [], 0, 0, [], []
for day in C.VAL:
    for stem in C.day_stems(C.LIVE / day):
        if not cat[day]["photos"].get(stem.split("__")[-1]):
            continue
        n_cat += 1
        rec = C.load("val", stem); s, _, ms = C.inst(rec, j, a.thr, G)
        lv = C.live_mask(day, stem, G); s6 = C.s6_mask_val(day, stem, G); ours = lv | s6
        if not lv.any(): live_empty.append(f"{day}/{stem}")
        if not s6.any(): s6_empty.append(f"{day}/{stem}")
        if not ms:
            empty.append(f"{day}/{stem}")
        for sc, m in zip(s, ms):
            n_inst += 1; on = (m & ours).sum() / max(m.sum(), 1)
            if on < 0.3:
                off.append({"item": f"{day}/{stem}", "score": round(float(sc), 3), "on_ours": round(float(on), 3), "share": round(float(m.mean()), 5)})
res = {"prompt": a.prompt, "thr": a.thr, "catalogue": n_cat, "sam3_no_instance": len(empty), "sam3_instances": n_inst,
       "sam3_instances_off_ours": len(off), "live_mask_empty": len(live_empty), "s6_mask_empty": len(s6_empty),
       "empty_items": empty, "off_items": off, "both_empty_sam3_and_s6": sorted(set(empty) & set(s6_empty)), "s6_empty_items": s6_empty}
a.out.write_text(json.dumps(res, indent=1)); print({k: v for k, v in res.items() if not isinstance(v, list)})
