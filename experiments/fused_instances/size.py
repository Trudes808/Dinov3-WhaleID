"""Step 1: how many instances of the shadow root hold two animals, on train + validation days (eval days refused).
Per instance of every photo (the parts_hr "<stem>~k" split, rebuilt from DAY/cse as parts.py does): fin / fluke
geometry (fcommon.instance_geometry), components, the SAM 3 whale instances inside it (where SAM 3 ran), the filter
state, and the truth sources: the researcher's "other whale" clicks (click days) and membership of an impure safe
cluster.

    PYTHONPATH=. .venv/bin/python experiments/fused_instances/size.py --root SHADOW_DAYS --out FI/size_rows.jsonl [--days D,..]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fcommon as F  # noqa: E402


def click_truth(root: Path, day: str) -> dict:
    """instance key (in root) -> {"other": own/fused/none list, "main": True} from the researcher's clicks."""
    sys.path.insert(0, str(F.REPO / "experiments" / "instances_v2"))
    import common as IC
    from model_fine_tuning.cluster.instances_v2 import old_instances
    photos, oth = IC.load_marks(day)
    out = {}
    for key, kp in photos.items():
        stem = key.rsplit("~", 1)[0]
        m = cv2.imread(str(root / day / "canon" / f"{stem}_mask.png"), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        H, W = m.shape
        lk, _ = old_instances(F.REPO / "experiments" / "days_all" / day, stem, W, H, F.table())
        if key not in lk:
            continue
        keys, masks = old_instances(root / day, stem, W, H, F.table())

        def major(pts):
            v = [IC.which(masks, (p["x"] * W, p["y"] * H), 25.0) for p in pts.values()]
            v = [x for x in v if x is not None]
            return Counter(v).most_common(1)[0][0] if v else None
        mj = major(kp)
        res = []
        for w in (oth.get(key) or []):
            if not w:
                continue
            m2 = major(w)
            res.append("none" if m2 is None else "fused" if m2 == mj else "own")
            if m2 is not None and m2 != mj:
                out.setdefault(keys[m2], {}).setdefault("is_other", True)
        if mj is not None:
            out.setdefault(keys[mj], {})["main_others"] = res
    return out


def one(args):
    root, day, stem, parts_row, drop = args
    from model_fine_tuning.cluster.parts import split_instances
    z = F.cse(root / day, stem)
    if z is None or z["use"].sum() < 400:
        return []
    insts = split_instances(z["use"], 0.1, u=z["u"])
    sam = F.sam_in_crop(day, stem, z["x0y0s"], z["use"].shape)
    rows = []
    for k, m in enumerate(insts):
        key = f"{stem}~{k}"
        r = {"day": day, "stem": stem, "key": key, "k": k, "n_inst": len(insts), "parts": parts_row.get(key), "dropped": key in drop}
        r.update(F.instance_geometry(m, z["u"], z["signed"], z["x0y0s"]))
        if sam is None:
            r["sam"] = None
        else:
            r["sam_n_all"] = len(sam)
            r["sam"] = [round(sh, 3) for _, _, sh in sorted(F.sam_split(m, sam, 0.0, 1), key=lambda t: -t[2])][:4]
            r["sam_px"] = [int((m & mm).sum()) for _, mm, _ in sorted(F.sam_split(m, sam, 0.0, 1), key=lambda t: -t[2])][:4]
        rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--root", type=Path, default=F.SR); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--days", default=",".join(F.TRAIN + F.VAL)); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    days = [d for d in a.days.split(",") if d]
    for d in days:
        F.check_day(d)
    jobs = []; meta = {}
    for day in days:
        D = a.root / day
        if not (D / "parts_hr" / "index.jsonl").exists():
            continue
        prow = {}
        for l in open(D / "parts_hr" / "index.jsonl"):
            r = json.loads(l)
            for i in r.get("instances", []):
                prow.setdefault(r["stem"], {})[i["key"]] = i["parts"]
        nwf = D / "safe" / "not_whale.json"
        drop = set(json.loads(nwf.read_text()).get("drop", [])) if nwf.exists() else set()
        for s in F.stems(D):
            jobs.append((a.root, day, s, prow.get(s, {}), drop))
        C = json.loads((D / "safe" / "clusters.json").read_text())
        labels = F.catalogue(day)
        imp = {}
        for c in C["clusters"]:
            cat = Counter(labels.get(p.split("__")[-1]) for p in c["main_photos"]); cat.pop(None, None)
            if len(cat) > 1:
                maj = cat.most_common(1)[0][0]
                for k in c["keys"]:
                    s = k.rsplit("~", 1)[0]
                    imp[k] = {"cluster": c["id"], "animal": labels.get(s.split("__")[-1]), "minority": labels.get(s.split("__")[-1]) != maj, "links": len(c["links"])}
        meta[day] = {"impure": imp, "clicks": click_truth(a.root, day) if day in F.CLICK else {}}
    with Pool(a.workers) as pool, open(a.out, "w") as h:
        for rows in pool.imap(one, jobs, chunksize=4):
            for r in rows:
                m = meta[r["day"]]
                r["impure"] = m["impure"].get(r["key"]); r["click"] = m["clicks"].get(r["key"])
                h.write(json.dumps(r) + "\n")
    print("rows ->", a.out)


if __name__ == "__main__":
    main()
