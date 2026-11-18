"""End to end: the instances_v2 stage (cluster/instances_v2.py, imported and run unchanged) on the masks of another
segmentation checkpoint, for the 3 validation days and the 2 click days, scored as experiments/instances_v2 scored it.

    flock /var/tmp/sat3737-parts-cache/gpu_train.lock env PYTHONPATH=. .venv/bin/python -m model_fine_tuning.seg_real.e2e run \
        --name s2 --checkpoint experiments/seg_splash_s2/checkpoints/last.pt
    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.seg_real.e2e score --names ref,s2      (ref = the existing DAY/instances_v2, seg_r04_ft4)

The stage writes into DAY/instances_v2; here DAY is a shadow folder CACHE/e2e/<name>/<day>/ whose images, cse,
parts_hr and safe are symlinks to the real day (read only) and whose instances_v2/ is new. Nothing in a day folder is
written. Scores: instances per photo (0 / 1 / 2 / 3+), candidates rejected (filter / gate), and with the researcher's
clicks (2025-01-02, 2025-03-18; experiments/instances_v2/common.py helpers, read only): the marked other whales as
own / fused / none, flukes (both tips clicked) with both tips in one instance, main whales with all clicks in one kept
instance. -> experiments/seg_splash/e2e.json
"""
from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from collections import Counter
from pathlib import Path

import numpy as np

from . import common as C

E2E = C.CACHE / "e2e"
DAYS_E2E = C.VAL + ["2025-01-02", "2025-03-18"]


def shadow(name: str, day: str) -> Path:
    C.check_day(day)
    d = E2E / name / day; (d / "instances_v2").mkdir(parents=True, exist_ok=True)
    for sub in ("images", "cse", "parts_hr", "safe"):
        if not (d / sub).exists() and (C.DAYS / day / sub).exists():
            (d / sub).symlink_to(C.DAYS / day / sub)
    return d


def root_of(name: str) -> Path:
    return C.DAYS if name == "ref" else E2E / name


def cmd_run(a) -> None:
    from ..cluster import instances_v2 as V
    for day in a.days.split(","):
        d = shadow(a.name, day)
        if not (d / "instances_v2" / "feat.npz").exists():
            V.cmd_candidates(Namespace(day_dir=str(d), collect=100, workers=4, seg_checkpoint=str(Path(a.checkpoint).resolve())))
        V.cmd_assemble(Namespace(day_dir=str(d), **V.DEFAULTS))


def score(name: str) -> dict:
    sys.path.insert(0, str(C.REPO / "experiments" / "instances_v2"))
    import common as IC
    IC.DAYS = root_of(name); IC._IDX.clear()
    res = {"per_day": {}}
    for day in DAYS_E2E:
        rows = [json.loads(l) for l in open(root_of(name) / day / "instances_v2" / "index.jsonl")]
        n = Counter(min(len(r["instances"]), 3) for r in rows)
        res["per_day"][day] = {"photos": len(rows), "inst_0": n[0], "inst_1": n[1], "inst_2": n[2], "inst_3+": n[3],
                               "instances": sum(len(r["instances"]) for r in rows),
                               "rejected_nw": sum(x["why"] == "not_whale" for r in rows for x in r["rejected"]),
                               "rejected_gate": sum(x["why"] == "gate" for r in rows for x in r["rejected"])}
    for grp, days in (("val", C.VAL), ("click_days", ["2025-01-02", "2025-03-18"])):
        P = [res["per_day"][d] for d in days]
        res[grp] = {k: sum(p[k] for p in P) for k in P[0]}
        res[grp]["instances_per_photo"] = round(res[grp]["instances"] / res[grp]["photos"], 3)
    # clicks
    others, flukes_main, flukes_other, main_one = [], [], [], []
    for day in ["2025-01-02", "2025-03-18"]:
        photos, oth = IC.load_marks(day); idx = IC.new_index(day)
        for key, kp in photos.items():
            stem = key.rsplit("~", 1)[0]
            if stem not in idx:
                continue
            r = idx[stem]; W, H = r["W"], r["H"]
            okeys, _ = IC.old_masks(day, stem, W, H)
            if key not in okeys:
                continue
            nkeys, nmasks, _ = IC.new_masks(day, stem)

            def whale(pts):
                P = {n: (p["x"] * W, p["y"] * H) for n, p in pts.items()}
                nw = {n: IC.which(nmasks, p, 25.0) for n, p in P.items()}
                v = [x for x in nw.values() if x is not None]
                major = Counter(v).most_common(1)[0][0] if v else None
                fl = None
                if "fluke_tip_left" in nw and "fluke_tip_right" in nw:
                    fl = None if None in (nw["fluke_tip_left"], nw["fluke_tip_right"]) else nw["fluke_tip_left"] == nw["fluke_tip_right"]
                return major, nw, fl, "fluke_tip_left" in P and "fluke_tip_right" in P
            mj, nw, fl, has = whale(kp)
            main_one.append(None not in nw.values() and len(set(nw.values())) == 1)
            if has:
                flukes_main.append(fl)
            for w in (oth.get(key) or []):
                if not w:
                    continue
                m2, _, fl2, has2 = whale(w)
                others.append("none" if m2 is None else "fused" if m2 == mj else "own")
                if has2:
                    flukes_other.append(fl2)
    c = Counter(others)
    res["clicks"] = {"other_whales": len(others), "own": c["own"], "fused": c["fused"], "none": c["none"],
                     "flukes_main": len(flukes_main), "flukes_main_one_instance": sum(x is True for x in flukes_main),
                     "flukes_all": len(flukes_main) + len(flukes_other), "flukes_all_one_instance": sum(x is True for x in flukes_main + flukes_other),
                     "main_whales": len(main_one), "main_all_clicks_one_instance": sum(main_one)}
    return res


def cmd_score(a) -> None:
    f = C.EXP / "e2e.json"; R = json.loads(f.read_text()) if f.exists() else {}
    for n in a.names.split(","):
        R[n] = score(n)
    f.write_text(json.dumps(R, indent=1))
    for n in a.names.split(","):
        r = R[n]
        print(n, "val", r["val"], "\n   click days", r["click_days"], "\n   clicks", r["clicks"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run"); p.add_argument("--name", required=True); p.add_argument("--checkpoint", required=True); p.add_argument("--days", default=",".join(DAYS_E2E))
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("score"); p.add_argument("--names", default="ref"); p.set_defaults(fn=cmd_score)
    a = ap.parse_args(argv); a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
