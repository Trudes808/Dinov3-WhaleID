"""The pair_teacher_g -> pair_student_g recipe (cluster/pairnet.py, unchanged) re-run on other cut-outs.

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.cutouts_v3.train --arm R0|R1|RP [--r1-fin finB|finP6] -- <pairnet arguments>

pairnet's data come from scan_pairs(..., xr_tag="parts_hr"): the first-rule 1536-px cut-outs of DAY/parts_hr. Here
that function is replaced, nothing else:
  R0  every part from DAY/parts_up (the cut-outs the pages use since 2026-11-16)
  R1  bodies from DAY/parts_up; flukes from CACHE/<day>/fluke where a v3 cut-out exists, else parts_up; fins from
      CACHE/<day>/<r1-fin> where it exists, else parts_up (A)
  RP  (pose_side retrain test, 2026-11-17) fins from CACHE/<day>/<r1-fin> (e.g. finP6) where it exists, else parts_up;
      bodies AND flukes from parts_up exactly as R0
The key set is that of parts_up in both arms (main instances ~0 of labelled photos, as scan_pairs), so the two arms
see the same animals, photos and bursts; only the pictures differ. The networks read the jpg (any size; resized to
512 by pairnet) and the npz's "whale" mask (token mask).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import CACHE

ARM = {"arm": "R0", "fin": "finB"}
PARTS = ("body", "fin", "fluke")


def scan_pairs_v3(data_root: Path, labels: dict, days: list[str], xr_tag: str = "parts_hr", hr_tag: str = "parts_hr") -> dict:
    from ..cluster.bursts import make_bursts
    out = {}; n_v3 = {p: 0 for p in PARTS}; n_all = {p: 0 for p in PARTS}
    for day in days:
        d = Path(data_root) / day
        if not (d / "timestamps.json").exists():
            continue
        ts = json.loads((d / "timestamps.json").read_text()); photos = labels["days"].get(day, {}).get("photos", {})
        burst_of = {s: b["id"] for b in make_bursts(ts, 1.5) for s in b["photos"]}
        groups: dict = {}
        for part in PARTS:
            src = d / "parts_up"
            for npz in src.glob(f"*__{part}.npz"):
                key = npz.name[: -len(f"__{part}.npz")]
                stem = key.rsplit("~", 1)[0] if "~" in key else key
                if "~" in key and not key.endswith("~0"):
                    continue
                ident = photos.get(stem.split("__")[-1])
                if ident is None or ident.split("_")[1].startswith("O") or stem not in burst_of:
                    continue
                jpg, z = src / "charts" / f"{key}__{part}.jpg", npz
                if ARM["arm"] == "RP" and part == "fin":       # pose_side retrain test: fins only, bodies and flukes as R0
                    vs = CACHE / day / ARM["fin"]
                    if (vs / f"{key}.jpg").exists():
                        jpg, z = vs / f"{key}.jpg", vs / f"{key}.npz"; n_v3[part] += 1
                if ARM["arm"] == "R1" and part != "body":
                    vs = CACHE / day / ("fluke" if part == "fluke" else ARM["fin"])
                    if (vs / f"{key}.jpg").exists():
                        jpg, z = vs / f"{key}.jpg", vs / f"{key}.npz"; n_v3[part] += 1
                n_all[part] += 1
                groups.setdefault((ident, part), {}).setdefault(burst_of[stem], []).append({"key": key, "part": part, "jpg": jpg, "npz": z, "day": day, "stem": stem, "ident": ident})
        out[day] = {k: v for k, v in groups.items() if len(v) >= 2}
    print(f"[cutouts_v3 {ARM['arm']}] {len(days)} days, cut-outs {n_all}, of them v3 {n_v3}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", choices=["R0", "R1", "RP"], required=True); ap.add_argument("--r1-fin", default="finB", choices=["finB", "finC", "finP", "finP4", "finP6"])
    argv = sys.argv[1:]; cut = argv.index("--") if "--" in argv else len(argv)
    a = ap.parse_args(argv[:cut]); ARM.update(arm=a.arm, fin=a.r1_fin)
    from ..cluster import pairnet
    pairnet.scan_pairs = scan_pairs_v3
    return pairnet.main(argv[cut + 1:])


if __name__ == "__main__":
    raise SystemExit(main())
