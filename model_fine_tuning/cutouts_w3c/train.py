"""The pair_teacher_g -> pair_student_g recipe (cluster/pairnet.py, unchanged) with fins from a W3c cut-out set: the
"RP" arm of cutouts_v3/train.py (fins from the set where it exists, else parts_up; bodies and flukes from parts_up as
R0), pointed at this cache. cutouts_v3/train.py itself is not changed.

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.cutouts_w3c.train --fin s0_wide -- <pairnet arguments>
"""
from __future__ import annotations

import argparse
import sys

from . import CACHE


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fin", default="s0_wide")
    argv = sys.argv[1:]; cut = argv.index("--") if "--" in argv else len(argv)
    a = ap.parse_args(argv[:cut])
    from ..cutouts_v3 import train as v3train
    from ..cluster import pairnet
    v3train.CACHE = CACHE                                   # scan_pairs_v3 reads CACHE/<day>/<fin>/<key>.jpg|npz
    v3train.ARM.update(arm="RP", fin=a.fin)
    pairnet.scan_pairs = v3train.scan_pairs_v3
    return pairnet.main(argv[cut + 1:])


if __name__ == "__main__":
    raise SystemExit(main())
