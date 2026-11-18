"""Predict + score one real-water arm with the parts agent's predict.py / evaluate_real.py, unchanged, except that
their day folder is a shadow copy (symlinks to experiments/days_all/<DAY>/*) so the arm's class maps are written
under /var/tmp, never into experiments/days_all (read only for this experiment).

    PYTHONPATH=. .venv/bin/python experiments/realwater/eval_arm.py --name parts_seg_r05_rw

-> experiments/<name>/real_metrics.json (same json as the baseline's), class maps in
   /var/tmp/sat3737-parts-cache/realwater/days_shadow/<DAY>/<name>/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from model_fine_tuning.common import paths  # noqa: E402
from model_fine_tuning.parts_seg import evaluate_real, predict  # noqa: E402

SHADOW = Path("/var/tmp/sat3737-parts-cache/realwater/days_shadow")


def make_shadow(days: list[str]) -> None:
    for day in days:
        src = paths.EXPERIMENTS_DIR / "days_all" / day
        dst = SHADOW / day
        dst.mkdir(parents=True, exist_ok=True)
        for child in src.iterdir():
            link = dst / child.name
            if not link.exists() and not link.is_symlink() and not child.name.startswith("parts_seg_r05_"):
                os.symlink(child, link)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--checkpoint", default=None, help="default experiments/<name>/checkpoints/best.pt")
    ap.add_argument("--days", default="train,val")
    ap.add_argument("--skip-predict", action="store_true")
    a = ap.parse_args()
    days = predict.day_list(a.days)
    make_shadow(days)
    predict.DAYS = SHADOW
    evaluate_real.DAYS = SHADOW
    ck = a.checkpoint or str(paths.EXPERIMENTS_DIR / a.name / "checkpoints" / "best.pt")
    if not a.skip_predict:
        predict.main(["--checkpoint", ck, "--days", a.days, "--out-name", a.name])
    out = paths.EXPERIMENTS_DIR / a.name / "real_metrics.json"
    evaluate_real.main(["--out", str(out), "--name", a.name, "--days", a.days])
    print(json.dumps({"wrote": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
