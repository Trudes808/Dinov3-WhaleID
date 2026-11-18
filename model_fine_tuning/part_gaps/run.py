"""ONE command for the part-model gap table, to repeat after every factory round or model change.

    PYTHONPATH=. .venv/bin/python -m model_fine_tuning.part_gaps.run --model <experiment name> \
        [--renders <factory>/data/renders/<set>/dataset ...] [--score-renders] [--variant hr|full]

1. coverage.py on every --renders dataset (labels only, CPU, minutes; cached per dataset tag)
2. the model's real-photo outputs must exist (DAY/<model>/instances.json under days_all or a days_shadow root); if not,
   prints the predict command (GPU: run it under flock /var/tmp/sat3737-parts-cache/gpu_train.lock)
3. --score-renders: render_eval.py for the model on every --renders dataset (CPU about 3-5 s per shot; --device cuda
   under the GPU lock is faster), so the "model on renders of that category" column is filled
4. report.py: the ranked gap table (real sample = labels_ai.txt + labels_ai_relook.txt, fixed; arms = DEFAULT_ARMS
   plus --model), written to experiments/part_gaps/report_<model>_<variant>.json and printed.
A new render set only needs step 1 (coverage) to update the coverage columns; a new model needs step 2 (and 3 for the
render columns). The real labels do not change between rounds, so numbers are comparable round to round.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from . import common as C
from .report import DEFAULT_RENDERS


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="parts_seg_r05_a")
    ap.add_argument("--renders", nargs="*", type=Path, default=None, help="extra render datasets, added to run05 + the run06 tests + checks")
    ap.add_argument("--score-renders", action="store_true")
    ap.add_argument("--render-n", type=int, default=None, help="shots per render set for --score-renders (run05: 300 val shots)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--variant", default="hr", choices=["hr", "full"])
    ap.add_argument("--nearest", action="store_true", help="add the extra --renders sets to the nearest-render pool, re-match, "
                    "nearest sheets (set framed orange) and a blind by-eye sheet of every whale of each extra set")
    a = ap.parse_args(argv)
    ap_extra = list(a.renders or [])
    renders = [d for d in dict.fromkeys(list(DEFAULT_RENDERS) + ap_extra) if (d / "manifest.jsonl").exists()]
    py = [sys.executable, "-m"]
    subprocess.run(py + ["model_fine_tuning.part_gaps.coverage", "--dataset", *map(str, renders)], check=True)
    if C.arm_root(a.model) is None:
        print(f"no real-photo outputs for {a.model}; run first (GPU, under the lock):\n"
              f"  flock /var/tmp/sat3737-parts-cache/gpu_train.lock .venv/bin/python -m model_fine_tuning.parts_seg.predict "
              f"--checkpoint experiments/{a.model}/checkpoints/best.pt --days train,val --out-name {a.model}")
        return 1
    if a.score_renders:
        for d in renders:
            extra = ["--split", "val", "--n", str(a.render_n or 300)] if d.parent.name.startswith("run0") else (["--n", str(a.render_n)] if a.render_n else [])
            subprocess.run(py + ["model_fine_tuning.part_gaps.render_eval", "--dataset", str(d), "--model", a.model, "--device", a.device, *extra], check=True)
    subprocess.run(py + ["model_fine_tuning.part_gaps.report", "--model", a.model, "--variant", a.variant, "--renders", *map(str, renders)], check=True)
    if a.nearest and ap_extra:
        tags = [d.parent.name if d.name == "dataset" else d.name for d in ap_extra]
        subprocess.run(py + ["model_fine_tuning.part_gaps.nearest", "extract", "--add", *tags], check=True)
        subprocess.run(py + ["model_fine_tuning.part_gaps.nearest", "match", "--model", a.model, "--variant", a.variant], check=True)
        subprocess.run(py + ["model_fine_tuning.part_gaps.sheets", "nearest", "--model", a.model, "--variant", a.variant,
                             "--mark", *tags, "--cats", "D_small", "B_back", "P_pec", "F_low"], check=True)
        for t in tags:
            subprocess.run(py + ["model_fine_tuning.part_gaps.sheets", "eye", "--tag", t], check=True)
        print("label the eye sheets blind into experiments/part_gaps/labels_ai_<tag>.txt (format of labels_ai_r6b.txt), then "
              "re-run report.py for the by-eye coverage column")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
