"""Write (or check) the freeze of removal rule v2: moonshot_cache/sam3_labeller/v3/removal_FREEZE.json with the sha256 of
the rule file and of every piece of code it runs, the thresholds, and the time.

    .venv/bin/python experiments/removal_v2/freeze.py write [--note TEXT]
    .venv/bin/python experiments/removal_v2/freeze.py check      # exit 1 on any mismatch (run before the held-out day)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NW = REPO.parent
OUT = NW / "moonshot_cache" / "sam3_labeller" / "v3" / "removal_FREEZE.json"
RULE = REPO / "model_fine_tuning" / "seg_multi" / "removal.py"
CODE = [REPO / "experiments" / "removal_v2" / "prompts_trace.json", REPO / "experiments" / "sam3_compare" / "run_sam3.py",
        REPO / "experiments" / "removal_v2" / "run_sam3_cpu.py",
        REPO / "experiments" / "removal_v2" / "score_cull.py", NW / "moonshot_cache" / "labeling_markers" / "markers_adapter.py"]
INPUTS = [REPO / "model_fine_tuning" / "seg_multi" / "markers.py", REPO / "model_fine_tuning" / "seg_multi" / "run_new_day.sh"]      # the runner that makes the inputs: recorded, not checked (frozen by its own FREEZE.json)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["write", "check"]); ap.add_argument("--note", default="")
    a = ap.parse_args()
    if a.cmd == "check":
        f = json.loads(OUT.read_text()); bad = [p for p, h in {f["rule_file"]: f["rule_sha256"], **f["code_sha256"]}.items() if sha(Path(p)) != h]
        print("FREEZE OK" if not bad else "FREEZE MISMATCH: " + ", ".join(bad)); sys.exit(1 if bad else 0)
    sys.path.insert(0, str(REPO))
    from model_fine_tuning.seg_multi import removal as R
    doc = {"what": "removal rule v2 (experiments/removal_v2/README.md): propose only when every reading sees no trace of a whale; else 'faint whale?'",
           "time": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "unix": time.time(),
           "rule_file": str(RULE), "rule_sha256": sha(RULE), "code_sha256": {str(p): sha(p) for p in CODE},
           "inputs_at_freeze_sha256": {str(p): sha(p) for p in INPUTS if p.exists()},
           "thresholds": {"whale_px": R.WHALE_PX, "person": R.PERSON, "land": R.LAND, "trace": {k: list(v) for k, v in R.TRACE.items()}},
           "sam3_families": R.FAMILIES, "sam3_weights": "facebook/sam3 (local HF cache), via the Underwater-WhaleID venv read-only",
           "developed_on": "2026-04-07 (development day, second by-eye truth); checked on val 2025-01-15 / 2025-02-27 / 2025-03-14 and train 2025-01-02 / 2025-01-17 / 2025-02-12 / 2025-03-06 / 2025-03-18; evaluation days not read",
           "note": a.note}
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(doc, indent=1)); print(OUT)


if __name__ == "__main__":
    main()
