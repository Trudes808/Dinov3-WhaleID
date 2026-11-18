"""Cut-outs v3 (2026-11-17 night): flukes turned and never flipped (instances_v2 blade joins, stock direction from the
part map parts_seg_r05_a or from the photo), dorsal fins centred on the part map's fin region at two scales; the
fingerprints of them scored against the catalogue on the validation days. Offline: nothing of a day folder is written.
See make.py, embed.py, score.py, train.py, sheets.py."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DAYS = ROOT / "experiments" / "days_all"
CACHE = Path("/var/tmp/sat3737-parts-cache/cutouts_v3")
EVAL_DAYS = {"2024-12-18", "2024-12-26", "2025-02-19", "2025-03-21"}


def day_list(spec: str) -> list[str]:
    import json
    split = json.loads((ROOT / "model_fine_tuning" / "splits" / "days.json").read_text())
    days = [d for x in spec.split(",") if x for d in (split[x] if x in split else [x])]
    bad = [d for d in days if d in EVAL_DAYS]
    if bad:
        raise SystemExit(f"evaluation days must not be read: {bad}")
    return days
