#!/usr/bin/env bash
# Process one day of RAW photos with the run04 models and the CSE map (2026-10-18):
#   images (RAW -> jpg + EXIF index) -> segmentation + crops + direct rotation (canon.predict: seg_r04_ft4 + canon_r0304_ft4)
#   -> CSE map in the canon layout (predict_cse --as-canon, cse_r04_ft2) -> gated part charts -> timestamps
# Usage: process_day_v2.sh "SRC_DAY_DIR" OUT_ROOT     (the day folder name starts with YYYY-M-D; output OUT_ROOT/<ISO day>/)
set -euo pipefail
cd "$(dirname "$0")/../.."
SRC="${1:?day folder}"; ROOT="${2:?output root}"
NAME="$(basename "$SRC")"; DATE="${NAME%% *}"; IFS=- read -r Y M D <<< "$DATE"; DAY=$(printf "%04d-%02d-%02d" "$((10#$Y))" "$((10#$M))" "$((10#$D))")
OUT="$ROOT/$DAY"; mkdir -p "$OUT"
PY=.venv/bin/python; export PYTHONPATH=.
SEG=experiments/seg_r04_ft4/checkpoints/best.pt; CANON=experiments/canon_r0304_ft4/checkpoints/best.pt; CSE=experiments/cse_r04_ft2/checkpoints/best.pt
echo "== $DAY: images"; [ -f "$OUT/images/index.jsonl" ] || $PY test_scripts/python/convert_raw.py "$SRC" "$OUT/images" --long-side 2560 --workers "${RAW_WORKERS:-10}" > "$OUT/images.log" 2>&1
echo "== $DAY: segmentation + crops"; [ -f "$OUT/canon/points.jsonl" ] || $PY -m model_fine_tuning.canon.predict --seg-checkpoint $SEG --checkpoint $CANON --images "$OUT/images" --out "$OUT/canon" --max-side 1280 > "$OUT/canon.log" 2>&1
echo "== $DAY: CSE map"; [ -f "$OUT/cse/points.jsonl" ] || $PY -m model_fine_tuning.canon.predict_cse --checkpoint $CSE --canon-dir "$OUT/canon" --images "$OUT/images" --out "$OUT/cse" --overlays 0 --as-canon > "$OUT/cse.log" 2>&1
echo "== $DAY: part charts"; [ -f "$OUT/parts/index.jsonl" ] || $PY -m model_fine_tuning.cluster.parts --index "$OUT/images/index.jsonl" --canon-dir "$OUT/cse" --out "$OUT/parts" --size 512 --workers 10 --fluke-min-fraction 0.3 --fin-min-fraction 0.03 2>&1 | grep -v -i "warn\|_check_multiprocessing" | tail -1
echo "== $DAY: timestamps"; [ -f "$OUT/timestamps.json" ] || $PY - "$OUT" <<'PYEOF'
import sys, json, struct, io, datetime
from pathlib import Path
import exifread
out = Path(sys.argv[1]); rows = {}
for l in open(out / "images/index.jsonl"):
    r = json.loads(l); src = r["source"]; t = {}
    try:
        if src.lower().endswith(".cr3"):
            data = open(src, "rb").read(4_000_000)
            for tag in (b"CMT1", b"CMT2"):
                i = data.find(tag)
                if i >= 0:
                    size = struct.unpack(">I", data[i - 4:i])[0]; t.update({k: str(v) for k, v in exifread.process_file(io.BytesIO(data[i + 4:i + size]), details=False).items()})
        else:
            with open(src, "rb") as fh: t = {k: str(v) for k, v in exifread.process_file(fh, details=False).items()}
    except Exception:
        t = {}
    dto = t.get("EXIF DateTimeOriginal") or t.get("Image DateTimeOriginal"); sub = (t.get("EXIF SubSecTimeOriginal") or t.get("Image SubSecTimeOriginal") or "0").strip()
    try:
        ts = datetime.datetime.strptime(dto, "%Y:%m:%d %H:%M:%S").timestamp() + float("0." + sub) if dto else None
    except Exception:
        ts = None
    rows[Path(r["image"]).stem] = {"time": ts, "datetime": dto, "camera": t.get("Image Model", "?"), "focal_mm": t.get("EXIF FocalLength") or t.get("Image FocalLength"), "folder": r["folder"]}
json.dump(rows, open(out / "timestamps.json", "w"), indent=1); print(len(rows), "timestamps")
PYEOF
echo "== $DAY: done"
