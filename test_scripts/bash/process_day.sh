#!/usr/bin/env bash
# Process one day of RAW photos into everything the identity matcher needs:
#   images (RAW -> jpg + EXIF index) -> segmentation + canonical surface (canon.predict) -> part charts (fin, fluke, body)
# Usage: process_day.sh SRC_DAY_DIR OUT_ROOT   e.g. process_day.sh /datasets/NOAA_whales_data/2025-1-16 experiments/days
# Output: OUT_ROOT/<ISO day>/{images,canon,parts}. Idempotent per stage.
set -euo pipefail
cd "$(dirname "$0")/../.."
SRC="${1:?day folder}"; ROOT="${2:?output root}"
NAME="$(basename "$SRC")"; IFS=- read -r Y M D <<< "$NAME"; DAY=$(printf "%04d-%02d-%02d" "$Y" "$M" "$D")
OUT="$ROOT/$DAY"; mkdir -p "$OUT"
PY=.venv/bin/python; export PYTHONPATH=.
SEG=experiments/seg_r02_ft4/checkpoints/best.pt; CANON=experiments/canon_r02_ft4/checkpoints/best.pt
echo "== $DAY: images"; [ -f "$OUT/images/index.jsonl" ] || $PY test_scripts/python/convert_raw.py "$SRC" "$OUT/images" --long-side 2560
echo "== $DAY: segmentation + canonical surface"; [ -f "$OUT/canon/points.jsonl" ] || $PY -m model_fine_tuning.canon.predict --seg-checkpoint $SEG --checkpoint $CANON --images "$OUT/images" --out "$OUT/canon" --max-side 1280 > "$OUT/canon.log" 2>&1
echo "== $DAY: part charts"; [ -f "$OUT/parts/index.jsonl" ] || $PY -m model_fine_tuning.cluster.parts --index "$OUT/images/index.jsonl" --canon-dir "$OUT/canon" --out "$OUT/parts" --size 512 --workers 10 2>&1 | grep -v -i "warn\|_check_multiprocessing" | tail -1
echo "== $DAY: timestamps"; [ -f "$OUT/timestamps.json" ] || $PY - "$OUT" <<'PYEOF'
import sys, json, struct, io, datetime
from pathlib import Path
import exifread
out = Path(sys.argv[1]); rows = {}
for l in open(out / "images/index.jsonl"):
    r = json.loads(l); src = r["source"]; t = {}
    if src.lower().endswith(".cr3"):
        data = open(src, "rb").read(4_000_000)
        for tag in (b"CMT1", b"CMT2"):
            i = data.find(tag)
            if i >= 0:
                size = struct.unpack(">I", data[i - 4:i])[0]; t.update({k: str(v) for k, v in exifread.process_file(io.BytesIO(data[i + 4:i + size]), details=False).items()})
    else:
        with open(src, "rb") as fh: t = {k: str(v) for k, v in exifread.process_file(fh, details=False).items()}
    dto = t.get("EXIF DateTimeOriginal") or t.get("Image DateTimeOriginal"); sub = (t.get("EXIF SubSecTimeOriginal") or t.get("Image SubSecTimeOriginal") or "0").strip()
    ts = datetime.datetime.strptime(dto, "%Y:%m:%d %H:%M:%S").timestamp() + float("0." + sub) if dto else None
    rows[Path(r["image"]).stem] = {"time": ts, "datetime": dto, "camera": t.get("Image Model", "?"), "focal_mm": t.get("EXIF FocalLength") or t.get("Image FocalLength"), "folder": r["folder"]}
json.dump(rows, open(out / "timestamps.json", "w"), indent=1); print(len(rows), "timestamps")
PYEOF
echo "== $DAY: done"
