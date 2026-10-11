#!/usr/bin/env bash
# Waits for the factory's run02 build to finish, checks the dataset, then runs run02_pipeline.sh.
#   nohup setsid test_scripts/bash/run02_after_build.sh > experiments/run02_pipeline.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
RUN="$HOME/NOAA_whales/Whale-Image-Data-Factory/data/renders/run02"
LOG="$RUN.log"
echo "[$(date -u +%FT%TZ)] waiting for $LOG"
while ! grep -q "build_dataset: done" "$LOG" 2>/dev/null; do
    if grep -qE "Traceback|refus|terminat" "$LOG" 2>/dev/null; then echo "build log shows an error; not training"; exit 1; fi
    if ! pgrep -f "build_dataset.sh data/renders/run02" >/dev/null; then echo "build process gone without done marker; not training"; exit 1; fi
    sleep 120
done
n=$(wc -l < "$RUN/dataset/manifest.jsonl")
echo "[$(date -u +%FT%TZ)] build done: $n shots in manifest"
[ "$n" -ge 1900 ] || { echo "too few shots; not training"; exit 1; }
with2=$(.venv/bin/python - <<PY
import json, numpy as np
from PIL import Image
rows=[json.loads(l) for l in open("$RUN/dataset/manifest.jsonl")]
print(sum(r.get("through_water_pixels",0)>0 for r in rows))
PY
)
echo "shots with through-water pixels: $with2"
[ "$with2" -gt 100 ] || { echo "through-water class missing; not training"; exit 1; }
exec test_scripts/bash/run02_pipeline.sh "$RUN"
