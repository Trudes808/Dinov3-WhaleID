#!/usr/bin/env bash
# Waits until every train and val day has its part charts, then runs identity_pipeline.sh.
#   nohup setsid test_scripts/bash/identity_after_processing.sh id_v1 > experiments/identity_after_processing.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
NAME="${1:-id_v1}"; ROOT="${2:-experiments/days}"; export PYTHONPATH=.
while true; do
    missing=$(.venv/bin/python - <<PYEOF
import json, os
d = json.load(open("model_fine_tuning/splits/days.json"))
print(" ".join(x for x in d["train"] + d["val"] if not os.path.exists(f"$ROOT/{x}/parts/index.jsonl")))
PYEOF
)
    if [ -z "$missing" ]; then break; fi
    echo "$(date -u +%FT%TZ) waiting for: $missing"; sleep 600
done
echo "$(date -u +%FT%TZ) all train/val days processed; starting the identity pipeline"
exec test_scripts/bash/identity_pipeline.sh "$NAME" "$ROOT"
