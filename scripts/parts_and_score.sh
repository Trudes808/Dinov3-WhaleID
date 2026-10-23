#!/usr/bin/env bash
# Charts from a canon-layout map dir + per-part WildFusion scoring on 2024-12-18.
#   scripts/parts_and_score.sh <canon dir> <tag> [fluke_min_fraction] [fin_min_fraction]
set -uo pipefail
cd "$(dirname "$0")/.."
CANON="$1"; TAG="$2"; FL="${3:-0.0}"; FN="${4:-0.0}"; DAY=experiments/days/2024-12-18; LAB=model_fine_tuning/splits/labels_all_days.json
[ -f "$DAY/parts_$TAG/index.jsonl" ] || .venv/bin/python -m model_fine_tuning.cluster.parts --index "$DAY/images/index.jsonl" --canon-dir "$CANON" --out "$DAY/parts_$TAG" --size 512 --workers 8 --fluke-min-fraction "$FL" --fin-min-fraction "$FN" 2>&1 | grep -v -i "warn\|_check_multiprocessing" | tail -n 1
[ -f "$DAY/wildfusion_parts_$TAG/metrics.json" ] || .venv/bin/python scripts/wildfusion_parts.py --parts "$DAY/parts_$TAG" --labels "$LAB" --out "$DAY/wildfusion_parts_$TAG" > "$DAY/wildfusion_parts_$TAG.log" 2>&1
echo "ALLDONE $TAG"
