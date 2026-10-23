#!/usr/bin/env bash
# Per-part WildFusion matchers on the charts of the old map (parts) and the CSE map (parts_cse) of one day.
set -uo pipefail
cd "$(dirname "$0")/.."
DAY=experiments/days/2024-12-18; LAB=model_fine_tuning/splits/labels_all_days.json
for v in parts parts_cse; do
  [ -f "$DAY/wildfusion_$v/metrics.json" ] || .venv/bin/python scripts/wildfusion_parts.py --parts "$DAY/$v" --labels "$LAB" --out "$DAY/wildfusion_$v" > "$DAY/wildfusion_$v.log" 2>&1
  echo "$v done: $(date -u +%T)"
done
echo ALLDONE
