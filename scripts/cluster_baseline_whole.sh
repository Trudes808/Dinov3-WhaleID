#!/usr/bin/env bash
# Whole-crop MegaDescriptor baseline through the same burst clustering, on the eval and val days.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.
for d in 2024-12-18 2024-12-26 2025-02-19 2025-03-21 2025-01-15 2025-02-27 2025-03-14; do
  [ -f experiments/days_all/$d/cluster_bursts_whole/metrics.json ] || .venv/bin/python -m model_fine_tuning.cluster.cluster_bursts --day-dir experiments/days_all/$d \
      --labels model_fine_tuning/splits/labels_all_days.json --out experiments/days_all/$d/cluster_bursts_whole --whole-crop --gap 1.5 > experiments/days_all/$d/cluster_bursts_whole.log 2>&1
  echo "$d done $(date -u +%T)"
done
echo ALLDONE
