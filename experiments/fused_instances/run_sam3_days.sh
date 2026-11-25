#!/usr/bin/env bash
# SAM 3 "humpback whale" on whole days for the fused-instance study; one day per GPU-lock hold, resumable.
#   experiments/fused_instances/run_sam3_days.sh OUTROOT DAY ...     (lists in OUTROOT/lists/DAY.tsv)
set -uo pipefail
cd "$(dirname "$0")/../.."
O="$1"; shift
PY3=/home/sat3737/NOAA_whales/Underwater-WhaleID/.venv/bin/python       # read-only use of the other thread's venv
for d in "$@"; do
  case "$d" in 2024-12-18|2024-12-26|2025-02-19|2025-03-21) [ "${FUSED_EVAL_OK-}" = 1 ] || { echo "eval day $d refused"; continue; };; esac
  while [ ! -f "$O/raw/$d/_DONE" ]; do
    flock /var/tmp/sat3737-parts-cache/gpu_train.lock $PY3 experiments/sam3_compare/run_sam3.py --list "$O/lists/$d.tsv" \
      --prompts experiments/fused_instances/prompts_whale.json --set "$d" --out "$O/raw" --budget 900 || { echo "fail $d"; break; }
  done
  echo "DAY_DONE $d $(date +%T)"
done
echo "ALL_DONE $(date +%T)"
