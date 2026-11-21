#!/usr/bin/env bash
# Switch prep, loss kind (a): the S6-src arm with fallback rule F (fallback_use.py) -> cse onward through the live chain.
#   experiments/test_day/switch_fallback.sh S6ROOT ARMROOT DAY ...
set -uo pipefail
cd "$(dirname "$0")/../.."
S6="${1:?s6 root}"; ARM="${2:?arm root}"; shift 2
PY=.venv/bin/python; export PYTHONPATH=.; LOCK="flock /var/tmp/sat3737-parts-cache/gpu_train.lock"
CSE=experiments/cse_r04_ft2/checkpoints/best.pt
for DAY in "$@"; do
  case "$DAY" in 2024-12-18|2024-12-26|2025-02-19|2025-03-21) echo "refusing eval day $DAY"; continue;; esac
  D=$ARM/$DAY; mkdir -p "$D"
  echo "== $(date +%T) $DAY fallback use"; [ -f "$D/fallback.json" ] || $PY experiments/test_day/fallback_use.py --src "$S6/$DAY" --dst "$D"
  echo "== $(date +%T) $DAY cse"; [ -f "$D/cse/points.jsonl" ] || $LOCK $PY -m model_fine_tuning.canon.predict_cse --checkpoint $CSE --canon-dir "$D/canon" --images "$D/images" --out "$D/cse" --overlays 0 --as-canon > "$D/cse.log" 2>&1
  echo "== $(date +%T) $DAY parts"; [ -f "$D/parts/index.jsonl" ] || $PY -m model_fine_tuning.cluster.parts --index "$D/images/index.jsonl" --canon-dir "$D/cse" --out "$D/parts" --size 512 --workers 10 --fluke-min-fraction 0.3 --fin-min-fraction 0.03 2>&1 | grep -v -i "warn\|_check_multiprocessing" | tail -1
  experiments/test_day/run_pipeline.sh "$D"
  echo "FB_DONE $DAY $(date +%T)"
done
echo "SWITCH_FB_DONE $(date +%T)"
