#!/usr/bin/env bash
# Switch prep (2026-11-20): the candidate mask S6h swapped in at the source (s6_swap.py canon; everything behind it the
# live chain, run_pipeline.sh unchanged) on catalogue days, into an untracked shadow root. Production day folders are
# read only (images are symlinked, timestamps copied).
#   experiments/test_day/switch_val.sh OUT_ROOT full DAY ...     whole chain (val days)
#   experiments/test_day/switch_val.sh OUT_ROOT inst DAY ...     canon + cse + parts_hr only (click days: instance counts)
# Eval days are refused.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT="${1:?out root}"; MODE="${2:?full|inst}"; shift 2
PY=.venv/bin/python; export PYTHONPATH=.; LOCK="flock /var/tmp/sat3737-parts-cache/gpu_train.lock"
CSE=experiments/cse_r04_ft2/checkpoints/best.pt
for DAY in "$@"; do
  case "$DAY" in 2024-12-18|2024-12-26|2025-02-19|2025-03-21) echo "refusing eval day $DAY"; continue;; esac
  L=experiments/days_all/$DAY; D=$OUT/$DAY
  mkdir -p "$D"; [ -e "$D/images" ] || ln -s "$(realpath $L/images)" "$D/images"; [ -f "$D/timestamps.json" ] || cp "$L/timestamps.json" "$D/"
  echo "== $(date +%T) $DAY canon (candidate mask)"; [ -f "$D/canon/points.jsonl" ] || $LOCK $PY experiments/test_day/s6_swap.py canon --images "$D/images" --out "$D/canon" > "$D/canon.log" 2>&1
  echo "== $(date +%T) $DAY cse"; [ -f "$D/cse/points.jsonl" ] || $LOCK $PY -m model_fine_tuning.canon.predict_cse --checkpoint $CSE --canon-dir "$D/canon" --images "$D/images" --out "$D/cse" --overlays 0 --as-canon > "$D/cse.log" 2>&1
  if [ "$MODE" = inst ]; then
    [ -f "$D/parts_hr/index.jsonl" ] || $PY -m model_fine_tuning.cluster.parts --index "$D/images/index.jsonl" --canon-dir "$D/cse" --out "$D/parts_hr" --size 1536 --workers 12 --fluke-min-fraction 0.3 --fin-min-fraction 0.03 --instances 2>&1 | grep -v -i "warn\|_check_multiprocessing" | tail -1
    echo "INST_DONE $DAY $(date +%T)"; continue
  fi
  echo "== $(date +%T) $DAY parts"; [ -f "$D/parts/index.jsonl" ] || $PY -m model_fine_tuning.cluster.parts --index "$D/images/index.jsonl" --canon-dir "$D/cse" --out "$D/parts" --size 512 --workers 10 --fluke-min-fraction 0.3 --fin-min-fraction 0.03 2>&1 | grep -v -i "warn\|_check_multiprocessing" | tail -1
  experiments/test_day/run_pipeline.sh "$D"
  echo "FULL_DONE $DAY $(date +%T)"
done
echo "SWITCH_VAL_DONE $(date +%T)"
