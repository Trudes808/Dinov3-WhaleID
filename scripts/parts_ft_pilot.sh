#!/usr/bin/env bash
# Real-photo fine-tune pilot of the fin / fluke landmark detector (2026-11-02): parts_v2 fine-tuned on the researcher's
# clicks of the pilot train day(s), then predicted on the eval day 2024-12-18 (never trained on) and scored against
# the researcher's clicks there (scratchpad eval/landmark_error.py: PCK@0.1 of the clicked part extent).
#   scripts/parts_ft_pilot.sh NAME [extra train args...]      e.g. NAME=parts_v2_ft_pilotA --real-only
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; NAME="$1"; shift
EVAL=/tmp/claude-1072903897/-home-sat3737-NOAA-whales/4fbba7ff-3ba5-4b50-9cb7-970c81250409/scratchpad/eval/landmark_error.py
OUT=experiments/part_landmarks/$NAME; mkdir -p "$OUT"
echo "== train $NAME $(date -u +%FT%TZ)"
$PY -m model_fine_tuning.landmarks.parts train --name "$NAME" --real-days 2025-01-02 --init-from experiments/part_landmarks/parts_v2/best.pt \
    --epochs 12 --batch-size 8 --lr-head 1e-4 --lr-backbone 1e-5 --warmup-steps 20 --real-repeat 4 --workers 6 "$@" 2>&1 | grep -v -i warn
echo "== predict 2024-12-18 $(date -u +%FT%TZ)"
for ck in best last; do
  $PY -m model_fine_tuning.landmarks.parts predict --ckpt "$OUT/$ck.pt" --day 2024-12-18 --suffix "_${NAME}_$ck" --sheet-count 48 2>&1 | grep -v -i warn | tail -1
done
$PY -m model_fine_tuning.landmarks.parts predict --ckpt "$OUT/best.pt" --day 2024-12-18 --suffix "_${NAME}_bestr" --refine --sheet-count 48 2>&1 | grep -v -i warn | tail -1
echo "== score $(date -u +%FT%TZ)"
$PY -I "$EVAL" 2024-12-18 parts_v2 parts_v2r "${NAME}_best" "${NAME}_last" "${NAME}_bestr"
echo "== done $(date -u +%FT%TZ)"
