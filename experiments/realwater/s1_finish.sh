#!/usr/bin/env bash
# 2026-11-17: the paste arms were dropped by the user; only the noise run s1 (baseline recipe, seed 1) is finished.
# Waits for the s1 training (flock pid given as $1, started by chain2.sh) to end, scores s1 under the shared lock,
# writes the comparison, then the marker /var/tmp/sat3737-parts-cache/realwater/s1.done.
set -u
REPO=/home/sat3737/NOAA_whales/Dinov3-WhaleID
cd "$REPO"
PY="$REPO/.venv/bin/python"
export PYTHONPATH="$REPO"
LOG=/var/tmp/sat3737-parts-cache/realwater
while kill -0 "$1" 2>/dev/null; do sleep 60; done
echo "$(date +%T) s1 training process ended"
if grep -q '"done"' experiments/parts_seg_r05_s1/log.jsonl 2>/dev/null && [ ! -f experiments/parts_seg_r05_s1/real_metrics.json ]; then
  echo "$(date +%T) score parts_seg_r05_s1"
  flock /var/tmp/sat3737-parts-cache/gpu_train.lock $PY experiments/realwater/eval_arm.py --name parts_seg_r05_s1 > $LOG/eval_s1.out 2>&1
  echo "$(date +%T) score exit $?"
fi
$PY experiments/realwater/compare.py --base parts_seg_r05_a --arms parts_seg_r05_sc parts_seg_r05_rw parts_seg_r05_s1 \
  --out experiments/realwater/compare.json > $LOG/compare_final.txt 2>&1
{ date; [ -f experiments/parts_seg_r05_s1/real_metrics.json ] && echo "s1 trained and scored" || echo "s1 NOT scored, see $LOG/eval_s1.out / train_s1.out"; } > $LOG/s1.done
