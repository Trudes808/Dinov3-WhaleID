#!/usr/bin/env bash
# Weak real labels for the part model: train W1 (and optionally the ablation without the "some fluke / fin is here" terms),
# score each on a shadow day folder. Every GPU step waits on the shared lock.
#   nohup bash experiments/parts_weak/run_w1.sh w1 noneed > /var/tmp/sat3737-parts-cache/parts_weak/run.log 2>&1 &
# Recipe = parts_seg_r05_a's phase 2 (init phase1_last.pt, 10 epochs, eval every 5, warm-up 100, batch 8, 512 crops,
# seed 0); --workers 5 for the render loader (+ 2 real-crop workers + main = 8 processes; no sample changes).
set -u
REPO=/home/sat3737/NOAA_whales/Dinov3-WhaleID
cd "$REPO"
PY="$REPO/.venv/bin/python"
LOCK="flock /var/tmp/sat3737-parts-cache/gpu_train.lock"
export PYTHONPATH="$REPO"
C=/var/tmp/sat3737-parts-cache/parts_weak
RECIPE="--init-weights /var/tmp/sat3737-parts-cache/phase1_last.pt --epochs 10 --eval-every 5 --warmup-steps 100 --batch-size 8 --crop 512 --seed 0 --workers 5"

for arm in "$@"; do
  case $arm in
    w1)     NAME=parts_seg_r05_w1;        WEAK="--n-real 2 --w-need 0.5"; ROOT=experiments; METRICS=experiments/parts_seg_r05_w1/real_metrics.json ;;
    noneed) NAME=parts_seg_r05_w1_noneed; WEAK="--n-real 2 --w-need 0";   ROOT=$C/runs;     METRICS=experiments/parts_weak/parts_seg_r05_w1_noneed_real_metrics.json ;;
    *) echo "unknown arm $arm"; exit 1 ;;
  esac
  if ! grep -q '"done"' "$ROOT/$NAME/log.jsonl" 2>/dev/null; then
    echo "$(date +%T) queue train $NAME"
    $LOCK $PY -m model_fine_tuning.parts_weak.train $WEAK -- --name $NAME --out-root $ROOT $RECIPE > $C/train_$arm.out 2>&1
    echo "$(date +%T) train $NAME exit $?"
  fi
  [ -f "$ROOT/$NAME/checkpoints/best.pt" ] || { echo "$NAME has no best.pt"; continue; }
  if [ ! -f "$METRICS" ]; then
    echo "$(date +%T) queue score $NAME"
    $LOCK $PY experiments/parts_weak/eval_w.py --name $NAME --checkpoint $ROOT/$NAME/checkpoints/best.pt --metrics $METRICS > $C/eval_$arm.out 2>&1
    echo "$(date +%T) score $NAME exit $?"
  fi
done
echo "$(date +%T) all done"
