#!/usr/bin/env bash
# Real-water measuring stick: train + score the arms one after the other, each after the GPU is free of other training.
#   nohup bash experiments/realwater/run_arms.sh sc rw rwall s1 > /var/tmp/sat3737-parts-cache/realwater/run_arms.log 2>&1 &
# Recipe = the baseline parts_seg_r05_a's phase 2 exactly (same init phase1_last.pt, 10 epochs, eval every 5, warmup 100,
# batch 8, 512 crops, 6 workers, seed 0); only the training dataset differs (synth_aug/train_rw.py).
set -u
REPO=/home/sat3737/NOAA_whales/Dinov3-WhaleID
cd "$REPO"
PY="$REPO/.venv/bin/python"
LOCK="flock /var/tmp/sat3737-parts-cache/gpu_train.lock"   # shared GPU queue for every training / long inference
export PYTHONPATH="$REPO"
BASE=experiments/parts_seg_r05_a
INIT=/var/tmp/sat3737-parts-cache/phase1_last.pt
RECIPE="--init-weights $INIT --epochs 10 --eval-every 5 --warmup-steps 100 --workers 6"

wait_gpu() {
  # the shared flock queues trainings from all agents; counting other training processes here would also count the
  # ones merely queued on the lock and starve this runner, so only the lock decides (set 2026-11-17 03:58)
  return 0
}

echo "$(date +%T) waiting for the baseline ($BASE/checkpoints/best.pt and $BASE/real_metrics.json)"
while [ ! -f "$BASE/checkpoints/best.pt" ] || [ ! -f "$BASE/real_metrics.json" ]; do sleep 120; done
echo "$(date +%T) baseline present"

for arm in "$@"; do
  case $arm in
    rw)    MODE=rw_rs; P=0.5 ;;
    sc)    MODE=rs;    P=0.5 ;;
    rwall) MODE=rw_rs; P=1.0 ;;
    s1)    MODE=none;  P=0.0 ;;     # the baseline recipe with seed 1: run-to-run noise
    *) echo "unknown arm $arm"; exit 1 ;;
  esac
  NAME=parts_seg_r05_$arm
  if [ "$MODE" = rw_rs ]; then        # real water needs the finished plate index
    while ! grep -q "^plates " /var/tmp/sat3737-parts-cache/realwater/plates_build.log 2>/dev/null; do sleep 60; done
  fi
  if [ ! -f "experiments/$NAME/checkpoints/last.pt" ] || ! grep -q '"done"' "experiments/$NAME/log.jsonl" 2>/dev/null; then
    wait_gpu
    echo "$(date +%T) train $NAME mode=$MODE p=$P"
    if [ "$MODE" = none ]; then
      $LOCK $PY -m model_fine_tuning.parts_seg.train --name $NAME $RECIPE --seed 1 > /var/tmp/sat3737-parts-cache/realwater/train_$arm.out 2>&1; rc=$?
    else
      $LOCK $PY -m model_fine_tuning.synth_aug.train_rw --rw-mode $MODE --rw-p $P -- --name $NAME $RECIPE \
        > /var/tmp/sat3737-parts-cache/realwater/train_$arm.out 2>&1; rc=$?
    fi
    echo "$(date +%T) train $NAME exit $rc"
  fi
  if [ ! -f "experiments/$NAME/checkpoints/best.pt" ]; then echo "$(date +%T) $NAME has no best.pt, not scored"; continue; fi
  if [ ! -f "experiments/$NAME/real_metrics.json" ]; then
    echo "$(date +%T) score $NAME"
    $LOCK $PY experiments/realwater/eval_arm.py --name $NAME > /var/tmp/sat3737-parts-cache/realwater/eval_$arm.out 2>&1
    rc=$?
    echo "$(date +%T) score $NAME exit $rc"
  fi
done
echo "$(date +%T) all done"
