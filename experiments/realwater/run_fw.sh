#!/usr/bin/env bash
# Modelled-sea arm parts_seg_r05_fw: the rw recipe exactly (real-water mode, p 0.5, same init / schedule / scoring), but
# the plates are the offline water engine's (synth_aug/fw_plates.py index, rebuilt right before training so it holds
# every plate whose JSON exists by then). Queued after run_arms.sh (waits for its "all done"), under the shared lock.
#   nohup bash experiments/realwater/run_fw.sh > /var/tmp/sat3737-parts-cache/realwater/run_fw.log 2>&1 &
set -u
REPO=/home/sat3737/NOAA_whales/Dinov3-WhaleID
cd "$REPO"
PY="$REPO/.venv/bin/python"
export PYTHONPATH="$REPO"
LOCK="flock /var/tmp/sat3737-parts-cache/gpu_train.lock"
INIT=/var/tmp/sat3737-parts-cache/phase1_last.pt
RECIPE="--init-weights $INIT --epochs 10 --eval-every 5 --warmup-steps 100 --workers 6"
NAME=parts_seg_r05_fw
LOG=/var/tmp/sat3737-parts-cache/realwater

echo "$(date +%T) waiting for run_arms.sh to finish"
until grep -q "all done" $LOG/run_arms.log; do sleep 120; done
echo "$(date +%T) building the modelled-sea plate index"
$PY -m model_fine_tuning.synth_aug.fw_plates build
if [ ! -f "experiments/$NAME/checkpoints/last.pt" ] || ! grep -q '"done"' "experiments/$NAME/log.jsonl" 2>/dev/null; then
  echo "$(date +%T) train $NAME"
  $LOCK $PY -m model_fine_tuning.synth_aug.train_rw --rw-mode rw_rs --rw-p 0.5 --rw-index $LOG/fw_plates_index.jsonl -- \
    --name $NAME $RECIPE > $LOG/train_fw.out 2>&1
  rc=$?
  echo "$(date +%T) train $NAME exit $rc"
fi
if [ -f "experiments/$NAME/checkpoints/best.pt" ] && [ ! -f "experiments/$NAME/real_metrics.json" ]; then
  echo "$(date +%T) score $NAME"
  $LOCK $PY experiments/realwater/eval_arm.py --name $NAME > $LOG/eval_fw.out 2>&1
  rc=$?
  echo "$(date +%T) score $NAME exit $rc"
fi
echo "$(date +%T) fw done"
