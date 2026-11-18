#!/usr/bin/env bash
# Second chain (2026-11-17 12:40), replaces run_arms.sh / run_fw.sh for the remaining arms, in the coordinator's order:
#   s1 (baseline recipe, seed 1: run-to-run noise), sw (swap paste, synth_aug/swap_run05.py), fw (modelled sea), rwall.
# Every GPU job under the shared lock. Progress: /var/tmp/sat3737-parts-cache/realwater/chain2.log;
# done marker: /var/tmp/sat3737-parts-cache/realwater/chain2.done (written last, also on failure of a step).
#   setsid nohup bash experiments/realwater/chain2.sh > /var/tmp/sat3737-parts-cache/realwater/chain2.log 2>&1 < /dev/null &
set -u
REPO=/home/sat3737/NOAA_whales/Dinov3-WhaleID
cd "$REPO"
PY="$REPO/.venv/bin/python"
export PYTHONPATH="$REPO"
LOCK="flock /var/tmp/sat3737-parts-cache/gpu_train.lock"
LOG=/var/tmp/sat3737-parts-cache/realwater
INIT=/var/tmp/sat3737-parts-cache/phase1_last.pt
RECIPE="--init-weights $INIT --epochs 10 --eval-every 5 --warmup-steps 100 --workers 6"
rm -f $LOG/chain2.done

trained() { [ -f "experiments/$1/checkpoints/last.pt" ] && grep -q '"done"' "experiments/$1/log.jsonl" 2>/dev/null; }
score() {
  if [ -f "experiments/$1/checkpoints/best.pt" ] && [ ! -f "experiments/$1/real_metrics.json" ]; then
    echo "$(date +%T) score $1"
    $LOCK $PY experiments/realwater/eval_arm.py --name $1 > $LOG/eval_$2.out 2>&1
    echo "$(date +%T) score $1 exit $?"
  fi
}

# s1
if ! trained parts_seg_r05_s1; then
  echo "$(date +%T) train parts_seg_r05_s1"
  $LOCK $PY -m model_fine_tuning.parts_seg.train --name parts_seg_r05_s1 $RECIPE --seed 1 > $LOG/train_s1.out 2>&1
  echo "$(date +%T) train parts_seg_r05_s1 exit $?"
fi
score parts_seg_r05_s1 s1

# sw (needs the CPU precompute)
until [ -f $LOG/sw/precompute.done ]; do sleep 60; done
if ! trained parts_seg_r05_sw; then
  echo "$(date +%T) train parts_seg_r05_sw"
  $LOCK $PY -m model_fine_tuning.synth_aug.swap_run05 train --p 0.5 -- --name parts_seg_r05_sw $RECIPE > $LOG/train_sw.out 2>&1
  echo "$(date +%T) train parts_seg_r05_sw exit $?"
fi
score parts_seg_r05_sw sw

# fw (modelled sea: plate index rebuilt from every plate with a JSON)
$PY -m model_fine_tuning.synth_aug.fw_plates build
if ! trained parts_seg_r05_fw; then
  echo "$(date +%T) train parts_seg_r05_fw"
  $LOCK $PY -m model_fine_tuning.synth_aug.train_rw --rw-mode rw_rs --rw-p 0.5 --rw-index $LOG/fw_plates_index.jsonl -- \
    --name parts_seg_r05_fw $RECIPE > $LOG/train_fw.out 2>&1
  echo "$(date +%T) train parts_seg_r05_fw exit $?"
fi
score parts_seg_r05_fw fw

# rwall
if ! trained parts_seg_r05_rwall; then
  echo "$(date +%T) train parts_seg_r05_rwall"
  $LOCK $PY -m model_fine_tuning.synth_aug.train_rw --rw-mode rw_rs --rw-p 1.0 -- --name parts_seg_r05_rwall $RECIPE > $LOG/train_rwall.out 2>&1
  echo "$(date +%T) train parts_seg_r05_rwall exit $?"
fi
score parts_seg_r05_rwall rwall

$PY experiments/realwater/compare.py --base parts_seg_r05_a --arms parts_seg_r05_sc parts_seg_r05_rw parts_seg_r05_s1 parts_seg_r05_sw \
  parts_seg_r05_fw parts_seg_r05_rwall --out experiments/realwater/compare.json > $LOG/compare_final.txt 2>&1
echo "$(date +%T) chain2 all done"
date > $LOG/chain2.done
