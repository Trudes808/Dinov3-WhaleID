#!/usr/bin/env bash
# Pilot r06_pilot22 training test (plan.md here). ARM = add (pilot share 1/3) or only (pilot share 1). Seeds 0 and 1:
# train (W3e's command through model_fine_tuning/pilot22_train/train.py), predict + score real photos (predict.py +
# evaluate_real.py unchanged, on the OLD day root W3e was scored on: pilot22_train/oldroot.py), run05 val synth IoU (w3e_scores.py synth, unchanged), pilot val IoU (score_pilot.py), part-gap report
# (report.py unchanged, old root, --out outside the repo). Every GPU step under the lock, logged to $C/gpu_time.log.
#   setsid nohup bash experiments/pilot22_train/run_p22.sh add > /home/sat3737/NOAA_whales/moonshot_cache/pilot22/d/run_p22_add.log 2>&1 &
# Waits for the other task's GPU gate first. Marker $C/p22_<arm>.done, or p22_<arm>.failed.
set -u
ARM=$1
REPO=/home/sat3737/NOAA_whales/Dinov3-WhaleID
cd "$REPO"
PY="$REPO/.venv/bin/python"
LOCKF=/var/tmp/sat3737-parts-cache/gpu_train.lock
export PYTHONPATH="$REPO"
C=/home/sat3737/NOAA_whales/moonshot_cache/pilot22/d
PW=/var/tmp/sat3737-parts-cache/parts_weak
E=experiments/pilot22_train
GATE=/tmp/claude-1072903897/-home-sat3737-NOAA-whales/4fbba7ff-3ba5-4b50-9cb7-970c81250409/scratchpad/slabfix22
case $ARM in add) SHARE=0.3333333;; only) SHARE=1;; *) echo "arm add|only"; exit 2;; esac
fail() { echo "$(date +%T) FAILED: $1"; echo "$1" > $C/p22_$ARM.failed; exit 1; }
gpu() {   # gpu TAG command...  (runs under the lock; start / end seconds logged after the lock is taken)
  local TAG=$1; shift
  flock $LOCKF bash -c 'echo "START $(date +%s) $0" >> '"$C"'/gpu_time.log; "$@"; rc=$?; echo "END $(date +%s) $0 rc=$rc" >> '"$C"'/gpu_time.log; exit $rc' "$TAG" "$@"
}
PYTHONPATH=$REPO $PY -m model_fine_tuning.pilot22_train.oldroot make
until [ -f $GATE/check.done ] || [ -f $GATE/check.failed ]; do sleep 60; done
echo "$(date +%T) gate open ($(ls $GATE | grep check))"
for N in parts_seg_r05_w3e parts_seg_r05_w3es1; do
  [ -f $C/pilot_val/$N.json ] || gpu pilotval_$N $PY -m model_fine_tuning.pilot22_train.score_pilot --model $N --checkpoint $PW/runs/$N/checkpoints/best.pt > $C/pilotval_$N.out 2>&1 || echo "pilot val $N failed"
done
for SEED in 0 1; do
  NAME=parts_seg_p22$ARM; [ $SEED = 1 ] && NAME=parts_seg_p22${ARM}s1
  RUNS=$C/runs
  RECIPE="--init-weights /var/tmp/sat3737-parts-cache/phase1_last.pt --epochs 10 --eval-every 5 --warmup-steps 100 --batch-size 8 --crop 512 --seed $SEED --workers 5"
  if ! grep -q '"done"' "$RUNS/$NAME/log.jsonl" 2>/dev/null; then
    echo "$(date +%T) queue train $NAME (GPU used: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader))"
    gpu train_$NAME $PY -m model_fine_tuning.pilot22_train.train --pilot-share $SHARE -- \
      --n-real 2 --w-need 0.5 --w-whale 1.0 --maps-dir $PW/maps_tri3 -- \
      --name $NAME --out-root $RUNS --cache /var/tmp/sat3737-parts-cache/run05/labels_triA $RECIPE > $C/train_$NAME.out 2>&1 || fail "train $NAME (see $C/train_$NAME.out)"
    echo "$(date +%T) train $NAME done"
  fi
  CK=$RUNS/$NAME/checkpoints/best.pt
  [ -f "$CK" ] || fail "no best.pt for $NAME"
  rm -f "$RUNS/$NAME/checkpoints/last.pt"
  if [ ! -f "$E/${NAME}_real_metrics.json" ]; then
    echo "$(date +%T) queue predict + score $NAME"
    gpu eval_$NAME $PY -m model_fine_tuning.pilot22_train.oldroot predict --name $NAME --checkpoint $CK --metrics $E/${NAME}_real_metrics.json > $C/eval_$NAME.out 2>&1 || fail "predict/score $NAME (see $C/eval_$NAME.out)"
  fi
  [ -f /home/sat3737/NOAA_whales/moonshot_cache/parts_tri_expand/synth_val/$NAME.json ] || gpu synth_$NAME $PY experiments/parts_tri_expand/w3e_scores.py synth --model $NAME --checkpoint $CK > $C/synth_$NAME.out 2>&1 || echo "synth $NAME failed"
  [ -f $C/pilot_val/$NAME.json ] || gpu pilotval_$NAME $PY -m model_fine_tuning.pilot22_train.score_pilot --model $NAME --checkpoint $CK > $C/pilotval_$NAME.out 2>&1 || echo "pilot val $NAME failed"
  [ -f $C/part_gaps/report_${NAME}_hr.json ] || $PY -m model_fine_tuning.pilot22_train.oldroot partgap --model $NAME --variant hr --arms parts_seg_r05_w3e --out $C/part_gaps/report_${NAME}_hr.json > $C/part_gaps/report_${NAME}_hr.out 2>&1 || echo "part-gap report $NAME failed"
  echo "$(date +%T) $NAME scored"
done
date +%T > $C/p22_$ARM.done
echo "$(date +%T) arm $ARM done; marker p22_$ARM.done"
