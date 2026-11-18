#!/bin/bash
# student_curve training chain (2026-11-18): the id_v4 -> pair_teacher -> pair_student (R0, parts_up) lineage retrained
# from DINOv3 on the first N days of bench.splits()[0]'s pool, per seed. Existing code, unchanged; only the day lists,
# seed, output folder, a fixed schedule instead of val-day selection, and no --reference (val-time reporting only).
#   stage id      = model_fine_tuning.cluster.train_id with id_v4's arguments (parts/, 512 px, P8 K4, 10 x 300 steps,
#                   ArcFace + SupCon, group-hard), --val-days none (no val read), last.pt used
#   stage teacher = cutouts_v3.train --arm R0 -- train (pair_teacher recipe on parts_up), init + residual ref = stage id,
#                   --max-steps 300 of the 10-epoch schedule = the epoch-0 checkpoint that val selected in 4 of the 6
#                   existing teachers; --val-days <one training day> --val-batches 2 (in-sample, ignored)
#   stage student = cutouts_v3.train --arm R0 -- distill (10 x 300 steps, teacher mix 0.3, init stage id), last.pt
#                   (val selected the last epoch in all 6 existing students)
#   stage embed   = fin / fluke fingerprints of the 8 test days (bench.VAL + hold) with the student and the id stage
# One flock per run (size, seed): the four stages of a run (about 80 min of GPU) run under one lock, others get turns
# between runs (the 60-step test waited 45-60 min for the lock at each of its four stages). Resumable: a stage or run
# with its marker is skipped.
#   setsid nohup bash experiments/moonshot_sizing/student_curve/chain.sh s5_0 s16_0 s5_1 s16_1 > $C/logs/chain.log 2>&1 &
# Run name s<N>_<seed>. Env MAXSTEPS=k makes a timing test (every stage capped at k steps, names prefixed test_).
set -uo pipefail
cd /home/sat3737/NOAA_whales/Dinov3-WhaleID
PY=.venv/bin/python; export PYTHONPATH=.
LOCK=/var/tmp/sat3737-parts-cache/gpu_train.lock
C=/home/sat3737/NOAA_whales/moonshot_cache/student_curve
R=$C/runs; M=$C/markers; L=$C/logs
mkdir -p $R $M $L
[ "${INLOCK:-}" != 1 ] && echo $$ > $C/chain.pid
TESTDAYS=$($PY -c "import sys; sys.path.insert(0,'experiments/moonshot_sizing'); import bench; h,p=bench.splits()[0]; print(','.join(bench.VAL+h))")
TEST=${MAXSTEPS:+test_}
CAP_ID=${MAXSTEPS:+--max-steps $MAXSTEPS}
CAP_T=--max-steps\ ${MAXSTEPS:-300}
CAP_S=${MAXSTEPS:+--max-steps $MAXSTEPS}

stage () {   # marker logfile command...   (runs inside the run's lock)
  local mk=$M/$1 log=$2; shift 2
  [ -f $mk.done ] && { echo "skip $(basename $mk)"; return 0; }
  local t0=$(date +%s); echo "== $(basename $mk) $(date '+%F %T')"
  "$@" > $log 2>&1; local rc=$?
  echo "   seconds $(( $(date +%s) - t0 ))"
  if [ $rc -ne 0 ]; then echo "FAILED $(basename $mk) rc $rc $(date +%T)"; touch $mk.failed; return $rc; fi
  touch $mk.done
}

if [ "${INLOCK:-}" != 1 ]; then          # outer chain: one flock per run
  for run in "$@"; do
    NAME=$TEST$run
    [ -f $M/$NAME.done ] && { echo "skip run $NAME"; continue; }
    echo "#### $NAME waiting for the lock $(date '+%F %T')"
    INLOCK=1 flock $LOCK bash "$0" "$run" || { echo "#### $NAME FAILED $(date '+%F %T')"; exit 1; }
  done
  echo "CHAIN_DONE $(date '+%F %T')"; touch $M/chain_${TEST}done; exit 0
fi

for run in "$@"; do
  N=${run#s}; N=${N%_*}; SEED=${run#*_}; NAME=$TEST$run
  DAYS=$($PY -c "import sys; sys.path.insert(0,'experiments/moonshot_sizing'); import bench; h,p=bench.splits()[0]; print(' '.join(p[:$N]))")
  VDAY=${DAYS%% *}
  T0=$(date +%s); echo "#### $NAME lock taken $(date '+%F %T'): $N days, seed $SEED: $DAYS"
  ID=$R/${NAME}_id/checkpoints/last.pt; T=$R/${NAME}_teacher/checkpoints/last.pt; S=$R/${NAME}_student/checkpoints/last.pt
  stage ${NAME}_id $L/${NAME}_id.log $PY -W ignore -m model_fine_tuning.cluster.train_id --name ${NAME}_id --out-root $R --data-root experiments/days_all \
      --train-days $DAYS --val-days none --size 512 --dim 512 --unfreeze-blocks 4 --P 8 --K 4 --batches-per-epoch 300 --epochs 10 \
      --lr-head 0.0003 --lr-backbone 2e-05 --w-arc 1.0 --w-supcon 1.0 --arc-margin 0.4 --workers 6 --seed $SEED --group-hard --mix-prob 0.3 $CAP_ID || exit 1
  COMMON="--data-root experiments/days_all --out-root $R --train-days $DAYS --val-days $VDAY --P 6 --size 512 --batches-per-epoch 300 --epochs 10 --val-batches 2 --lr-head 0.0003 --lr-backbone 2e-05 --workers 6 --seed $SEED --group-hard"
  stage ${NAME}_teacher $L/${NAME}_teacher.log $PY -W ignore -m model_fine_tuning.cutouts_v3.train --arm R0 -- train --name ${NAME}_teacher $COMMON $CAP_T \
      --dim 256 --layers 3 --unfreeze-blocks 2 --init-backbone $ID --residual-ref $ID || exit 1
  stage ${NAME}_student $L/${NAME}_student.log $PY -W ignore -m model_fine_tuning.cutouts_v3.train --arm R0 -- distill --name ${NAME}_student $COMMON $CAP_S \
      --teacher $T --dim 512 --unfreeze-blocks 4 --init $ID --teacher-mix 0.3 || exit 1
  stage ${NAME}_embed $L/${NAME}_embed.log bash -c "$PY -W ignore experiments/moonshot_sizing/student_curve/embed.py --net ${NAME}=$S --days $TESTDAYS && $PY -W ignore experiments/moonshot_sizing/student_curve/embed.py --net ${NAME}_idstage=$ID --days $TESTDAYS" || exit 1
  # keep only the student's checkpoint (the id stage is embedded already; the teacher is only a distillation source)
  rm -f $R/${NAME}_id/checkpoints/*.pt $R/${NAME}_teacher/checkpoints/*.pt $R/${NAME}_student/checkpoints/best.pt
  touch $M/${NAME}.done; echo "#### $NAME DONE (lock held $(( $(date +%s) - T0 )) s) $(date '+%F %T')"
done
