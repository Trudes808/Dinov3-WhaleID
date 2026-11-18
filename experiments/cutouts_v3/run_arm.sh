#!/usr/bin/env bash
# One retraining arm of the cut-outs v3 test: the pair_teacher_g -> pair_student_g recipe (cluster/pairnet.py, the
# arguments of experiments/pair_teacher_g/config.json and experiments/pair_student_g/config.json) on other cut-outs
# (model_fine_tuning/cutouts_v3/train.py). Each training waits for the shared GPU lock.
#   experiments/cutouts_v3/run_arm.sh R0|R1 [finB|finC]
set -euo pipefail
cd "$(dirname "$0")/../.."
ARM=$1; FIN=${2:-finB}; L=$(echo "$ARM" | tr 'A-Z' 'a-z')
PY=.venv/bin/python; export PYTHONPATH=.
LOCK=/var/tmp/sat3737-parts-cache/gpu_train.lock
ID=experiments/id_v4/checkpoints/best.pt
COMMON="--data-root experiments/days_all --P 6 --size 512 --batches-per-epoch 300 --epochs 10 --val-batches 60 --lr-head 0.0003 --lr-backbone 2e-05 --workers 6 --seed 0 --reference $ID --group-hard"
T=pair_student_v3_${L}_teacher; S=pair_student_v3_${L}
if [ ! -f experiments/$T/checkpoints/best.pt ] || ! grep -q '"done"' experiments/$T/log.jsonl 2>/dev/null; then
  echo "== $ARM teacher waiting for the GPU lock $(date +%T)"
  flock $LOCK $PY -W ignore -m model_fine_tuning.cutouts_v3.train --arm $ARM --r1-fin $FIN -- train --name $T $COMMON --dim 256 --layers 3 --unfreeze-blocks 2 --init-backbone $ID --residual-ref $ID
fi
echo "== $ARM student waiting for the GPU lock $(date +%T)"
flock $LOCK $PY -W ignore -m model_fine_tuning.cutouts_v3.train --arm $ARM --r1-fin $FIN -- distill --name $S $COMMON --teacher experiments/$T/checkpoints/best.pt --dim 512 --unfreeze-blocks 4 --init $ID --teacher-mix 0.3
echo "== $ARM embed waiting for the GPU lock $(date +%T)"
flock $LOCK $PY -W ignore -m model_fine_tuning.cutouts_v3.embed --net $ARM=experiments/$S/checkpoints/best.pt --days train,val --sets up,fluke,finB,finC
echo "ARM_DONE $ARM $(date +%T)"
