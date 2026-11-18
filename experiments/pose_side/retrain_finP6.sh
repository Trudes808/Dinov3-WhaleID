#!/bin/bash
# Fair retrain test of the point-based fin cut-outs (2026-11-17, ordered by the coordinator): R0 against RP6, two seeds.
#   R0s1   = the R0 recipe of experiments/cutouts_v3/run_arm.sh with --seed 1 (R0 seed 0 = pair_student_v3_r0 exists)
#   RP6s0/1 = the same recipe and keys, fins from finP6 (pose_side/make_finP.py --margin 6.0, B_both points), bodies and
#            flukes from parts_up as in R0 (train.py --arm RP)
# Every GPU step waits for the shared lock (no timeout). Run detached:
#   setsid nohup bash experiments/pose_side/retrain_finP6.sh > /var/tmp/sat3737-parts-cache/pose_side/retrain_finP6.log 2>&1 &
# marker /var/tmp/sat3737-parts-cache/pose_side/retrain_finP6.done (written only on success; retrain_finP6.failed otherwise)
set -euo pipefail
cd /home/sat3737/NOAA_whales/Dinov3-WhaleID
PY=.venv/bin/python; export PYTHONPATH=.
LOCK=/var/tmp/sat3737-parts-cache/gpu_train.lock
C=/var/tmp/sat3737-parts-cache/pose_side
CV=/var/tmp/sat3737-parts-cache/cutouts_v3
trap 'echo "FAILED at line $LINENO $(date +%T)"; touch $C/retrain_finP6.failed' ERR
TRAIN_DAYS=$(python3 -c "import json; print(','.join('all:' + d for d in json.load(open('model_fine_tuning/splits/days.json'))['train']))")
TRAIN_LIST=$(python3 -c "import json; print(','.join(json.load(open('model_fine_tuning/splits/days.json'))['train']))")
ID=experiments/id_v4/checkpoints/best.pt

echo "== 1 points on the training days, waiting for the GPU lock $(date +%T)"
if [ ! -f $C/pred_Bboth_train.json ]; then
  flock $LOCK $PY -m model_fine_tuning.pose_side.predict --head $C/B_both/head.pt --out $C/pred_Bboth_train.json --keys $TRAIN_DAYS
fi
echo "== 2 finP6 cut-outs on the training days $(date +%T)"
$PY -m model_fine_tuning.pose_side.make_finP $C/pred_Bboth_train.json --days $TRAIN_LIST --set finP6 --margin 6.0 --workers 12

run_arm () {   # name arm seed
  local NAME=$1 ARM=$2 SEED=$3
  local COMMON="--data-root experiments/days_all --P 6 --size 512 --batches-per-epoch 300 --epochs 10 --val-batches 60 --lr-head 0.0003 --lr-backbone 2e-05 --workers 6 --seed $SEED --reference $ID --group-hard"
  local T=pair_student_v3_${NAME}_teacher S=pair_student_v3_${NAME}
  if [ ! -f experiments/$T/checkpoints/best.pt ] || ! grep -q '"done"' experiments/$T/log.jsonl 2>/dev/null; then
    echo "== $NAME teacher waiting for the GPU lock $(date +%T)"
    flock $LOCK $PY -W ignore -m model_fine_tuning.cutouts_v3.train --arm $ARM --r1-fin finP6 -- train --name $T $COMMON --dim 256 --layers 3 --unfreeze-blocks 2 --init-backbone $ID --residual-ref $ID
  fi
  if [ ! -f experiments/$S/checkpoints/best.pt ] || ! grep -q '"done"' experiments/$S/log.jsonl 2>/dev/null; then
    echo "== $NAME student waiting for the GPU lock $(date +%T)"
    flock $LOCK $PY -W ignore -m model_fine_tuning.cutouts_v3.train --arm $ARM --r1-fin finP6 -- distill --name $S $COMMON --teacher experiments/$T/checkpoints/best.pt --dim 512 --unfreeze-blocks 4 --init $ID --teacher-mix 0.3
  fi
  echo "== $NAME trained $(date +%T)"
}
run_arm r0s1 R0 1
run_arm rp6s0 RP 0
run_arm rp6s1 RP 1

echo "== 4 embeddings on the validation days, waiting for the GPU lock $(date +%T)"
flock $LOCK $PY -W ignore -m model_fine_tuning.cutouts_v3.embed --net R0s1=experiments/pair_student_v3_r0s1/checkpoints/best.pt --days val --sets up
flock $LOCK $PY -W ignore -m model_fine_tuning.cutouts_v3.embed --net RP6s0=experiments/pair_student_v3_rp6s0/checkpoints/best.pt --days val --sets up,finP6
flock $LOCK $PY -W ignore -m model_fine_tuning.cutouts_v3.embed --net RP6s1=experiments/pair_student_v3_rp6s1/checkpoints/best.pt --days val --sets up,finP6
# R0 seed 0 needs finP6 nowhere; its embeddings of 'up' on the validation days exist (emb/R0)

echo "== 5 scores $(date +%T)"
$PY -m model_fine_tuning.cutouts_v3.score --days val --out $CV/score_val_retrain_finP6 --retrain > $CV/score_val_retrain_finP6.txt 2>&1
$PY -m model_fine_tuning.pose_side.retrain_summary $CV/score_val_retrain_finP6 --out experiments/pose_side/retrain_finP6.json > $C/retrain_finP6_summary.txt
for b in G R0 R0s1; do cp $CV/score_val_retrain_finP6/score_base_$b.json experiments/pose_side/score_val_retrain_finP6_base_$b.json 2>/dev/null || true; done
cat $C/retrain_finP6_summary.txt
echo "== done $(date +%T)"
touch $C/retrain_finP6.done
