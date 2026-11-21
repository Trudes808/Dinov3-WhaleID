#!/usr/bin/env bash
# NOT RUN. Retrain test of the W3c-centred fin cut-outs, in the style of experiments/cutouts_v3/run_arm.sh and
# experiments/pose_side/retrain_finP6.sh: teacher + student (the pair_teacher_g -> pair_student_g recipe) with fins from
# a W3c set (bodies and flukes from parts_up, as R0), seeds 0 and 1; compared with R0 s0 / s1 (pair_student_v3_r0,
# pair_student_v3_r0s1, both exist). About 90 min of GPU per seed (teacher ~50, student ~40) + a few minutes to embed.
#   setsid nohup bash experiments/cutouts_w3c/run_retrain.sh s0_wide > /var/tmp/sat3737-parts-cache/cutouts_w3c/retrain.log 2>&1 &
# Needs the training-day cut-outs of that set (make.py --seed s0 on train,val; done 2026-11-18).
set -euo pipefail
cd "$(dirname "$0")/../.."
FIN=${1:-s0_wide}
PY=.venv/bin/python; export PYTHONPATH=.
LOCK=/var/tmp/sat3737-parts-cache/gpu_train.lock
C=/var/tmp/sat3737-parts-cache/cutouts_w3c
ID=experiments/id_v4/checkpoints/best.pt
trap 'echo "FAILED at line $LINENO $(date +%T)"; touch $C/retrain_$FIN.failed' ERR
for SEED in 0 1; do
  COMMON="--data-root experiments/days_all --P 6 --size 512 --batches-per-epoch 300 --epochs 10 --val-batches 60 --lr-head 0.0003 --lr-backbone 2e-05 --workers 6 --seed $SEED --reference $ID --group-hard"
  T=pair_student_w3c_${FIN}_s${SEED}_teacher; S=pair_student_w3c_${FIN}_s${SEED}
  if [ ! -f experiments/$T/checkpoints/best.pt ] || ! grep -q '"done"' experiments/$T/log.jsonl 2>/dev/null; then
    echo "== teacher seed $SEED waiting for the GPU lock $(date +%T)"
    flock $LOCK $PY -W ignore -m model_fine_tuning.cutouts_w3c.train --fin $FIN -- train --name $T $COMMON --dim 256 --layers 3 --unfreeze-blocks 2 --init-backbone $ID --residual-ref $ID
  fi
  if [ ! -f experiments/$S/checkpoints/best.pt ] || ! grep -q '"done"' experiments/$S/log.jsonl 2>/dev/null; then
    echo "== student seed $SEED waiting for the GPU lock $(date +%T)"
    flock $LOCK $PY -W ignore -m model_fine_tuning.cutouts_w3c.train --fin $FIN -- distill --name $S $COMMON --teacher experiments/$T/checkpoints/best.pt --dim 512 --unfreeze-blocks 4 --init $ID --teacher-mix 0.3
  fi
  echo "== embed seed $SEED waiting for the GPU lock $(date +%T)"
  flock $LOCK $PY -W ignore -m model_fine_tuning.cutouts_w3c.embed --net RWs$SEED=experiments/$S/checkpoints/best.pt --days val --sets up,$FIN
done
flock $LOCK $PY -W ignore -m model_fine_tuning.cutouts_w3c.embed --net R0s1=experiments/pair_student_v3_r0s1/checkpoints/best.pt --days val --sets up
$PY -m model_fine_tuning.cutouts_w3c.score --days val --out $C/score_val_retrain_$FIN --retrain $FIN > $C/score_val_retrain_$FIN.txt 2>&1
for b in G R0 R0s1; do cp $C/score_val_retrain_$FIN/score_base_$b.json experiments/cutouts_w3c/score_val_retrain_${FIN}_base_$b.json; done
echo "RETRAIN_DONE $(date +%T)"; touch $C/retrain_$FIN.done
