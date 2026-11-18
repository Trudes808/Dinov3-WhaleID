#!/bin/bash
# point-based fin cut-outs at B's and A's scale (finP4, finP6): embed with G and R0 (GPU lock, a few seconds each), then
# score on the validation days with the cut-outs v3 scripts. Marker /var/tmp/sat3737-parts-cache/pose_side/finP46.done
cd /home/sat3737/NOAA_whales/Dinov3-WhaleID
export PYTHONPATH=.
PY=.venv/bin/python
L=/var/tmp/sat3737-parts-cache/gpu_train.lock
C=/var/tmp/sat3737-parts-cache/cutouts_v3
flock $L $PY -W ignore -m model_fine_tuning.cutouts_v3.embed --net G=experiments/pair_student_g/checkpoints/best.pt --days val --sets finP4,finP6 || exit 1
flock $L $PY -W ignore -m model_fine_tuning.cutouts_v3.embed --net R0=experiments/pair_student_v3_r0/checkpoints/best.pt --days val --sets finP4,finP6 || exit 1
$PY -m model_fine_tuning.cutouts_v3.score --days val --out $C/score_val_finP46 --finp fallback > $C/score_val_finP46_fallback.txt 2>&1
touch /var/tmp/sat3737-parts-cache/pose_side/finP46.done
