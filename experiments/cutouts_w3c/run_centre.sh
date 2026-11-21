#!/usr/bin/env bash
# Follow-up 2026-11-18 pm: the centre alone (centre.py variants "centre" and "nudge", parts_up's scale), frozen G and R0.
#   setsid nohup bash experiments/cutouts_w3c/run_centre.sh > /var/tmp/sat3737-parts-cache/cutouts_w3c/run_centre.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python; export PYTHONPATH=.
LOCK=/var/tmp/sat3737-parts-cache/gpu_train.lock
C=/var/tmp/sat3737-parts-cache/cutouts_w3c
for NET in G=experiments/pair_student_g/checkpoints/best.pt R0=experiments/pair_student_v3_r0/checkpoints/best.pt; do
  echo "== embed ${NET%%=*} waiting for the GPU lock $(date +%T)"
  flock $LOCK $PY -W ignore -m model_fine_tuning.cutouts_w3c.embed --net $NET --days val --sets s0_centre,s0_nudge,s1_centre,s1_nudge
done
$PY -m model_fine_tuning.cutouts_w3c.score --days val --out $C/score_val_centre --arms C=s0_centre,N=s0_nudge,C1=s1_centre,N1=s1_nudge > $C/score_val_centre.txt 2>&1
for b in G R0; do cp $C/score_val_centre/score_base_$b.json experiments/cutouts_w3c/score_val_centre_base_$b.json; done
echo "CENTRE_DONE $(date +%T)"
