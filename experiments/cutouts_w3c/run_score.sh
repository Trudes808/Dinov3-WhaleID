#!/usr/bin/env bash
# Frozen-network score of the W3c-centred fin cut-outs on the validation days (no training).
#   experiments/cutouts_w3c/run_score.sh [EXTRA_SET]      EXTRA_SET e.g. s1_wide (seed 1's regions, cut by make.py --seed s1)
# Embeds parts_up (all parts, re-embedded so the baseline is reproduced) and the W3c sets with pair_student_g (G) and
# the retrained student R0 (pair_student_v3_r0), each under the shared GPU lock, then scores (score.py).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python; export PYTHONPATH=.
LOCK=/var/tmp/sat3737-parts-cache/gpu_train.lock
C=/var/tmp/sat3737-parts-cache/cutouts_w3c
EXTRA=${1:-}
SETS=up,s0_wide,s0_tight,s1_wide,s1_tight; [ -n "$EXTRA" ] && SETS=$EXTRA
for d in 2025-01-15 2025-02-27 2025-03-14; do until [ -f $C/$d/meta_s0.jsonl ]; do sleep 30; done; done
for NET in G=experiments/pair_student_g/checkpoints/best.pt R0=experiments/pair_student_v3_r0/checkpoints/best.pt; do
  echo "== embed ${NET%%=*} $SETS waiting for the GPU lock $(date +%T)"
  flock $LOCK $PY -W ignore -m model_fine_tuning.cutouts_w3c.embed --net $NET --days val --sets $SETS
done
OUT=$C/score_val${EXTRA:+_$EXTRA}
$PY -m model_fine_tuning.cutouts_w3c.score --days val --out $OUT ${EXTRA:+--extra $EXTRA} > $OUT.txt 2>&1
cp $OUT/score_base_G.json experiments/cutouts_w3c/$(basename $OUT)_base_G.json
cp $OUT/score_base_R0.json experiments/cutouts_w3c/$(basename $OUT)_base_R0.json
echo "SCORE_DONE $(date +%T)"
