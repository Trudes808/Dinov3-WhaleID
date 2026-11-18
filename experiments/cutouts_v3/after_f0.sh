#!/usr/bin/env bash
# Unattended rest of the cut-outs v3 test: wait for the frozen-net (G) fingerprints of all 24 days, score F0 on the
# validation days, take the fin variant with the higher "first" (B on a tie) for R1, run the R1 arm, wait for R0 and
# R1 to be embedded, then score val (result) and train (in-sample).
set -uo pipefail
cd "$(dirname "$0")/../.."
C=/var/tmp/sat3737-parts-cache/cutouts_v3; PY=.venv/bin/python; export PYTHONPATH=.
until grep -q "^G 2025-04-11" $C/embed_G.log 2>/dev/null; do sleep 60; done
$PY -m model_fine_tuning.cutouts_v3.score --days val --out $C/score_val_F0 > $C/score_val_F0.txt 2>&1
FIN=$($PY -c "import json;r=json.load(open('$C/score_val_F0/score_base_G.json'))['fin'];print('finC' if r['G.C']['first']>r['G.B']['first'] else 'finB')")
echo "R1 fin variant chosen by F0 on the validation days: $FIN $(date +%T)"
experiments/cutouts_v3/run_arm.sh R1 $FIN > $C/arm_R1.log 2>&1
until grep -q ARM_DONE $C/arm_R0.log 2>/dev/null; do sleep 60; done
$PY -m model_fine_tuning.cutouts_v3.score --days val --out $C/score_val --r1-fin $FIN > $C/score_val.txt 2>&1
$PY -m model_fine_tuning.cutouts_v3.score --days train --out $C/score_train --r1-fin $FIN > $C/score_train.txt 2>&1
cp $C/score_val/score_base_*.json $C/score_val_F0/score_base_G.json experiments/cutouts_v3/ 2>/dev/null
echo "ALL_DONE $(date +%T)"
