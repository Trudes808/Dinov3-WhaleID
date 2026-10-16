#!/usr/bin/env bash
# Experiment 3b (2026-10-16): the canon model trained on run03 + run04 together (flat K9239 poses and the rigged whale),
# plain recipe; waits for the GPU (the real-photo fine-tune) to finish first; then the run04 synthetic test, the real
# photos (predict, both outline metrics, bend fit).
set -uo pipefail
cd "$(dirname "$0")/.."
F=$HOME/NOAA_whales/Whale-Image-Data-Factory/data/renders; PY=.venv/bin/python; NAME="${R0304_NAME:-canon_r0304_ft4}"
SEG=experiments/seg_r04_ft4/checkpoints/best.pt; REAL=experiments/real_2024-12-18; export WHALE_RUN_DIR="$F/run04"
until [ -f experiments/canon_r04_ft4_real/log.jsonl ] && grep -q '"event": "done"' experiments/canon_r04_ft4_real/log.jsonl; do sleep 120; done
echo "== train $NAME"; date
if ! { [ -f experiments/$NAME/log.jsonl ] && grep -q '"event": "done"' experiments/$NAME/log.jsonl; }; then
  $PY -m model_fine_tuning.canon.train --name $NAME --run-dir $F/run04 --split model_fine_tuning/splits/run04.json \
     --extra-run-dir $F/run03 --extra-split model_fine_tuning/splits/run03.json \
     --unfreeze-blocks 4 --epochs "${R0304_EPOCHS:-30}" --batch-size 8 --workers 6 --min-visible 300 > experiments/$NAME.stdout 2>&1
fi
CK=experiments/$NAME/checkpoints/best.pt
echo "== run04 synthetic test"; date
[ -f experiments/$NAME/eval_test/metrics.json ] || $PY -m model_fine_tuning.canon.evaluate --checkpoint $CK --seg-checkpoint $SEG --run-dir $F/run04 --split-file model_fine_tuning/splits/run04.json --split test > experiments/$NAME.eval_test.stdout 2>&1
echo "== real photos"; date
[ -f $REAL/canon_$NAME/points.jsonl ] || $PY -m model_fine_tuning.canon.predict --seg-checkpoint $SEG --checkpoint $CK --images $REAL/images --out $REAL/canon_$NAME --max-side 1280 > $REAL/canon_$NAME.log 2>&1
[ -f $REAL/outline_metric_$NAME.json ] || $PY -m model_fine_tuning.canon.outline_metric --canon-dir $REAL/canon_$NAME --images $REAL/images --timestamps $REAL/timestamps.json --out $REAL/outline_metric_$NAME.json > $REAL/outline_metric_$NAME.log 2>&1
[ -f $REAL/outline_metric_${NAME}_direct.json ] || $PY -m model_fine_tuning.canon.outline_metric --canon-dir $REAL/canon_$NAME --rotation direct --images $REAL/images --timestamps $REAL/timestamps.json --out $REAL/outline_metric_${NAME}_direct.json > $REAL/outline_metric_${NAME}_direct.log 2>&1
[ -f $REAL/fit_$NAME/fit.json ] || $PY -m model_fine_tuning.canon.fit_mesh --canon-dir $REAL/canon_$NAME --images $REAL/images --timestamps $REAL/timestamps.json --out $REAL/fit_$NAME --overlays 40 --workers 8 > $REAL/fit_$NAME.log 2>&1
echo "== done"; date
