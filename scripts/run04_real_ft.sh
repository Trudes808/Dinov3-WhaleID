#!/usr/bin/env bash
# Experiment 3a (2026-10-16): fine-tune the run04 canon model with the mirror-consistency loss on the real photos
# (EMA teacher, stop-gradient), joint with the synthetic supervision; then the synthetic test split, the real-photo
# predictions and both outline metrics, so it slots into run04_report-style comparisons.
#   scripts/run04_real_ft.sh            (env: R04_REAL_NAME canon_r04_ft4_real, R04_REAL_EPOCHS 6, R04_REAL_W 1.0)
set -uo pipefail
cd "$(dirname "$0")/.."
RUN=$HOME/NOAA_whales/Whale-Image-Data-Factory/data/renders/run04; SPLIT=model_fine_tuning/splits/run04.json
SEG=experiments/seg_r04_ft4/checkpoints/best.pt; PY=.venv/bin/python; NAME="${R04_REAL_NAME:-canon_r04_ft4_real}"
REAL=experiments/real_2024-12-18; export WHALE_RUN_DIR="$RUN"
echo "== train $NAME"; date
if ! { [ -f experiments/$NAME/log.jsonl ] && grep -q '"event": "done"' experiments/$NAME/log.jsonl; }; then
  $PY -m model_fine_tuning.canon.train --name $NAME --run-dir $RUN --split $SPLIT --init-from experiments/canon_r04_ft4/checkpoints/best.pt \
     --real-images $REAL/images --real-masks $REAL/canon_canon_r04_ft4 --w-consistency "${R04_REAL_W:-1.0}" --consistency-teacher ema \
     --unfreeze-blocks 4 --epochs "${R04_REAL_EPOCHS:-6}" --lr-head 1e-4 --lr-backbone 1e-5 --warmup-steps 50 --batch-size 8 --real-batch-size 4 --workers 6 \
     > experiments/$NAME.stdout 2>&1
fi
CK=experiments/$NAME/checkpoints/best.pt
echo "== synthetic test"; date
[ -f experiments/$NAME/eval_test/metrics.json ] || $PY -m model_fine_tuning.canon.evaluate --checkpoint $CK --seg-checkpoint $SEG --run-dir $RUN --split-file $SPLIT --split test > experiments/$NAME.eval_test.stdout 2>&1
echo "== real photos"; date
[ -f $REAL/canon_$NAME/points.jsonl ] || $PY -m model_fine_tuning.canon.predict --seg-checkpoint $SEG --checkpoint $CK --images $REAL/images --out $REAL/canon_$NAME --max-side 1280 > $REAL/canon_$NAME.log 2>&1
[ -f $REAL/outline_metric_$NAME.json ] || $PY -m model_fine_tuning.canon.outline_metric --canon-dir $REAL/canon_$NAME --images $REAL/images --timestamps $REAL/timestamps.json --out $REAL/outline_metric_$NAME.json > $REAL/outline_metric_$NAME.log 2>&1
[ -f $REAL/outline_metric_${NAME}_direct.json ] || $PY -m model_fine_tuning.canon.outline_metric --canon-dir $REAL/canon_$NAME --rotation direct --images $REAL/images --timestamps $REAL/timestamps.json --out $REAL/outline_metric_${NAME}_direct.json > $REAL/outline_metric_${NAME}_direct.log 2>&1
[ -f $REAL/compare_$NAME/summary.json ] || $PY test_scripts/python/compare_real_before_after.py $REAL/side_diag/real_side.json $REAL/canon_$NAME $REAL/compare_$NAME > $REAL/compare_$NAME.log 2>&1 || true
echo "== done"; date
