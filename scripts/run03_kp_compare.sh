#!/usr/bin/env bash
# Compare the run03 canon model with the keypoint/state/deformation heads (canon_r03_kp) against the
# plain run03 retrain (canon_r03_ft4): synthetic test split by category, real photos (prediction,
# outline metric with the dense map alone and with the detected keypoints added to the pose fit,
# side-flip rate). Waits for the training to finish. Idempotent per stage.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; export PYTHONPATH=.
RUN=/home/sat3737/NOAA_whales/Whale-Image-Data-Factory/data/renders/run03
SPLIT=model_fine_tuning/splits/run03.json
SEG=experiments/seg_r03_ft4/checkpoints/best.pt
R=experiments/real_2024-12-18
export WHALE_RUN_DIR=$RUN
until grep -q '"event": "done"' experiments/canon_r03_kp/log.jsonl 2>/dev/null; do sleep 120; done
for NAME in canon_r03_ft4 canon_r03_kp; do
  CK=experiments/$NAME/checkpoints/best.pt
  echo "== synthetic test, by category: $NAME"; date
  [ -f experiments/$NAME/eval_test_kp/metrics.json ] || $PY -m model_fine_tuning.canon.evaluate --checkpoint $CK --seg-checkpoint $SEG --run-dir $RUN --split-file $SPLIT --split test --out experiments/$NAME/eval_test_kp > experiments/$NAME.eval_kp.stdout 2>&1
done
NAME=canon_r03_kp; CK=experiments/$NAME/checkpoints/best.pt
echo "== real photos: predict $NAME"; date
[ -f $R/canon_$NAME/points.jsonl ] || $PY -m model_fine_tuning.canon.predict --seg-checkpoint $SEG --checkpoint $CK --images $R/images --out $R/canon_$NAME --max-side 1280 > $R/canon_$NAME.log 2>&1
echo "== outline metric: dense map only / dense map + keypoints"; date
[ -f $R/outline_metric_$NAME.json ] || $PY -m model_fine_tuning.canon.outline_metric --canon-dir $R/canon_$NAME --images $R/images --timestamps $R/timestamps.json --out $R/outline_metric_$NAME.json > $R/outline_metric_$NAME.log 2>&1
[ -f $R/outline_metric_${NAME}_kp.json ] || $PY -m model_fine_tuning.canon.outline_metric --canon-dir $R/canon_$NAME --images $R/images --timestamps $R/timestamps.json --kp-weight 30 --out $R/outline_metric_${NAME}_kp.json > $R/outline_metric_${NAME}_kp.log 2>&1
echo "== side-flip rate"; date
$PY test_scripts/python/compare_real_before_after.py $R/side_diag/real_side.json $R/canon_$NAME $R/compare_$NAME > $R/compare_$NAME.log 2>&1 || true
echo "== report"; date
$PY scripts/run03_kp_report.py --out experiments/run03_kp_report.md
echo "== done"; date
