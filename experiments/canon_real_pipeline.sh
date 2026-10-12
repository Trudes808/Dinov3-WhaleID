#!/usr/bin/env bash
# Mirror-consistency fine-tune on the real photos (joint with synthetic supervision), then evaluate and predict.
set -uo pipefail
cd /home/sat3737/NOAA_whales/Dinov3-WhaleID
RUN=$HOME/NOAA_whales/Whale-Image-Data-Factory/data/renders/run02; SPLIT=model_fine_tuning/splits/run02.json
SEG=experiments/seg_r02_ft4/checkpoints/best.pt; PY=.venv/bin/python; NAME=canon_r02_ft4_real
echo "== train $NAME"; date
if ! { [ -f experiments/$NAME/log.jsonl ] && grep -q '"event": "done"' experiments/$NAME/log.jsonl; }; then
  $PY -m model_fine_tuning.canon.train --name $NAME --run-dir $RUN --split $SPLIT --init-from experiments/canon_r02_ft4/checkpoints/best.pt \
     --real-images experiments/real_2024-12-18/images --real-masks experiments/real_2024-12-18/pred_r02 --w-consistency 5 \
     --unfreeze-blocks 4 --epochs 10 --lr-head 1e-4 --lr-backbone 1e-5 --warmup-steps 50 --batch-size 8 --real-batch-size 4 --workers 6 > experiments/$NAME.stdout 2>&1
fi
CK=experiments/$NAME/checkpoints/best.pt
echo "== evaluate"; date
[ -f experiments/$NAME/eval_test/metrics.json ] || $PY -m model_fine_tuning.canon.evaluate --checkpoint $CK --seg-checkpoint $SEG --run-dir $RUN --split-file $SPLIT --split test 2>&1 | grep -v -i warn | tail -3
echo "== real photos"; date
[ -f experiments/real_2024-12-18/$NAME/points.jsonl ] || $PY -m model_fine_tuning.canon.predict --seg-checkpoint $SEG --checkpoint $CK --images experiments/real_2024-12-18/images --out experiments/real_2024-12-18/$NAME --max-side 1280 > experiments/real_2024-12-18/$NAME.log 2>&1
echo "== compare"; date
$PY test_scripts/python/compare_real_before_after.py experiments/real_2024-12-18/side_diag/real_side.json experiments/real_2024-12-18/$NAME experiments/real_2024-12-18/${NAME}_compare
echo "== done"; date
