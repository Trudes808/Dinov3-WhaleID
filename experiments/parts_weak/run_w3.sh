#!/usr/bin/env bash
# W3 = W2's recipe (whale term on, same seed) with the NEW dorsal definition (fin + hump; user's decision 2026-11-17):
# widened render labels (parts_weak/labels_wide.py -> run05/labels_wide) and the whole clicked triangle on 2025-01-02
# (parts_weak/build_tri.py -> maps_tri). Queued after W2 and after w2b (if that runs). Marker w3.done = trained AND scored.
#   setsid nohup bash experiments/parts_weak/run_w3.sh > /var/tmp/sat3737-parts-cache/parts_weak/run_w3.log 2>&1 &
set -u
REPO=/home/sat3737/NOAA_whales/Dinov3-WhaleID
cd "$REPO"
PY="$REPO/.venv/bin/python"
LOCK="flock /var/tmp/sat3737-parts-cache/gpu_train.lock"
export PYTHONPATH="$REPO"
C=/var/tmp/sat3737-parts-cache/parts_weak
NAME=parts_seg_r05_w3
RECIPE="--init-weights /var/tmp/sat3737-parts-cache/phase1_last.pt --epochs 10 --eval-every 5 --warmup-steps 100 --batch-size 8 --crop 512 --seed 0 --workers 5"

until grep -q '"seconds"' /var/tmp/sat3737-parts-cache/parts_weak/labels_wide.log 2>/dev/null && [ -f $C/maps_tri_summary.json ]; do sleep 60; done
echo "$(date +%T) labels ready; waiting for w2.done and w2b.done / w2b.skipped"
until [ -f $C/w2.done ] && { [ -f $C/w2b.done ] || [ -f $C/w2b.skipped ]; }; do sleep 60; done

if ! grep -q '"done"' "$C/runs/$NAME/log.jsonl" 2>/dev/null; then
  echo "$(date +%T) queue train $NAME"
  $LOCK $PY -m model_fine_tuning.parts_weak.train --n-real 2 --w-need 0.5 --w-whale 1.0 --maps-dir $C/maps_tri -- \
    --name $NAME --out-root $C/runs --cache /var/tmp/sat3737-parts-cache/run05/labels_wide $RECIPE > $C/train_$NAME.out 2>&1
  echo "$(date +%T) train $NAME exit $?"
fi
[ -f "$C/runs/$NAME/checkpoints/best.pt" ] || { echo "$NAME has no best.pt"; exit 1; }
if [ ! -f "experiments/parts_weak/${NAME}_real_metrics.json" ]; then
  echo "$(date +%T) queue score $NAME"
  $LOCK $PY experiments/parts_weak/eval_w.py --name $NAME --checkpoint $C/runs/$NAME/checkpoints/best.pt \
    --metrics experiments/parts_weak/${NAME}_real_metrics.json > $C/eval_$NAME.out 2>&1
  echo "$(date +%T) score $NAME exit $?"
fi
ARMS="parts_seg_r05_w1 parts_seg_r05_w2"
[ -f $C/w2b.done ] && ARMS="$ARMS parts_seg_r05_w2b"
$PY experiments/parts_weak/compare.py --arms $ARMS $NAME --out experiments/parts_weak/compare_$NAME.json > $C/compare_$NAME.out 2>&1
$PY experiments/parts_weak/detail.py --arm $NAME --out experiments/parts_weak/detail_$NAME.json > $C/detail_$NAME.out 2>&1
$PY experiments/parts_weak/fin_definition.py --models parts_seg_r05_a $ARMS $NAME --out experiments/parts_weak/fin_definition_w3.json \
  --sheet-name fin_definition_w3.jpg > $C/fin_definition_w3.out 2>&1
$PY experiments/parts_weak/sheets.py render --name $NAME > $C/sheets_$NAME.out 2>&1
date +%T > $C/w3.done
echo "$(date +%T) $NAME scored, marker w3.done"
