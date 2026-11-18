#!/usr/bin/env bash
# W2 with seed 1 (parts_seg_r05_w2s1): W2's recipe and flags exactly, only --seed 1. Queued after W3 is trained and
# scored (w3.done). Marker w2s1.done when trained AND scored; then the numbers for the final report (report_final.done).
#   setsid nohup bash experiments/parts_weak/run_w2s1.sh > /var/tmp/sat3737-parts-cache/parts_weak/run_w2s1.log 2>&1 &
set -u
REPO=/home/sat3737/NOAA_whales/Dinov3-WhaleID
cd "$REPO"
PY="$REPO/.venv/bin/python"
LOCK="flock /var/tmp/sat3737-parts-cache/gpu_train.lock"
export PYTHONPATH="$REPO"
C=/var/tmp/sat3737-parts-cache/parts_weak
NAME=parts_seg_r05_w2s1
RECIPE="--init-weights /var/tmp/sat3737-parts-cache/phase1_last.pt --epochs 10 --eval-every 5 --warmup-steps 100 --batch-size 8 --crop 512 --seed 1 --workers 5"

echo "$(date +%T) waiting for w3.done"
until [ -f $C/w3.done ]; do sleep 60; done
if ! grep -q '"done"' "$C/runs/$NAME/log.jsonl" 2>/dev/null; then
  echo "$(date +%T) queue train $NAME"
  $LOCK $PY -m model_fine_tuning.parts_weak.train --n-real 2 --w-need 0.5 --w-whale 1.0 -- --name $NAME --out-root $C/runs $RECIPE > $C/train_$NAME.out 2>&1
  echo "$(date +%T) train $NAME exit $?"
fi
[ -f "$C/runs/$NAME/checkpoints/best.pt" ] || { echo "$NAME has no best.pt"; exit 1; }
if [ ! -f "experiments/parts_weak/${NAME}_real_metrics.json" ]; then
  echo "$(date +%T) queue score $NAME"
  $LOCK $PY experiments/parts_weak/eval_w.py --name $NAME --checkpoint $C/runs/$NAME/checkpoints/best.pt \
    --metrics experiments/parts_weak/${NAME}_real_metrics.json > $C/eval_$NAME.out 2>&1
  echo "$(date +%T) score $NAME exit $?"
fi
date +%T > $C/w2s1.done
echo "$(date +%T) $NAME scored, marker w2s1.done"

# numbers for the final report (CPU)
O=$C/report_final
mkdir -p $O
M="parts_seg_r05_a parts_seg_r05_s1 parts_seg_r05_w2 parts_seg_r05_w2s1 parts_seg_r05_w3"
$PY experiments/parts_weak/compare.py --arms parts_seg_r05_s1 parts_seg_r05_w2 parts_seg_r05_w2s1 parts_seg_r05_w3 --out experiments/parts_weak/compare_final.json > $O/compare.out 2>&1
for base in parts_seg_r05_a parts_seg_r05_s1 parts_seg_r05_w2; do
  for arm in parts_seg_r05_w2s1 parts_seg_r05_w3; do
    $PY experiments/parts_weak/detail.py --base $base --arm $arm --out experiments/parts_weak/detail_${arm#parts_seg_r05_}_vs_${base#parts_seg_r05_}.json > $O/detail_${arm}_${base}.out 2>&1
  done
done
$PY experiments/parts_weak/fin_definition.py --models $M --out experiments/parts_weak/fin_definition_final.json --sheet-name fin_definition_final.jpg > $O/findef.out 2>&1
$PY experiments/parts_weak/shares.py --models $M --out experiments/parts_weak/shares_final.json > $O/shares.out 2>&1
$PY experiments/parts_weak/finless.py --models $M --out experiments/parts_weak/finless_final.json > $O/finless.out 2>&1
$PY experiments/parts_weak/sheets.py render --name $NAME > $O/sheets.out 2>&1
date +%T > $C/report_final.done
echo "$(date +%T) final numbers written"
