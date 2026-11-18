#!/usr/bin/env bash
# W3b = W2's recipe (whale term on, seed 0) + the whole clicked triangle on 2025-01-02 (maps_tri), with the OLD run05
# render labels (no widening). Separates the two sources of W3's band. One seed: only the triangle question is asked.
#   setsid nohup bash experiments/parts_weak/run_w3b.sh > /var/tmp/sat3737-parts-cache/parts_weak/run_w3b.log 2>&1 &
# Marker at the end: w3b.done (trained, predicted, scored) or w3b.failed (with the failing step).
set -u
REPO=/home/sat3737/NOAA_whales/Dinov3-WhaleID
cd "$REPO"
PY="$REPO/.venv/bin/python"
LOCK="flock /var/tmp/sat3737-parts-cache/gpu_train.lock"
export PYTHONPATH="$REPO"
C=/var/tmp/sat3737-parts-cache/parts_weak
NAME=parts_seg_r05_w3b
RECIPE="--init-weights /var/tmp/sat3737-parts-cache/phase1_last.pt --epochs 10 --eval-every 5 --warmup-steps 100 --batch-size 8 --crop 512 --seed 0 --workers 5"
fail() { echo "$(date +%T) FAILED: $1"; echo "$1" > $C/w3b.failed; exit 1; }

if ! grep -q '"done"' "$C/runs/$NAME/log.jsonl" 2>/dev/null; then
  echo "$(date +%T) queue train $NAME"
  $LOCK $PY -m model_fine_tuning.parts_weak.train --n-real 2 --w-need 0.5 --w-whale 1.0 --maps-dir $C/maps_tri -- \
    --name $NAME --out-root $C/runs $RECIPE > $C/train_$NAME.out 2>&1 || fail "train (see $C/train_$NAME.out)"
  echo "$(date +%T) train $NAME done"
fi
[ -f "$C/runs/$NAME/checkpoints/best.pt" ] || fail "no best.pt"
if [ ! -f "experiments/parts_weak/${NAME}_real_metrics.json" ]; then
  echo "$(date +%T) queue predict + score $NAME"
  $LOCK $PY experiments/parts_weak/eval_w.py --name $NAME --checkpoint $C/runs/$NAME/checkpoints/best.pt \
    --metrics experiments/parts_weak/${NAME}_real_metrics.json > $C/eval_$NAME.out 2>&1 || fail "predict/score (see $C/eval_$NAME.out)"
  echo "$(date +%T) score $NAME done"
fi
O=$C/report_w3b
mkdir -p $O
M="parts_seg_r05_a parts_seg_r05_s1 parts_seg_r05_w2 parts_seg_r05_w2s1 parts_seg_r05_w3 $NAME"
$PY experiments/parts_weak/compare.py --arms parts_seg_r05_s1 parts_seg_r05_w2 parts_seg_r05_w2s1 parts_seg_r05_w3 $NAME --out experiments/parts_weak/compare_w3b.json > $O/compare.out 2>&1 || fail "compare"
$PY experiments/parts_weak/fin_definition.py --models $M --out experiments/parts_weak/fin_definition_w3b.json --sheet-name fin_definition_w3b.jpg > $O/findef.out 2>&1 || fail "fin_definition"
$PY experiments/parts_weak/shares.py --models $M --out experiments/parts_weak/shares_w3b.json > $O/shares.out 2>&1 || fail "shares"
$PY experiments/parts_weak/finless.py --models $M --out experiments/parts_weak/finless_w3b.json > $O/finless.out 2>&1 || fail "finless"
for base in parts_seg_r05_w2 parts_seg_r05_w3; do
  $PY experiments/parts_weak/detail.py --base $base --arm $NAME --out experiments/parts_weak/detail_w3b_vs_${base#parts_seg_r05_}.json > $O/detail_$base.out 2>&1 || fail "detail vs $base"
done
$PY experiments/parts_weak/fin_sheet.py --models parts_seg_r05_a parts_seg_r05_w2 parts_seg_r05_w3 $NAME --name w3b_vs_w2_fins.jpg > $O/sheet.out 2>&1 || fail "fin sheet"
$PY experiments/parts_weak/sheets.py render --name $NAME > $O/sheets.out 2>&1 || fail "before/after sheets"
date +%T > $C/w3b.done
echo "$(date +%T) $NAME trained, predicted and scored; marker w3b.done"
