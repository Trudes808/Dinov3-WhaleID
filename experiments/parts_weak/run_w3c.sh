#!/usr/bin/env bash
# W3c = W3b with a TWO-SIDED real label: the clicked triangle = dorsal AND the clicked whale below the base line / beyond the
# bases = not dorsal (build_tri2.py -> maps_tri2, value 6, loss "not dorsal"). OLD render labels, whale term on, seed 0.
#   setsid nohup bash experiments/parts_weak/run_w3c.sh > /var/tmp/sat3737-parts-cache/parts_weak/run_w3c.log 2>&1 &
# Marker at the end: w3c.done (trained, predicted, scored) or w3c.failed (with the failing step).
set -u
REPO=/home/sat3737/NOAA_whales/Dinov3-WhaleID
cd "$REPO"
PY="$REPO/.venv/bin/python"
LOCK="flock /var/tmp/sat3737-parts-cache/gpu_train.lock"
export PYTHONPATH="$REPO"
C=/var/tmp/sat3737-parts-cache/parts_weak
NAME=parts_seg_r05_w3c
RECIPE="--init-weights /var/tmp/sat3737-parts-cache/phase1_last.pt --epochs 10 --eval-every 5 --warmup-steps 100 --batch-size 8 --crop 512 --seed 0 --workers 5"
fail() { echo "$(date +%T) FAILED: $1"; echo "$1" > $C/w3c.failed; exit 1; }

if ! grep -q '"done"' "$C/runs/$NAME/log.jsonl" 2>/dev/null; then
  echo "$(date +%T) queue train $NAME"
  $LOCK $PY -m model_fine_tuning.parts_weak.train --n-real 2 --w-need 0.5 --w-whale 1.0 --maps-dir $C/maps_tri2 -- \
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
O=$C/report_w3c
mkdir -p $O
M="parts_seg_r05_a parts_seg_r05_s1 parts_seg_r05_w2 parts_seg_r05_w2s1 parts_seg_r05_w3 parts_seg_r05_w3b $NAME"
$PY experiments/parts_weak/compare.py --arms parts_seg_r05_s1 parts_seg_r05_w2 parts_seg_r05_w2s1 parts_seg_r05_w3 parts_seg_r05_w3b $NAME --out experiments/parts_weak/compare_w3c.json > $O/compare.out 2>&1 || fail "compare"
$PY experiments/parts_weak/fin_definition.py --models $M --out experiments/parts_weak/fin_definition_w3c.json --sheet-name fin_definition_w3c.jpg > $O/findef.out 2>&1 || fail "fin_definition"
$PY experiments/parts_weak/shares.py --models $M --out experiments/parts_weak/shares_w3c.json > $O/shares.out 2>&1 || fail "shares"
$PY experiments/parts_weak/finless.py --models $M --out experiments/parts_weak/finless_w3c.json > $O/finless.out 2>&1 || fail "finless"
for base in parts_seg_r05_w2 parts_seg_r05_w3b; do
  $PY experiments/parts_weak/detail.py --base $base --arm $NAME --out experiments/parts_weak/detail_w3c_vs_${base#parts_seg_r05_}.json > $O/detail_$base.out 2>&1 || fail "detail vs $base"
done
$PY experiments/parts_weak/fin_sheet.py --models parts_seg_r05_a parts_seg_r05_w2 parts_seg_r05_w3 parts_seg_r05_w3b $NAME --name w3c_vs_w2_fins.jpg > $O/sheet.out 2>&1 || fail "fin sheet"
$PY experiments/parts_weak/sheets.py render --name $NAME > $O/sheets.out 2>&1 || fail "before/after sheets"
date +%T > $C/w3c.done
echo "$(date +%T) $NAME trained, predicted and scored; marker w3c.done"
