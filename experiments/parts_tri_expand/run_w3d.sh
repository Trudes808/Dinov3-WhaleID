#!/usr/bin/env bash
# W3d = W3c's recipe exactly (two-sided real label, OLD render labels, whale term on) with the EXPANDED clicked triangle
# as the real dorsal label (model_fine_tuning/parts_weak/build_tri3.py -> maps_tri3). Seeds 0 and 1, each trained, then
# predicted + scored by eval_w.py as W3c was (shadow day folders, never days_all).
#   setsid nohup bash experiments/parts_tri_expand/run_w3d.sh > /var/tmp/sat3737-parts-cache/parts_weak/run_w3d.log 2>&1 &
# Marker: /var/tmp/sat3737-parts-cache/parts_weak/w3d.done, or w3d.failed with the failing step.
set -u
REPO=/home/sat3737/NOAA_whales/Dinov3-WhaleID
cd "$REPO"
PY="$REPO/.venv/bin/python"
LOCK="flock /var/tmp/sat3737-parts-cache/gpu_train.lock"
export PYTHONPATH="$REPO"
C=/var/tmp/sat3737-parts-cache/parts_weak
fail() { echo "$(date +%T) FAILED: $1"; echo "$1" > $C/w3d.failed; exit 1; }
[ -f $C/maps_tri3_summary.json ] || fail "maps_tri3 missing (run build_tri3 first)"
for SEED in 0 1; do
  NAME=parts_seg_r05_w3d; [ $SEED = 1 ] && NAME=parts_seg_r05_w3ds1
  RECIPE="--init-weights /var/tmp/sat3737-parts-cache/phase1_last.pt --epochs 10 --eval-every 5 --warmup-steps 100 --batch-size 8 --crop 512 --seed $SEED --workers 5"
  if ! grep -q '"done"' "$C/runs/$NAME/log.jsonl" 2>/dev/null; then
    echo "$(date +%T) queue train $NAME (GPU used: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader))"
    $LOCK $PY -m model_fine_tuning.parts_weak.train --n-real 2 --w-need 0.5 --w-whale 1.0 --maps-dir $C/maps_tri3 -- \
      --name $NAME --out-root $C/runs $RECIPE > $C/train_$NAME.out 2>&1 || fail "train $NAME (see $C/train_$NAME.out)"
    echo "$(date +%T) train $NAME done"
  fi
  [ -f "$C/runs/$NAME/checkpoints/best.pt" ] || fail "no best.pt for $NAME"
  if [ ! -f "experiments/parts_tri_expand/${NAME}_real_metrics.json" ]; then
    echo "$(date +%T) queue predict + score $NAME"
    $LOCK $PY experiments/parts_weak/eval_w.py --name $NAME --checkpoint $C/runs/$NAME/checkpoints/best.pt \
      --metrics experiments/parts_tri_expand/${NAME}_real_metrics.json > $C/eval_$NAME.out 2>&1 || fail "predict/score $NAME (see $C/eval_$NAME.out)"
    echo "$(date +%T) score $NAME done"
  fi
  rm -f "$C/runs/$NAME/checkpoints/last.pt"          # best.pt is the one used (as W3c); saves 350 MB
done
date +%T > $C/w3d.done
echo "$(date +%T) W3d two seeds trained, predicted and scored; marker w3d.done"
