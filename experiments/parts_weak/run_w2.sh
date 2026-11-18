#!/usr/bin/env bash
# W2 = W1 + "no background on certain-whale pixels" (build_whale.py), same seed and recipe; queued after the noneed run.
#   setsid nohup bash experiments/parts_weak/run_w2.sh > /var/tmp/sat3737-parts-cache/parts_weak/run_w2.log 2>&1 &
# Markers: w2.done when W2 is trained AND scored (+ compare / detail written); then, by a rule fixed before any result:
# if the noneed run's fin regions on the 2025-03-18 clicks (hr, median region length / clicked base) are at most 0.51
# (the midpoint of baseline 0.29 and W1 0.74), the "some fin" term makes the fat fins and w2b = W2 without that term
# is trained and scored too (w2b.done); otherwise w2b.skipped.
set -u
REPO=/home/sat3737/NOAA_whales/Dinov3-WhaleID
cd "$REPO"
PY="$REPO/.venv/bin/python"
LOCK="flock /var/tmp/sat3737-parts-cache/gpu_train.lock"
export PYTHONPATH="$REPO"
C=/var/tmp/sat3737-parts-cache/parts_weak
RECIPE="--init-weights /var/tmp/sat3737-parts-cache/phase1_last.pt --epochs 10 --eval-every 5 --warmup-steps 100 --batch-size 8 --crop 512 --seed 0 --workers 5"

until grep -q '"seconds"' $C/build_whale.log 2>/dev/null; do sleep 60; done
echo "$(date +%T) whale maps ready; waiting for noneed.done"
until [ -f $C/noneed.done ]; do sleep 60; done

arm() {   # name, extra weak args, marker
  local NAME=$1 WEAK=$2 MARK=$3
  if ! grep -q '"done"' "$C/runs/$NAME/log.jsonl" 2>/dev/null; then
    echo "$(date +%T) queue train $NAME"
    $LOCK $PY -m model_fine_tuning.parts_weak.train --n-real 2 --w-need 0.5 --w-whale 1.0 $WEAK -- --name $NAME --out-root $C/runs $RECIPE > $C/train_$NAME.out 2>&1
    echo "$(date +%T) train $NAME exit $?"
  fi
  [ -f "$C/runs/$NAME/checkpoints/best.pt" ] || { echo "$NAME has no best.pt"; return 1; }
  if [ ! -f "experiments/parts_weak/${NAME}_real_metrics.json" ]; then
    echo "$(date +%T) queue score $NAME"
    $LOCK $PY experiments/parts_weak/eval_w.py --name $NAME --checkpoint $C/runs/$NAME/checkpoints/best.pt \
      --metrics experiments/parts_weak/${NAME}_real_metrics.json > $C/eval_$NAME.out 2>&1
    echo "$(date +%T) score $NAME exit $?"
  fi
  $PY experiments/parts_weak/compare.py --arms parts_seg_r05_w1 parts_seg_r05_w1_noneed $NAME --out experiments/parts_weak/compare_$NAME.json > $C/compare_$NAME.out 2>&1
  $PY experiments/parts_weak/detail.py --arm $NAME --out experiments/parts_weak/detail_$NAME.json > $C/detail_$NAME.out 2>&1
  date +%T > $C/$MARK
  echo "$(date +%T) $NAME scored, marker $MARK"
}

arm parts_seg_r05_w2 "" w2.done

if $PY - <<'EOF'
import sys
sys.path.insert(0, "experiments/parts_weak")
import numpy as np, fin_definition as FD
rows, _ = FD.real("2025-03-18", ["parts_seg_r05_w1_noneed"])
v = [r["parts_seg_r05_w1_noneed"]["length_over_base"] for r in rows if "length_over_base" in r["parts_seg_r05_w1_noneed"]]
med = float(np.median(v))
print(f"noneed fin length / clicked base on 2025-03-18 (hr): median {med:.3f} (n {len(v)}); rule: w2b if <= 0.51")
sys.exit(0 if med <= 0.51 else 1)
EOF
then
  arm parts_seg_r05_w2b "--no-need-fin" w2b.done
else
  date +%T > $C/w2b.skipped
  echo "$(date +%T) w2b skipped by the rule"
fi
echo "$(date +%T) all done"
