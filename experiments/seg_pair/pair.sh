#!/bin/bash
# Paired reflection test (2026-11-17). Waits for the factory's r06_pair.done, checks the pairing, trains 8 models
# (A = renders only from the DINOv3 weights, B = seg_r05_ft4 fine-tuned; each on arm m1 = mirror, m2 = no mirror;
# seeds 0 and 1), scores them on real photos and on both arms' renders. Every GPU step under the shared lock.
# Marker /var/tmp/sat3737-parts-cache/seg_pair/pair.done (or pair.failed with the step); log pair.log next to it.
set -u
cd /home/sat3737/NOAA_whales/Dinov3-WhaleID
R=/home/sat3737/NOAA_whales/Whale-Image-Data-Factory/data/renders
CK=/var/tmp/sat3737-parts-cache/seg_pair; LOCK=/var/tmp/sat3737-parts-cache/gpu_train.lock; PY=.venv/bin/python; export PYTHONPATH=.
mkdir -p $CK; rm -f $CK/pair.done $CK/pair.failed
fail() { echo "$(date +%T) FAILED: $1"; echo "$1" > $CK/pair.failed; exit 1; }
echo "$(date +%T) waiting for $R/r06_pair.done"
until [ -f $R/r06_pair.done ] || [ -f $R/r06_pair.failed ]; do sleep 60; done
[ -f $R/r06_pair.failed ] && fail "factory chain failed: $(cat $R/r06_pair.failed)"
$PY - <<'EOF' || fail "pairing check"
import json; c = json.load(open("/home/sat3737/NOAA_whales/Whale-Image-Data-Factory/data/renders/r06_pair_check.json")); print(c)
assert c["shots_m2"] == c["shots_m1"] and c["shotlist_lines_identical"] == c["shots_m2"], "shot lists differ"
for k in ("instances", "parts"):
    assert c[k]["different"] == 0 and c[k]["missing_in_m1"] == 0, f"{k} labels differ"
EOF
$PY -m model_fine_tuning.seg_real.pair split || fail "split / masks check"
$PY -m model_fine_tuning.seg_real.pair pairsheet > $CK/pairsheet.log 2>&1 || echo "pairsheet failed (not fatal)"
SPLIT=experiments/seg_pair/split_pair.json
for s in 0 1; do for m in m1 m2; do
  for rec in A B; do
    n=seg_pair_${rec}_${m}_s$s
    [ -f experiments/$n/checkpoints/last.pt ] && continue
    if [ $rec = A ]; then ARGS="--epochs 30 --eval-every 10"; else ARGS="--init experiments/seg_r05_ft4/checkpoints/best.pt --epochs 15 --eval-every 15 --warmup-steps 100"; fi
    echo "$(date +%T) train $n"
    flock $LOCK $PY -m model_fine_tuning.seg_real.train --name $n --run-dir $R/r06_pair_$m --split $SPLIT --real-p 0 --seed $s --workers 6 --batch-size 8 $ARGS > $CK/train_${rec}_${m}_s$s.log 2>&1 || fail "train $n"
  done
done; done
for rec in A B; do for m in m1 m2; do for s in 0 1; do
  n=${rec}_${m}_s$s; echo "$(date +%T) predict $n"
  flock $LOCK $PY -m model_fine_tuning.seg_real.predict --checkpoint experiments/seg_pair_$n/checkpoints/last.pt --name $n > $CK/pred_$n.log 2>&1 || fail "predict $n"
done; done; done
echo "$(date +%T) renders eval"; flock $LOCK $PY -m model_fine_tuning.seg_real.pair renders > $CK/renders.log 2>&1 || fail "renders eval"
echo "$(date +%T) real"; $PY -m model_fine_tuning.seg_real.pair real > $CK/real.log 2>&1 || fail "real"
echo "$(date +%T) knn"; nice $PY -m model_fine_tuning.seg_real.pair knn > $CK/knn.log 2>&1 || echo "knn failed (not fatal)"
$PY -m model_fine_tuning.seg_real.pair realsheet > $CK/realsheet.log 2>&1 || echo "realsheet failed (not fatal)"
$PY -m model_fine_tuning.seg_real.pair summary > $CK/summary.log 2>&1 || fail "summary"
date +%T > $CK/pair.done; echo "$(date +%T) pair done"
