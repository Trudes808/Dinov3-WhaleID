#!/bin/bash
# Second chain (after S3's first launch failed on short composites and S4 on a missing label cache): for each variant
# train (if no last.pt) -> predict -> e2e, every GPU step under the shared lock. Usage: run_rest.sh s4 s3 s0
set -u
cd /home/sat3737/NOAA_whales/Dinov3-WhaleID
LOCK=/var/tmp/sat3737-parts-cache/gpu_train.lock; LOGS=/var/tmp/sat3737-parts-cache/seg_splash/logs; PY=.venv/bin/python; export PYTHONPATH=.
COMMON="--init experiments/seg_r05_ft4/checkpoints/best.pt --epochs 8 --eval-every 4 --workers 6 --batch-size 8"
until [ -f /var/tmp/sat3737-parts-cache/seg_splash/real_strict/index.json ]; do sleep 10; done
for v in "$@"; do
  case $v in
    s3) ARGS="--real-p 0.3 --paste-p 0.5 --rw-p 0.3";;
    s0) ARGS="--real-p 0";;
    s4) ARGS="--real-p 0.3 --real-dir /var/tmp/sat3737-parts-cache/seg_splash/real_strict";;
  esac
  name=seg_splash_$v
  if [ ! -f experiments/$name/checkpoints/last.pt ]; then
    rm -rf experiments/$name
    echo "$(date +%T) train $name $ARGS"
    flock $LOCK $PY -m model_fine_tuning.seg_real.train --name $name $ARGS $COMMON > $LOGS/train_$v.log 2>&1 || { echo "$(date +%T) $name failed"; continue; }
  fi
  echo "$(date +%T) predict $v"; flock $LOCK $PY -m model_fine_tuning.seg_real.predict --checkpoint experiments/$name/checkpoints/last.pt --name $v > $LOGS/pred_$v.log 2>&1
  echo "$(date +%T) e2e $v"; rm -rf /var/tmp/sat3737-parts-cache/seg_splash/e2e/$v
  flock $LOCK $PY -m model_fine_tuning.seg_real.e2e run --name $v --checkpoint experiments/$name/checkpoints/last.pt > $LOGS/e2e_$v.log 2>&1
  echo "$(date +%T) $v done (e2e exit $?)"
done
