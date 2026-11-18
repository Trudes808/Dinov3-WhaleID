#!/bin/bash
# Second seed of S5 (seed 1, otherwise identical): train -> predict -> e2e, each under the shared lock (no priority).
set -u
cd /home/sat3737/NOAA_whales/Dinov3-WhaleID
LOCK=/var/tmp/sat3737-parts-cache/gpu_train.lock; LOGS=/var/tmp/sat3737-parts-cache/seg_splash/logs; PY=.venv/bin/python; export PYTHONPATH=.
echo "$(date +%T) train seg_splash_s5b (seed 1)"
flock $LOCK $PY -m model_fine_tuning.seg_real.train --name seg_splash_s5b --seed 1 --real-p 0.3 --real-dir /var/tmp/sat3737-parts-cache/seg_splash/real_s5 \
  --init experiments/seg_r05_ft4/checkpoints/best.pt --epochs 8 --eval-every 4 --workers 6 --batch-size 8 > $LOGS/train_s5b.log 2>&1 || { echo "$(date +%T) failed"; exit 1; }
echo "$(date +%T) predict s5b"; flock $LOCK $PY -m model_fine_tuning.seg_real.predict --checkpoint experiments/seg_splash_s5b/checkpoints/last.pt --name s5b > $LOGS/pred_s5b.log 2>&1
echo "$(date +%T) e2e s5b"; flock $LOCK $PY -m model_fine_tuning.seg_real.e2e run --name s5b --checkpoint experiments/seg_splash_s5b/checkpoints/last.pt > $LOGS/e2e_s5b.log 2>&1
echo "$(date +%T) s5b done (e2e exit $?)"
