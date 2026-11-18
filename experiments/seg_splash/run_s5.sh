#!/bin/bash
# S5 = S4 with a gentler outline label (core eroded 1 px, ignore to 5 px, background ring 5-12 px where r04 P < 0.25):
# S4's 3-12 px ring thinned low backs. Waits for run_rest.sh (pid $1), then train -> predict -> e2e under the lock.
set -u
cd /home/sat3737/NOAA_whales/Dinov3-WhaleID
LOCK=/var/tmp/sat3737-parts-cache/gpu_train.lock; LOGS=/var/tmp/sat3737-parts-cache/seg_splash/logs; PY=.venv/bin/python; export PYTHONPATH=.
while kill -0 "$1" 2>/dev/null; do sleep 20; done
until [ -f /var/tmp/sat3737-parts-cache/seg_splash/real_s5/index.json ]; do sleep 10; done
echo "$(date +%T) train seg_splash_s5"
flock $LOCK $PY -m model_fine_tuning.seg_real.train --name seg_splash_s5 --real-p 0.3 --real-dir /var/tmp/sat3737-parts-cache/seg_splash/real_s5 \
  --init experiments/seg_r05_ft4/checkpoints/best.pt --epochs 8 --eval-every 4 --workers 6 --batch-size 8 > $LOGS/train_s5.log 2>&1 || { echo failed; exit 1; }
echo "$(date +%T) predict s5"; flock $LOCK $PY -m model_fine_tuning.seg_real.predict --checkpoint experiments/seg_splash_s5/checkpoints/last.pt --name s5 > $LOGS/pred_s5.log 2>&1
echo "$(date +%T) e2e s5"; flock $LOCK $PY -m model_fine_tuning.seg_real.e2e run --name s5 --checkpoint experiments/seg_splash_s5/checkpoints/last.pt > $LOGS/e2e_s5.log 2>&1
echo "$(date +%T) s5 done (e2e exit $?)"
