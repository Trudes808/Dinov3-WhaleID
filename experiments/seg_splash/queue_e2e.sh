#!/bin/bash
# After run_chain.sh: queue every remaining GPU step at once, each under the shared lock (they run one at a time):
# S4 (S1 with strict labels: CACHE/real_strict, `data build --strict`) training + prediction, and the instances_v2
# stage end to end (seg_real.e2e run) for plain seg_r05_ft4 and for each variant.
# Usage: queue_e2e.sh <pid of run_chain.sh or 0>
set -u
cd /home/sat3737/NOAA_whales/Dinov3-WhaleID
LOCK=/var/tmp/sat3737-parts-cache/gpu_train.lock
LOGS=/var/tmp/sat3737-parts-cache/seg_splash/logs
export PYTHONPATH=.
while [ "${1:-0}" != 0 ] && kill -0 "$1" 2>/dev/null; do sleep 20; done
echo "$(date +%T) chain finished; queueing s4 and e2e"
( if [ ! -f experiments/seg_splash_s4/checkpoints/last.pt ]; then
    flock $LOCK .venv/bin/python -m model_fine_tuning.seg_real.train --name seg_splash_s4 --real-p 0.3 --real-dir /var/tmp/sat3737-parts-cache/seg_splash/real_strict \
      --init experiments/seg_r05_ft4/checkpoints/best.pt --epochs 8 --eval-every 4 --workers 6 --batch-size 8 > $LOGS/train_s4.log 2>&1
  fi
  flock $LOCK .venv/bin/python -m model_fine_tuning.seg_real.predict --checkpoint experiments/seg_splash_s4/checkpoints/last.pt --name s4 > $LOGS/pred_s4.log 2>&1
  echo "$(date +%T) s4 train+predict exit $?"
  flock $LOCK .venv/bin/python -m model_fine_tuning.seg_real.e2e run --name s4 --checkpoint experiments/seg_splash_s4/checkpoints/last.pt > $LOGS/e2e_s4.log 2>&1
  echo "$(date +%T) e2e s4 exit $?" ) &
sleep 2
declare -A CK=([r05]=experiments/seg_r05_ft4/checkpoints/best.pt [s1]=experiments/seg_splash_s1/checkpoints/last.pt
               [s2]=experiments/seg_splash_s2/checkpoints/last.pt [s3]=experiments/seg_splash_s3/checkpoints/last.pt
               [s0]=experiments/seg_splash_s0/checkpoints/last.pt)
for n in r05 s1 s2 s3 s0; do
  [ -f "${CK[$n]}" ] || { echo "skip $n (no checkpoint)"; continue; }
  ( flock $LOCK .venv/bin/python -m model_fine_tuning.seg_real.e2e run --name $n --checkpoint ${CK[$n]} > $LOGS/e2e_$n.log 2>&1; echo "$(date +%T) e2e $n exit $?" ) &
  sleep 2
done
wait
echo "$(date +%T) all e2e done"
