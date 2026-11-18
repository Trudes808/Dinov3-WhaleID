#!/bin/bash
# seg_splash chain (LOG 2026-11-17): train the variants one after another, each GPU step under the shared lock,
# then cache their predictions on the real photos. Usage: run_chain.sh s1 s2 s3 s0   (any subset, in order)
#   s1  run05 renders + real partial labels (real share 0.3)
#   s2  s1 + real splash / wake cut-outs pasted into half of the render crops
#   s3  s2 + real-water composites (synth_aug, unchanged) on 30 % of the render frames
#   s0  control: the same fine-tuning schedule on renders only
# All fine-tune from seg_r05_ft4 (best.pt) for 8 render epochs (6 440 steps of batch 8), recipe of seg_r05_ft4 otherwise.
set -u
cd /home/sat3737/NOAA_whales/Dinov3-WhaleID
LOCK=/var/tmp/sat3737-parts-cache/gpu_train.lock
LOGS=/var/tmp/sat3737-parts-cache/seg_splash/logs
PY=".venv/bin/python"
export PYTHONPATH=.
COMMON="--init experiments/seg_r05_ft4/checkpoints/best.pt --epochs ${EPOCHS:-8} --eval-every 4 --workers 6 --batch-size 8"
for v in "$@"; do
  case $v in
    s1) ARGS="--real-p 0.3";;
    s2) ARGS="--real-p 0.3 --paste-p 0.5";;
    s3) ARGS="--real-p 0.3 --paste-p 0.5 --rw-p 0.3";;
    s0) ARGS="--real-p 0";;
    *) echo "unknown $v"; exit 1;;
  esac
  name=seg_splash_$v
  if [ ! -f experiments/$name/checkpoints/last.pt ]; then
    echo "$(date +%T) train $name $ARGS"
    flock $LOCK $PY -m model_fine_tuning.seg_real.train --name $name $ARGS $COMMON > $LOGS/train_$v.log 2>&1 || { echo "$(date +%T) $name failed"; tail -20 $LOGS/train_$v.log; exit 1; }
  fi
  echo "$(date +%T) predict $v"
  flock $LOCK $PY -m model_fine_tuning.seg_real.predict --checkpoint experiments/$name/checkpoints/last.pt --name $v > $LOGS/pred_$v.log 2>&1 || { echo "predict $v failed"; exit 1; }
  echo "$(date +%T) $v done"
done
