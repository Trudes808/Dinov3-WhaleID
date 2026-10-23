#!/usr/bin/env bash
# Waits for scripts/run05_train.sh to log "done", then runs scripts/run05_evaluate.sh. Detached:
#   nohup setsid scripts/run05_after_train.sh > experiments/run05_after_train.log 2>&1 &
cd /home/sat3737/NOAA_whales/Dinov3-WhaleID
while ! grep -q "\[run05_train .*\] done" experiments/run05_train.log 2>/dev/null; do sleep 300; done
echo "[run05_after_train $(date -u +%FT%TZ)] training done; evaluating"
scripts/run05_evaluate.sh
echo "[run05_after_train $(date -u +%FT%TZ)] all done"
