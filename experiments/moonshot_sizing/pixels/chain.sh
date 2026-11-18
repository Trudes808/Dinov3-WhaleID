#!/bin/bash
# pixels extraction chain: one flock per <= ~17 min slice, until every job/day cache exists
cd "$(dirname "$0")"
PY=/home/sat3737/NOAA_whales/Dinov3-WhaleID/.venv/bin/python
DAYS="2025-01-15 2025-02-27 2025-03-14 2024-12-23 2025-01-02 2025-01-08 2025-01-16 2025-01-17 2025-01-20 2025-01-23 2025-02-03 2025-02-12 2025-02-24 2025-02-28 2025-03-03 2025-03-06 2025-03-13 2025-03-18 2025-03-20 2025-03-24 2025-03-26 2025-04-01 2025-04-09 2025-04-11"
JOBS=${1:-base,norm,res}
for i in $(seq 1 12); do
  OMP_NUM_THREADS=8 MAXSEC=900 flock /var/tmp/sat3737-parts-cache/gpu_train.lock $PY extract.py $JOBS $DAYS 2>&1 | grep --line-buffered -v Warning
  n=0; for j in ${JOBS//,/ }; do for d in $DAYS; do [ -f /home/sat3737/NOAA_whales/moonshot_cache/pixels/$j/$d.npz ] && n=$((n+1)); done; done
  echo "slice $i: $n caches"
  [ $n -ge $(( $(echo $JOBS | tr ',' ' ' | wc -w) * 24 )) ] && { echo CHAIN DONE; break; }
  sleep 20
done
