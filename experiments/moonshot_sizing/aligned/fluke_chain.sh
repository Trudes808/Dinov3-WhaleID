#!/bin/bash
# CPU chain after the fluke maps: decode tips, fluke tokens, matrices, bench
cd /home/sat3737/NOAA_whales/Dinov3-WhaleID/experiments/moonshot_sizing/aligned
PY=/home/sat3737/NOAA_whales/Dinov3-WhaleID/.venv/bin/python
L=/home/sat3737/NOAA_whales/moonshot_cache/aligned
while ! grep -q "^done" $L/fluke_maps.log; do grep -q Traceback $L/fluke_maps.log && exit 1; sleep 20; done
date; OMP_NUM_THREADS=2 $PY fluke.py decode || exit 1
OMP_NUM_THREADS=2 $PY sheet_fluke.py 24 3 fluke_sheet1
ALIGNED_DEV=cpu OMP_NUM_THREADS=16 $PY extract.py fluke || exit 1
date; for v in posc edge; do OMP_NUM_THREADS=6 $PY precompute.py fluke $v & done; wait
date; OMP_NUM_THREADS=8 $PY bench_fluke.py > bench_fluke.log 2>&1
OMP_NUM_THREADS=8 $PY learned.py fluke 3 --fuse 2>&1 | grep -v Warning > bench_fluke_cells.log
date; echo CHAIN_DONE
