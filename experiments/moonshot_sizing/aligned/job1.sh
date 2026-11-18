#!/bin/bash
# one GPU job: dorsal tokens for all train + val days, then fluke probability maps for the train days
cd /home/sat3737/NOAA_whales/Dinov3-WhaleID/experiments/moonshot_sizing/aligned
export OMP_NUM_THREADS=8
PY=/home/sat3737/NOAA_whales/Dinov3-WhaleID/.venv/bin/python
date; $PY extract.py dorsal && date && $PY fluke_maps.py; date
