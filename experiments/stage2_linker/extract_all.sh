#!/usr/bin/env bash
# Window features (frozen DINOv3, whitened) and instance crop features for every day with safe clusters.
# The reference day first (it fits the whitening), then the val days, the other train days, the eval days.
# Eval days: features only (the page serves their suggestions too); nothing is tuned on them.
set -uo pipefail
R=/home/sat3737/NOAA_whales/Dinov3-WhaleID; cd $R
export PYTHONPATH=$R OMP_NUM_THREADS=1
PY=$R/.venv/bin/python; D=$R/experiments/days_all
$PY -u -m model_fine_tuning.cluster.link_scores windows --day-dir $D/2025-01-02 --fit-whitening || exit 1
for day in 2025-01-15 2025-02-27 2025-03-14 \
           2024-12-23 2025-01-08 2025-01-16 2025-01-17 2025-01-20 2025-01-23 2025-02-03 2025-02-12 2025-02-24 2025-02-28 2025-03-03 2025-03-06 2025-03-13 2025-03-18 2025-03-20 2025-03-24 2025-03-26 2025-04-01 2025-04-09 2025-04-11 \
           2024-12-18 2024-12-26 2025-02-19 2025-03-21; do
  [ -f $D/$day/safe/windows.npz ] && { echo "$day: already done"; continue; }
  $PY -u -m model_fine_tuning.cluster.link_scores windows --day-dir $D/$day || echo "FAILED $day"
done
echo ALL DONE
