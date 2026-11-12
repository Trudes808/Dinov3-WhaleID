#!/usr/bin/env bash
# One worker of the window-feature extraction: takes the next day nobody has taken (mkdir is the lock), train days
# first, eval days last (features only: the page serves their suggestions too; nothing is tuned on them).
# Three of these run side by side: the crop pipeline is bound by one CPU thread per worker, not by the GPU.
R=/home/sat3737/NOAA_whales/Dinov3-WhaleID; cd $R; export PYTHONPATH=$R OMP_NUM_THREADS=1
D=$R/experiments/days_all; L=$R/experiments/stage2_linker/locks; mkdir -p $L
for day in 2025-01-17 2025-01-20 2025-01-23 2025-02-03 2025-02-12 2025-02-24 2025-02-28 2025-03-03 2025-03-06 2025-03-13 2025-03-18 2025-03-20 2025-03-24 2025-03-26 2025-04-01 2025-04-09 2025-04-11 \
           2024-12-18 2024-12-26 2025-02-19 2025-03-21; do
  [ -f $D/$day/safe/windows.npz ] && continue
  mkdir $L/$day 2>/dev/null || continue
  $R/.venv/bin/python -u -m model_fine_tuning.cluster.link_scores windows --day-dir $D/$day || echo "FAILED $day"
done; echo "WORKER $1 DONE"
