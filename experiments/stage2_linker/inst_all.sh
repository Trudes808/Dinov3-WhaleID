#!/usr/bin/env bash
# Instance crop features only (quick), for every day with safe clusters: input of the non-whale filter.
R=/home/sat3737/NOAA_whales/Dinov3-WhaleID; cd $R; export PYTHONPATH=$R OMP_NUM_THREADS=1
for d in $R/experiments/days_all/*/; do
  [ -f $d/safe/photos.json ] || continue; [ -f $d/safe/inst_feat.npz ] && continue
  $R/.venv/bin/python -u -m model_fine_tuning.cluster.link_scores instances --day-dir $d || echo "FAILED $d"
done; echo ALL DONE
