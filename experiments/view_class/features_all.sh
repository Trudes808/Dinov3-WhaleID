#!/bin/bash
# view-class features for every day with safe clusters (GPU, ~10-20 s per day)
cd /home/sat3737/NOAA_whales/Dinov3-WhaleID
for d in experiments/days_all/*/; do
  [ -f "$d/safe/clusters.json" ] || continue
  [ -f "$d/safe/view_feat.npz" ] && continue
  .venv/bin/python -m model_fine_tuning.cluster.view_class features --day-dir "$d"
done
echo ALL DONE
