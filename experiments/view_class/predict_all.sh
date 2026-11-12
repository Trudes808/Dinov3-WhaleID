#!/bin/bash
# views.json for every day with view-class features
cd /home/sat3737/NOAA_whales/Dinov3-WhaleID
for d in experiments/days_all/*/; do
  [ -f "$d/safe/view_feat.npz" ] || continue
  .venv/bin/python -m model_fine_tuning.cluster.view_class predict --day-dir "$d" "$@"
done
