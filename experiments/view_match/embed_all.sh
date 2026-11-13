#!/bin/bash
# view_match.npz for every day with a view_class.json
cd /home/sat3737/NOAA_whales/Dinov3-WhaleID
for d in experiments/days_all/*/; do
  [ -f "$d/safe/view_class.json" ] || continue
  .venv/bin/python -m model_fine_tuning.cluster.view_match embed --day-dir "${d%/}" "$@"
done
