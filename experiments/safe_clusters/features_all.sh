#!/usr/bin/env bash
cd /home/sat3737/NOAA_whales/Dinov3-WhaleID
for day in 2025-01-15 2025-02-27 2025-03-14 2024-12-18 2024-12-26 2025-02-19 2025-03-21; do
  .venv/bin/python -m model_fine_tuning.cluster.safe_clusters features --day-dir experiments/days_all/$day --workers 6 2>&1 | grep -v -i warn &
done
wait
echo FEATURES_DONE
