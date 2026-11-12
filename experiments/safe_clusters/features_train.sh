#!/usr/bin/env bash
cd /home/sat3737/NOAA_whales/Dinov3-WhaleID
for day in $(python3 -c "import json; print(' '.join(json.load(open('model_fine_tuning/splits/days.json'))['train']))"); do
  .venv/bin/python -m model_fine_tuning.cluster.safe_clusters features --day-dir experiments/days_all/$day --workers 4 2>&1 | grep -v -i warn &
  while [ $(jobs -r | wc -l) -ge 6 ]; do sleep 2; done
done
wait
echo TRAIN_FEATURES_DONE
