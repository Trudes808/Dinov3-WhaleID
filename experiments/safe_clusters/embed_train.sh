#!/usr/bin/env bash
# pair_student_g identity embeddings (body / fin / fluke charts) for the train days -> DAY/cluster_v3/{channels,embeddings}.npz
cd /home/sat3737/NOAA_whales/Dinov3-WhaleID
for day in $(python3 -c "import json; print(' '.join(json.load(open('model_fine_tuning/splits/days.json'))['train']))"); do
  d=experiments/days_all/$day
  [ -f $d/cluster_v3/embeddings.npz ] && { echo "$day: embeddings exist"; continue; }
  echo "== $day $(date -u +%T)"
  .venv/bin/python -m model_fine_tuning.cluster.cluster_day channels --day-dir $d --out $d/cluster_v3 --sources id:experiments/pair_student_g/checkpoints/best.pt --local "" --candidates 0 2>&1 | grep -v -i "warn" | tail -4
done
echo EMBED_TRAIN_DONE
