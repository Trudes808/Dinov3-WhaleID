#!/usr/bin/env bash
set -uo pipefail
cd /home/sat3737/NOAA_whales/Dinov3-WhaleID; export PYTHONPATH=.; R=experiments/real_2024-12-18; PY=.venv/bin/python
for cfg in "0.6 1.0 r06" "20.0 0.0 unaligned"; do
  set -- $cfg
  echo "== radius $1 far $2 ($3)"; date
  $PY -m model_fine_tuning.cluster.local score --local $R/local_up2 --atlas $R/atlas_v2 --timestamps $R/timestamps.json --out $R/cluster_local_$3 --radius $1 --far $2 --workers 28 2>&1 | grep -v -i warn | grep -v "^[0-9]*/[0-9]*$" | tail -2
  $PY test_scripts/python/pair_auc.py $R/cluster_local_$3/scores.npz $R/labels_from_lightroom.json 2>&1 | grep -v -i warn | grep -E "source|local_dino"
done
echo "== done"; date
