#!/usr/bin/env bash
set -uo pipefail
cd /home/sat3737/NOAA_whales/Dinov3-WhaleID; export PYTHONPATH=.; R=experiments/real_2024-12-18; PY=.venv/bin/python
echo "== extract"; date
echo "(extraction already done)"
echo "== score"; date
$PY -m model_fine_tuning.cluster.local score --local $R/local_up2 --atlas $R/atlas_v2 --timestamps $R/timestamps.json --out $R/cluster_local --workers 28 2>&1 | grep -v -i warn
echo "== AUC"; date
$PY test_scripts/python/pair_auc.py $R/cluster_local/scores.npz $R/labels_from_lightroom.json 2>&1 | grep -v -i warn
echo "== done"; date
