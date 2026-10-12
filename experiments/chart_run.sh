#!/usr/bin/env bash
set -uo pipefail
cd /home/sat3737/NOAA_whales/Dinov3-WhaleID; export PYTHONPATH=.; R=experiments/real_2024-12-18; PY=.venv/bin/python
echo "== extract full-resolution fin charts"; date
$PY -m model_fine_tuning.cluster.chart extract --index $R/images/index.jsonl --canon-dir $R/canon_r02_ft4_v2 --masks $R/pred_r02 --out $R/charts --size 768 --workers 10 2>&1 | grep -v -i warn | grep -v "^[0-9]* charts$"
echo "== score"; date
$PY -m model_fine_tuning.cluster.chart score --charts $R/charts --timestamps $R/timestamps.json --out $R/cluster_chart 2>&1 | grep -v -i warn | grep -v "^[0-9]*/[0-9]*$"
echo "== AUC"; date
$PY test_scripts/python/pair_auc.py $R/cluster_chart/scores.npz $R/labels_from_lightroom.json 2>&1 | grep -v -i warn | grep -E "source|chart|time"
echo "== done"; date
