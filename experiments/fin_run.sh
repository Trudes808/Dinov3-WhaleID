#!/usr/bin/env bash
set -uo pipefail
cd /home/sat3737/NOAA_whales/Dinov3-WhaleID; export PYTHONPATH=.; R=experiments/real_2024-12-18; PY=.venv/bin/python
echo "== fin extract"; date
$PY -m model_fine_tuning.cluster.fin extract --images $R/images --canon-dir $R/canon_r02_ft4_v2 --masks $R/pred_r02 --out $R/fin 2>&1 | grep -v -i warn | tail -3
echo "== fin score + AUC"; date
$PY -m model_fine_tuning.cluster.fin score --fin $R/fin --timestamps $R/timestamps.json --out $R/cluster_fin 2>&1 | grep -v -i warn | tail -1
$PY test_scripts/python/pair_auc.py $R/cluster_fin/scores.npz $R/labels_from_lightroom.json 2>&1 | grep -v -i warn
echo "== done"; date
