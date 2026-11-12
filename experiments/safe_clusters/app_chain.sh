#!/usr/bin/env bash
cd /home/sat3737/NOAA_whales/Dinov3-WhaleID
while ! grep -q EMBED_TRAIN_DONE experiments/safe_clusters/embed_train.log; do sleep 30; done
R=experiments/days_all; S=model_fine_tuning/splits
echo "=== fit gbt app on train"
.venv/bin/python -m model_fine_tuning.cluster.safe_clusters fit --root $R --days $S/days.json --labels $S/labels_all_days.json --out experiments/safe_clusters/gbt_app.pkl --kind gbt --feature-set app --split train 2>&1 | grep -v -i warn
echo "=== sweep gbt app"
.venv/bin/python -m model_fine_tuning.cluster.safe_clusters sweep --root $R --days $S/days.json --labels $S/labels_all_days.json --model experiments/safe_clusters/gbt_app.pkl --out experiments/safe_clusters/sweep_gbt_app.json 2>&1 | grep -v -i warn
echo APP_CHAIN_DONE
