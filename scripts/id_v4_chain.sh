#!/usr/bin/env bash
# id_v4: the id_v3 recipe (group-hard, 512 px) on a short cosine schedule (10 epochs), once the LightGlue channel pass has
# freed the GPU; then its channel on the eval + val days and the clustering with it (bursts rule) + tuning.
#   nohup setsid scripts/id_v4_chain.sh > experiments/id_v4_chain.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; export PYTHONPATH=.
until [ "$(grep -c 'channels:' experiments/cluster_v3_channels_lg.log)" -ge 7 ]; do sleep 60; done
echo "== train id_v4"; date
if ! { [ -f experiments/id_v4/log.jsonl ] && grep -q '"event": "done"' experiments/id_v4/log.jsonl; }; then
  $PY -m model_fine_tuning.cluster.train_id --name id_v4 --data-root experiments/days_all --labels model_fine_tuning/splits/labels_all_days.json \
      --days model_fine_tuning/splits/days.json --epochs 10 --batches-per-epoch 300 --unfreeze-blocks 4 --workers 6 --group-hard --size 512 > experiments/id_v4.stdout 2>&1
fi
echo "== id_v4 channel"; date
scripts/cluster_v3_all.sh channels "id:experiments/id_v4/checkpoints/best.pt" ""
echo "== cluster with id_v4"; date
scripts/cluster_v3_all.sh cluster _v4id --use id:id_v4 --thr-offset 0.0 --cannot-link none --max-link-cost 1e9
echo "ID_V4_DONE"; date
