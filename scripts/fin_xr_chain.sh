#!/usr/bin/env bash
# Native-resolution fin / fluke specialist (train_id on parts_xr, central 1536 px -> 1024, group-hard, selected on the
# within-group AUC), then its channel on the 7 days, the within-group table and the two-stage + single-stage clustering.
#   nohup setsid scripts/fin_xr_chain.sh > experiments/fin_xr_chain.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; export PYTHONPATH=.; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NAME="${XR_NAME:-fin_xr}"
echo "== train $NAME"; date
if ! { [ -f experiments/$NAME/log.jsonl ] && grep -q '"event": "done"' experiments/$NAME/log.jsonl; }; then
  $PY -m model_fine_tuning.cluster.train_id --name "$NAME" --data-root experiments/days_all --parts-tag parts_xr --only-parts fin,fluke --center-crop 1536 --size 1024 \
      --epochs "${XR_EPOCHS:-10}" --batches-per-epoch 300 --P 6 --K 3 --unfreeze-blocks 4 --group-hard --select within_group --workers 6 ${XR_EXTRA:-} > experiments/$NAME.stdout 2>&1
fi
echo "== channels idxr:$NAME"; date
scripts/cluster_v3_all.sh channels "idxr:experiments/$NAME/checkpoints/best.pt" ""
echo "== two-stage with specialist fins / flukes"; date
$PY -m model_fine_tuning.cluster.two_stage --root experiments/days_all --labels model_fine_tuning/splits/labels_all_days.json --days model_fine_tuning/splits/days.json \
    --source "body=id:pair_student,fin=idxr:$NAME,fluke=idxr:$NAME" --tune --tag two_stage_xr 2>&1 | grep -v -i warn
echo "== single-stage with student body + specialist fins / flukes"; date
scripts/cluster_v3_all.sh cluster _xr --use "id:pair_student,idxr:$NAME" --thr-offset 0.0 --cannot-link none --max-link-cost 1e9
echo "FIN_XR_DONE"; date
