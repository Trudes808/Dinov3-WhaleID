#!/usr/bin/env bash
# Shape-only fin / fluke embedding (train_id on parts_sil, group-hard, selected on within-group AUC) after the fin_xr2 chain,
# then its channel, the within-group table and the two-stage clustering with shape fins / flukes and with shape + texture.
#   nohup setsid scripts/sil_chain.sh > experiments/sil_chain.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; export PYTHONPATH=.; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NAME="${SIL_NAME:-sil_v1}"
until grep -q "FIN_XR_DONE" experiments/fin_xr2_chain.log; do sleep 60; done
echo "== train $NAME"; date
if ! { [ -f experiments/$NAME/log.jsonl ] && grep -q '"event": "done"' experiments/$NAME/log.jsonl; }; then
  $PY -m model_fine_tuning.cluster.train_id --name "$NAME" --data-root experiments/days_all --parts-tag parts_sil --only-parts fin,fluke --size 448 \
      --epochs "${SIL_EPOCHS:-10}" --batches-per-epoch 300 --P 8 --K 4 --unfreeze-blocks 4 --group-hard --select within_group --workers 6 ${SIL_EXTRA:-} > experiments/$NAME.stdout 2>&1
fi
echo "== channels idsil:$NAME"; date
scripts/cluster_v3_all.sh channels "idsil:experiments/$NAME/checkpoints/best.pt" ""
echo "== two-stage: shape fins / flukes"; date
$PY -m model_fine_tuning.cluster.two_stage --root experiments/days_all --labels model_fine_tuning/splits/labels_all_days.json --days model_fine_tuning/splits/days.json \
    --source "body=id:pair_student,fin=idsil:$NAME,fluke=idsil:$NAME" --tune --tag two_stage_sil 2>&1 | grep -v -i warn
echo "== single-stage: student + shape"; date
scripts/cluster_v3_all.sh cluster _sil --use "id:pair_student,idsil:$NAME" --thr-offset 0.0 --cannot-link none --max-link-cost 1e9
echo "SIL_DONE"; date
