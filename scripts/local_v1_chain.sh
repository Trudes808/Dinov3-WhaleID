#!/usr/bin/env bash
# Local mark descriptor: train on the train days' cross-burst pairs once the native fin / fluke charts exist, then extract
# native-resolution descriptors for the eval + val days and run the signature with them (web app: signature_local).
#   nohup setsid scripts/local_v1_chain.sh > experiments/local_v1_chain.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; export PYTHONPATH=.; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NAME="${LOCAL_NAME:-local_v1}"
until grep -q "PARTS_XR_ALLDONE" experiments/parts_xr_all.log; do sleep 60; done
echo "== train $NAME"; date
if ! { [ -f experiments/$NAME/log.jsonl ] && grep -q '"event": "done"' experiments/$NAME/log.jsonl; }; then
  $PY -m model_fine_tuning.cluster.train_local --name "$NAME" --data-root experiments/days_all --epochs "${LOCAL_EPOCHS:-8}" --batches-per-epoch 300 --P 6 --workers 6 ${LOCAL_EXTRA:-} > experiments/$NAME.stdout 2>&1
fi
echo "== extract + signature"; date
for d in 2025-02-27 2024-12-18 2025-01-15 2025-03-14 2025-02-19 2025-03-21 2024-12-26; do
  [ -f experiments/days_all/$d/local_xr_$NAME/features_v2.npz ] || $PY -m model_fine_tuning.cluster.local_feats --day-dir experiments/days_all/$d --out experiments/days_all/$d/local_xr_$NAME/features_v2.npz \
      --local-checkpoint experiments/$NAME/checkpoints/best.pt 2>&1 | grep -v -i warn | tail -1
  $PY -m model_fine_tuning.cluster.signature --day-dir experiments/days_all/$d --features experiments/days_all/$d/local_xr_$NAME/features_v2.npz \
      --labels model_fine_tuning/splits/labels_all_days.json --out experiments/days_all/$d/signature_$NAME --heat --shape --align-scale "${ALIGN_SCALE:-0.17}" --dense --dense-topk "${DENSE_TOPK:-0.2}" --rank dense 2>&1 | grep -v -i warn | grep "AUC\|Traceback\|Error"
done
echo "LOCAL_CHAIN_DONE"; date
