#!/usr/bin/env bash
# Train the hi-res DINO identity head on the train days' instance charts (parts_hr) once id_v3 has finished and the
# val + train parts_hr exist, then extract global + local features for the eval and val days.
#   nohup setsid scripts/dino_hr_train.sh > experiments/dino_hr_train.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; NAME="${HR_NAME:-dino_hr}"; export PYTHONPATH=.
need=$($PY -c "import json; d=json.load(open('model_fine_tuning/splits/days.json')); print(' '.join(d['val']+d['train']))")
until { [ -f experiments/id_v3/log.jsonl ] && grep -q '"event": "done"' experiments/id_v3/log.jsonl; }; do sleep 120; done
until ok=1; for d in $need; do [ -f experiments/days_all/$d/parts_hr/index.jsonl ] || ok=0; done; [ "$ok" = 1 ]; do sleep 120; done
echo "== train $NAME"; date
if ! { [ -f experiments/$NAME/log.jsonl ] && grep -q '"event": "done"' experiments/$NAME/log.jsonl; }; then
  $PY -m model_fine_tuning.cluster.dino_hr train --name "$NAME" --data-root experiments/days_all --parts-tag parts_hr --size "${HR_SIZE:-1024}" \
      --epochs "${HR_EPOCHS:-15}" --batches-per-epoch 300 --P 6 --K 3 --unfreeze-blocks 2 --group-hard --workers 6 ${HR_EXTRA:-} > experiments/$NAME.stdout 2>&1
fi
echo "== extract $NAME features (eval + val days)"; date
for d in 2024-12-18 2024-12-26 2025-02-19 2025-03-21 2025-01-15 2025-02-27 2025-03-14; do
  until [ -f experiments/days_all/$d/parts_hr/index.jsonl ]; do sleep 60; done
  [ -f experiments/days_all/$d/$NAME/features.npz ] || $PY -m model_fine_tuning.cluster.dino_hr extract --checkpoint experiments/$NAME/checkpoints/best.pt \
      --parts experiments/days_all/$d/parts_hr --out experiments/days_all/$d/$NAME > experiments/days_all/$d/$NAME.log 2>&1
  tail -1 experiments/days_all/$d/$NAME.log
done
echo "DINO_HR_ALLDONE"; date
