#!/usr/bin/env bash
# Train the CSE head on run04: frozen backbone, then the last two blocks tuned. Idempotent per experiment.
#   scripts/cse_train.sh            (env: CSE_EPOCHS 30, CSE_EXTRA)
set -uo pipefail
cd "$(dirname "$0")/.."
RUN=$HOME/NOAA_whales/Whale-Image-Data-Factory/data/renders/run04; SPLIT=model_fine_tuning/splits/run04.json; PY=.venv/bin/python
finished() { [ -f "experiments/$1/log.jsonl" ] && grep -q '"event": "done"' "experiments/$1/log.jsonl"; }
for v in "cse_r04_frozen 0" "cse_r04_ft2 2"; do
  set -- $v; NAME=$1; UB=$2
  echo "== train $NAME (unfreeze $UB)"; date
  finished "$NAME" || $PY -m model_fine_tuning.canon.train_cse --name "$NAME" --run-dir "$RUN" --split "$SPLIT" --unfreeze-blocks "$UB" \
      --epochs "${CSE_EPOCHS:-30}" --batch-size 8 --workers 6 --eval-every 2 --max-val 600 ${CSE_EXTRA:-} > "experiments/$NAME.stdout" 2>&1
done
echo "== cse training done"; date
