#!/usr/bin/env bash
# Native-resolution (3072 px) fin and fluke instance charts for every split day -> experiments/days_all/<day>/parts_xr
#   nohup setsid scripts/parts_xr_all.sh > experiments/parts_xr_all.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; export PYTHONPATH=.
DAYS=$($PY -c "import json; d=json.load(open('model_fine_tuning/splits/days.json')); print(' '.join(d['val']+d['eval']+d['train']))")
for d in $DAYS; do
  OUT=experiments/days_all/$d
  if [ -f "$OUT/parts_xr/index.jsonl" ]; then echo "== $d: exists"; continue; fi
  echo "== $d"; date
  $PY -m model_fine_tuning.cluster.parts --index "$OUT/images/index.jsonl" --canon-dir "$OUT/cse" --out "$OUT/parts_xr" --size 3072 --workers "${XR_WORKERS:-12}" \
      --fluke-min-fraction 0.3 --fin-min-fraction 0.03 --instances --only-parts fin,fluke 2>&1 | grep -v -i "warn\|_check_multiprocessing" | tail -1
done
echo "PARTS_XR_ALLDONE"; date
