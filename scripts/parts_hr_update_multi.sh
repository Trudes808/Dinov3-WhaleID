#!/usr/bin/env bash
# After parts_hr_all.sh: redo the multi-component photos of every split day with the coverage-based instance split.
#   nohup setsid scripts/parts_hr_update_multi.sh > experiments/parts_hr_update_multi.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; export PYTHONPATH=.
until grep -q PARTS_HR_ALLDONE experiments/parts_hr_all.log; do sleep 60; done
DAYS=$($PY -c "import json; d=json.load(open('model_fine_tuning/splits/days.json')); print(' '.join(d['val']+d['train']+d['eval']))")
for d in $DAYS; do
  OUT=experiments/days_all/$d; echo "== $d"
  $PY -m model_fine_tuning.cluster.parts --index "$OUT/images/index.jsonl" --canon-dir "$OUT/cse" --out "$OUT/parts_hr" --size 1536 --workers "${HR_WORKERS:-12}" \
      --fluke-min-fraction 0.3 --fin-min-fraction 0.03 --instances --update-multi 2>&1 | grep -v -i "warn\|_check_multiprocessing" | tail -2
done
echo "PARTS_HR_UPDATE_DONE"; date
