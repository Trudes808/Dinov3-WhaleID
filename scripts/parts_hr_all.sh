#!/usr/bin/env bash
# Hi-res (1536 px) instance-split part charts for every split day (val, train, eval order): cluster/parts.py --instances
#   nohup setsid scripts/parts_hr_all.sh > experiments/parts_hr_all.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; export PYTHONPATH=.
DAYS=$($PY -c "import json; d=json.load(open('model_fine_tuning/splits/days.json')); print(' '.join(d['val']+d['train']+d['eval']))")
for d in $DAYS; do
  OUT=experiments/days_all/$d
  if [ -f "$OUT/parts_hr/index.jsonl" ]; then echo "== $d: parts_hr exists"; continue; fi
  echo "== $d: parts_hr"; date
  $PY -m model_fine_tuning.cluster.parts --index "$OUT/images/index.jsonl" --canon-dir "$OUT/cse" --out "$OUT/parts_hr" --size 1536 --workers "${HR_WORKERS:-12}" \
      --fluke-min-fraction 0.3 --fin-min-fraction 0.03 --instances 2>&1 | grep -v -i "warn\|_check_multiprocessing" | tail -1
done
echo "PARTS_HR_ALLDONE"; date
