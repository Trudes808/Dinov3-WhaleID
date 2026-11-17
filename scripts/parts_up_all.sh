#!/usr/bin/env bash
# The cut-outs re-made 2026-11-16 ("up": fins and bodies never upside down or on their side, their square inside the
# photo; flukes as before) and the student's fingerprints of them, next to the first set, for every day that has one:
#   DAY/parts_up   (cluster/parts.py --pose up ..., same instances as DAY/parts_hr)
#   DAY/cluster_up (cluster_day.py channels: channels.npz + embeddings.npz of pair_student_g)
# DAY/parts_hr and DAY/cluster_v3 are left as they are (the safe clusters and link scores were built from them).
#   scripts/parts_up_all.sh [DAY ...]        default: every day with parts_hr; a finished day is skipped
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; export PYTHONPATH=.
DAYS=${*:-$(ls experiments/days_all | grep -E '^20[0-9]{2}-[0-9]{2}-[0-9]{2}$')}
for d in $DAYS; do
  OUT=experiments/days_all/$d
  [ -f "$OUT/parts_hr/index.jsonl" ] || { echo "== $d: no parts_hr, skipped"; continue; }
  if [ ! -f "$OUT/parts_up/index.jsonl" ]; then
    echo "== $d: parts_up $(date +%T)"
    $PY -m model_fine_tuning.cluster.parts --index "$OUT/images/index.jsonl" --canon-dir "$OUT/cse" --out "$OUT/parts_up" --size 1536 --workers "${HR_WORKERS:-10}" \
        --fluke-min-fraction 0.3 --fin-min-fraction 0.03 --instances --pose up --mirror side --level-max 30 --frame fin=fit,body=fit 2>&1 | grep -v -i "warn\|_check_multiprocessing" | tail -2
  fi
  if [ -f "$OUT/parts_up/index.jsonl" ] && [ ! -f "$OUT/cluster_up/embeddings.npz" ]; then
    $PY -m model_fine_tuning.cluster.cluster_day channels --day-dir "$OUT" --out "$OUT/cluster_up" --parts-tag parts_up \
        --sources id:experiments/pair_student_g/checkpoints/best.pt --local "" --candidates 0 2>&1 | grep -v -i "warn" | tail -2
  fi
done
echo "PARTS_UP_ALLDONE $(date +%T)"
