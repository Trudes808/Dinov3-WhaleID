#!/usr/bin/env bash
# Signatures (token distinctiveness + edge shape) for the eval + val days -> experiments/days_all/<day>/signature (web app).
#   nohup setsid scripts/signatures_all.sh > experiments/signatures_all.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; export PYTHONPATH=.
for d in 2025-02-27 2024-12-18 2025-01-15 2025-03-14 2025-02-19 2025-03-21 2024-12-26; do
  echo "== $d"; date
  $PY -m model_fine_tuning.cluster.signature --day-dir experiments/days_all/$d --features experiments/days_all/$d/dino_hr/features.npz \
      --labels model_fine_tuning/splits/labels_all_days.json --out experiments/days_all/$d/signature --heat --center --shape 2>&1 | grep -v -i warn | grep "AUC\|shapes\|Traceback\|Error\|tracklets"
done
echo "SIGNATURES_ALLDONE"; date
