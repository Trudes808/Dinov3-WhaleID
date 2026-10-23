#!/usr/bin/env bash
# Native-resolution re-segmentation of all fin / fluke charts (cluster/reseg.py) for the split days.
set -uo pipefail; cd "$(dirname "$0")/.."; PY=.venv/bin/python; export PYTHONPATH=.; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for d in $($PY -c "import json; d=json.load(open('model_fine_tuning/splits/days.json')); print(' '.join(d['val']+d['eval']+d['train']))"); do
  $PY -m model_fine_tuning.cluster.reseg --day-dir experiments/days_all/$d 2>&1 | grep -v -i warn
done; echo RESEG_ALLDONE
