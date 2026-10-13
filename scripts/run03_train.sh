#!/usr/bin/env bash
# Retrain the synthetic fine-tunes on a run03 factory dataset: the segmentation model, then the
# canonical-surface (body-coordinate) model. Idempotent per stage (a finished run's log has a
# "done" event). Called by Whale-Image-Data-Factory/tools/run03_pipeline.sh with the dataset dir.
#
#   scripts/run03_train.sh /path/to/run03/dataset
#
# Environment: R03_SUFFIX (experiment name suffix, e.g. _smoke for a test), R03_EPOCHS_SEG (30),
# R03_EPOCHS_CANON (40), R03_MIN_VISIBLE_PX (800), R03_EXTRA (extra trainer flags, e.g. --max-steps 5).
set -euo pipefail
cd "$(dirname "$0")/.."
DATASET="$(realpath -m "${1:?usage: $0 RUN_DIR/dataset}")"
RUN="$(dirname "$DATASET")"
SUFFIX="${R03_SUFFIX:-}"
SPLIT="model_fine_tuning/splits/run03${SUFFIX}.json"
SEG="seg_r03_ft4${SUFFIX}"
CANON="canon_r03_ft4${SUFFIX}"
PY=.venv/bin/python
export WHALE_RUN_DIR="$RUN"

finished() { [ -f "experiments/$1/log.jsonl" ] && grep -q '"event": "done"' "experiments/$1/log.jsonl"; }

echo "== split ($SPLIT)"; date
[ -f "$SPLIT" ] || $PY -m model_fine_tuning.common.splits make --run-dir "$RUN" --out "$SPLIT" \
    --min-visible-px "${R03_MIN_VISIBLE_PX:-800}" --drop-flagged

echo "== train $SEG"; date
finished "$SEG" || $PY -m model_fine_tuning.segmentation.train --name "$SEG" --run-dir "$RUN" --split "$SPLIT" \
    --unfreeze-blocks 4 --epochs "${R03_EPOCHS_SEG:-30}" --batch-size 8 --workers 6 ${R03_EXTRA:-} > "experiments/$SEG.stdout" 2>&1

echo "== train $CANON"; date
finished "$CANON" || $PY -m model_fine_tuning.canon.train --name "$CANON" --run-dir "$RUN" --split "$SPLIT" \
    --unfreeze-blocks 4 --epochs "${R03_EPOCHS_CANON:-40}" --batch-size 8 --workers 6 --min-visible 300 ${R03_EXTRA:-} > "experiments/$CANON.stdout" 2>&1

echo "== train done"; date
