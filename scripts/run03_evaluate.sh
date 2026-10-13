#!/usr/bin/env bash
# Evaluate the run03 fine-tunes: synthetic test split (oracle and predicted crops, with the run03
# segmentation model), the real photos of 2024-12-18 (prediction, label-free outline metric,
# side-flip rate against the pre-canon diagnostic), and a metrics report next to the baselines.
#
#   scripts/run03_evaluate.sh /path/to/run03/dataset        (R03_SUFFIX as for run03_train.sh)
set -euo pipefail
cd "$(dirname "$0")/.."
DATASET="$(realpath -m "${1:?usage: $0 RUN_DIR/dataset}")"
RUN="$(dirname "$DATASET")"
SUFFIX="${R03_SUFFIX:-}"
SPLIT="model_fine_tuning/splits/run03${SUFFIX}.json"
SEG="experiments/seg_r03_ft4${SUFFIX}/checkpoints/best.pt"
NAME="canon_r03_ft4${SUFFIX}"
CANON="experiments/$NAME/checkpoints/best.pt"
REAL=experiments/real_2024-12-18
PY=.venv/bin/python
export WHALE_RUN_DIR="$RUN"

echo "== synthetic test split"; date
[ -f "experiments/$NAME/eval_test/metrics.json" ] || $PY -m model_fine_tuning.canon.evaluate --checkpoint "$CANON" --seg-checkpoint "$SEG" \
    --run-dir "$RUN" --split-file "$SPLIT" --split test > "experiments/$NAME.eval.stdout" 2>&1

echo "== real photos: predict"; date
[ -f "$REAL/canon_$NAME/points.jsonl" ] || $PY -m model_fine_tuning.canon.predict --seg-checkpoint "$SEG" --checkpoint "$CANON" \
    --images "$REAL/images" --out "$REAL/canon_$NAME" --max-side 1280 > "$REAL/canon_$NAME.log" 2>&1

echo "== real photos: outline metric"; date
[ -f "$REAL/outline_metric_$NAME.json" ] || $PY -m model_fine_tuning.canon.outline_metric --canon-dir "$REAL/canon_$NAME" \
    --images "$REAL/images" --timestamps "$REAL/timestamps.json" --out "$REAL/outline_metric_$NAME.json" > "$REAL/outline_metric_$NAME.log" 2>&1

echo "== real photos: side-flip rate"; date
[ -f "$REAL/compare_$NAME/summary.json" ] || $PY test_scripts/python/compare_real_before_after.py "$REAL/side_diag/real_side.json" "$REAL/canon_$NAME" "$REAL/compare_$NAME" \
    > "$REAL/compare_$NAME.log" 2>&1 || true

echo "== report"; date
$PY scripts/run03_report.py --name "$NAME" --seg "seg_r03_ft4${SUFFIX}" --baseline canon_r02_ft4 --real "$REAL" --out "experiments/run03${SUFFIX}_report.md"
echo "== evaluate done"; date
