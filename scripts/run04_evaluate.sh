#!/usr/bin/env bash
# Evaluate the run04 fine-tunes (both variants, see run04_train.sh): segmentation and canonical-surface model on the
# synthetic test split (oracle and predicted crops, with the variant's own segmentation model; the boundary scores are
# split by the contour label into open-air silhouette vs seams), the real photos of 2024-12-18 (prediction, label-free
# outline metric, side-flip rate against the pre-canon diagnostic); the run03 models on the same run04 test split (the
# like-for-like synthetic comparison, eval_run04_test under their experiment dirs); then the Markdown report.
# Every step is skipped when its output exists.
#
#   scripts/run04_evaluate.sh /path/to/run04/dataset        (R04_SUFFIX, R04_VARIANTS as for run04_train.sh;
#   R04_REAL: the real-photo folder, R04_EVAL_EXTRA: extra flags for the synthetic evaluations, e.g. --max-images 8)
set -euo pipefail
cd "$(dirname "$0")/.."
DATASET="$(realpath -m "${1:?usage: $0 RUN_DIR/dataset}")"
RUN="$(dirname "$DATASET")"
SUFFIX="${R04_SUFFIX:-}"
SPLIT="model_fine_tuning/splits/run04${SUFFIX}.json"
REAL="${R04_REAL:-experiments/real_2024-12-18}"
PY=.venv/bin/python
EVAL_EXTRA="${R04_EVAL_EXTRA:-}"
export WHALE_RUN_DIR="$RUN"

eval_models() {   # $1 segmentation experiment, $2 canon experiment, $3 eval dir name (eval_test for the run04 models)
    local SEGX="$1" NAME="$2" TAG="$3"
    local SEG="experiments/$1/checkpoints/best.pt" CANON="experiments/$2/checkpoints/best.pt"
    if [ ! -f "$SEG" ] || [ ! -f "$CANON" ]; then echo "== $NAME: no checkpoints yet, skipped"; return 0; fi
    echo "== $NAME: segmentation on the test split"; date
    [ -f "experiments/$SEGX/$TAG/metrics.json" ] || $PY -m model_fine_tuning.segmentation.evaluate --checkpoint "$SEG" --run-dir "$RUN" \
        --split-file "$SPLIT" --split test --out "experiments/$SEGX/$TAG" $EVAL_EXTRA > "experiments/$SEGX.$TAG.stdout" 2>&1
    echo "== $NAME: canonical surface on the test split"; date
    [ -f "experiments/$NAME/$TAG/metrics.json" ] || $PY -m model_fine_tuning.canon.evaluate --checkpoint "$CANON" --seg-checkpoint "$SEG" \
        --run-dir "$RUN" --split-file "$SPLIT" --split test --out "experiments/$NAME/$TAG" $EVAL_EXTRA > "experiments/$NAME.$TAG.stdout" 2>&1
}

eval_real() {     # $1 segmentation experiment, $2 canon experiment
    local SEG="experiments/$1/checkpoints/best.pt" CANON="experiments/$2/checkpoints/best.pt" NAME="$2"
    if [ ! -f "$SEG" ] || [ ! -f "$CANON" ]; then return 0; fi
    echo "== $NAME: real photos: predict"; date
    [ -f "$REAL/canon_$NAME/points.jsonl" ] || $PY -m model_fine_tuning.canon.predict --seg-checkpoint "$SEG" --checkpoint "$CANON" \
        --images "$REAL/images" --out "$REAL/canon_$NAME" --max-side 1280 > "$REAL/canon_$NAME.log" 2>&1
    echo "== $NAME: real photos: outline metric"; date
    [ -f "$REAL/outline_metric_$NAME.json" ] || $PY -m model_fine_tuning.canon.outline_metric --canon-dir "$REAL/canon_$NAME" \
        --images "$REAL/images" --timestamps "$REAL/timestamps.json" --out "$REAL/outline_metric_$NAME.json" > "$REAL/outline_metric_$NAME.log" 2>&1
    echo "== $NAME: real photos: outline metric, direct rotation"; date
    [ -f "$REAL/outline_metric_${NAME}_direct.json" ] || $PY -m model_fine_tuning.canon.outline_metric --canon-dir "$REAL/canon_$NAME" --rotation direct \
        --images "$REAL/images" --timestamps "$REAL/timestamps.json" --out "$REAL/outline_metric_${NAME}_direct.json" > "$REAL/outline_metric_${NAME}_direct.log" 2>&1
    echo "== $NAME: real photos: side-flip rate"; date
    [ -f "$REAL/compare_$NAME/summary.json" ] || $PY test_scripts/python/compare_real_before_after.py "$REAL/side_diag/real_side.json" "$REAL/canon_$NAME" "$REAL/compare_$NAME" \
        > "$REAL/compare_$NAME.log" 2>&1 || true
}

for variant in ${R04_VARIANTS:-cw plain}; do
    case "$variant" in
        cw)    eval_models "seg_r04_cw${SUFFIX}"  "canon_r04_cw${SUFFIX}"  eval_test; eval_real "seg_r04_cw${SUFFIX}"  "canon_r04_cw${SUFFIX}" ;;
        plain) eval_models "seg_r04_ft4${SUFFIX}" "canon_r04_ft4${SUFFIX}" eval_test; eval_real "seg_r04_ft4${SUFFIX}" "canon_r04_ft4${SUFFIX}" ;;
        *) echo "unknown variant: $variant"; exit 1 ;;
    esac
done

echo "== run03 models on the run04 test split"; date
eval_models seg_r03_ft4 canon_r03_ft4 "eval_run04${SUFFIX}_test"

if [ -d "$REAL/canon_canon_r03_ft4" ] && [ ! -f "$REAL/outline_metric_canon_r03_ft4_direct.json" ]; then
    echo "== run03 baseline: real photos: outline metric, direct rotation"; date
    $PY -m model_fine_tuning.canon.outline_metric --canon-dir "$REAL/canon_canon_r03_ft4" --rotation direct --images "$REAL/images" \
        --timestamps "$REAL/timestamps.json" --out "$REAL/outline_metric_canon_r03_ft4_direct.json" > "$REAL/outline_metric_canon_r03_ft4_direct.log" 2>&1 || true
fi

echo "== report"; date
$PY scripts/run04_report.py --suffix "$SUFFIX" --real "$REAL" --out "experiments/run04${SUFFIX}_report.md"
echo "== evaluate done"; date
