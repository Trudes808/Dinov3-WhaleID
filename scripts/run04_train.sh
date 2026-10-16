#!/usr/bin/env bash
# Retrain the synthetic fine-tunes on a run04 factory dataset, two variants in turn, each evaluated as soon as it
# is trained (scripts/run04_evaluate.sh, idempotent), so the first comparison with run03 is ready early:
#   cw     contour-weighted losses (seg_r04_cw, canon_r04_cw): the run04 contour label re-weights the pixels around
#          the mask's boundary, up at the whale's own silhouette, down at the waterline / foam / calf seams
#   plain  the run03 recipe unchanged on the run04 data (seg_r04_ft4, canon_r04_ft4): the ablation that separates
#          the data's effect from the weighting's
# Idempotent per stage (a finished run's log has a "done" event). Called by
# Whale-Image-Data-Factory/tools/run04_pipeline.sh with the dataset dir.
#
#   scripts/run04_train.sh /path/to/run04/dataset
#
# Environment: R04_SUFFIX (experiment name suffix, e.g. _smoke for a test), R04_VARIANTS ("cw plain"), R04_EPOCHS_SEG (30),
# R04_EPOCHS_CANON (40), R04_MIN_VISIBLE_PX (800), R04_EXTRA (extra trainer flags, e.g. --max-steps 5), R04_CONTOUR (the cw
# variant's weights, default "--contour-band 3 --contour-w-true 4 --contour-w-seam 0.3"), R04_CONTOUR_XYZ ("--contour-w-xyz 3"),
# R04_NO_EVAL (set to skip the per-variant evaluation).
set -euo pipefail
cd "$(dirname "$0")/.."
DATASET="$(realpath -m "${1:?usage: $0 RUN_DIR/dataset}")"
RUN="$(dirname "$DATASET")"
SUFFIX="${R04_SUFFIX:-}"
SPLIT="model_fine_tuning/splits/run04${SUFFIX}.json"
PY=.venv/bin/python
export WHALE_RUN_DIR="$RUN"
CONTOUR="${R04_CONTOUR:---contour-band 3 --contour-w-true 4 --contour-w-seam 0.3}"
CONTOUR_XYZ="${R04_CONTOUR_XYZ:---contour-w-xyz 3}"

finished() { [ -f "experiments/$1/log.jsonl" ] && grep -q '"event": "done"' "experiments/$1/log.jsonl"; }

echo "== split ($SPLIT)"; date
[ -f "$SPLIT" ] || $PY -m model_fine_tuning.common.splits make --run-dir "$RUN" --out "$SPLIT" \
    --min-visible-px "${R04_MIN_VISIBLE_PX:-800}" --drop-flagged

for variant in ${R04_VARIANTS:-cw plain}; do
    case "$variant" in
        cw)    SEG="seg_r04_cw${SUFFIX}";  CANON="canon_r04_cw${SUFFIX}";  SEG_FLAGS="$CONTOUR"; CANON_FLAGS="$CONTOUR $CONTOUR_XYZ" ;;
        plain) SEG="seg_r04_ft4${SUFFIX}"; CANON="canon_r04_ft4${SUFFIX}"; SEG_FLAGS="";         CANON_FLAGS="" ;;
        *) echo "unknown variant: $variant"; exit 1 ;;
    esac
    echo "== [$variant] train $SEG"; date
    finished "$SEG" || $PY -m model_fine_tuning.segmentation.train --name "$SEG" --run-dir "$RUN" --split "$SPLIT" \
        --unfreeze-blocks 4 --epochs "${R04_EPOCHS_SEG:-30}" --batch-size 8 --workers 6 $SEG_FLAGS ${R04_EXTRA:-} > "experiments/$SEG.stdout" 2>&1
    echo "== [$variant] train $CANON"; date
    finished "$CANON" || $PY -m model_fine_tuning.canon.train --name "$CANON" --run-dir "$RUN" --split "$SPLIT" \
        --unfreeze-blocks 4 --epochs "${R04_EPOCHS_CANON:-40}" --batch-size 8 --workers 6 --min-visible 300 $CANON_FLAGS ${R04_EXTRA:-} > "experiments/$CANON.stdout" 2>&1
    if [ -z "${R04_NO_EVAL:-}" ]; then
        echo "== [$variant] evaluate"; date
        R04_VARIANTS="$variant" scripts/run04_evaluate.sh "$DATASET" || echo "== [$variant] evaluate FAILED (the pipeline's evaluate stage retries it)"
    fi
done
echo "== train done"; date
