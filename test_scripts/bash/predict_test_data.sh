#!/usr/bin/env bash
# Run both models on the photos in test_data_do_not_use (or any folder).
# Usage: predict_test_data.sh SEG_EXP BODY_EXP [IMAGES_DIR] [OUT_DIR] [extra predict args, e.g. --focal-mm 400]
# If IMAGES_DIR is a factory run (has dataset/manifest.jsonl) the labelled evaluations run instead.
set -euo pipefail
cd "$(dirname "$0")/../.."
SEG="${1:?segmentation experiment name}"; BODY="${2:?bodycoords experiment name}"
IMAGES="${3:-${WHALE_TEST_DATA:-$HOME/NOAA_whales/test_data_do_not_use}}"
OUT="${4:-experiments/predictions_$(basename "$IMAGES")}"
shift $(( $# < 4 ? $# : 4 ))
if [ -f "$IMAGES/dataset/manifest.jsonl" ]; then
    .venv/bin/python -m model_fine_tuning.segmentation.evaluate --checkpoint "experiments/$SEG/checkpoints/best.pt" --run-dir "$IMAGES" --split all --out "$OUT/segmentation"
    .venv/bin/python -m model_fine_tuning.bodycoords.evaluate --checkpoint "experiments/$BODY/checkpoints/best.pt" --seg-checkpoint "experiments/$SEG/checkpoints/best.pt" --run-dir "$IMAGES" --split all --out "$OUT/bodycoords"
else
    .venv/bin/python -m model_fine_tuning.bodycoords.predict --seg-checkpoint "experiments/$SEG/checkpoints/best.pt" --checkpoint "experiments/$BODY/checkpoints/best.pt" --images "$IMAGES" --out "$OUT" "$@"
fi
