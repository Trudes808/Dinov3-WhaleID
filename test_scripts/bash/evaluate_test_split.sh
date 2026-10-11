#!/usr/bin/env bash
# Held-out test-split evaluation of both models. Usage: evaluate_test_split.sh SEG_EXP BODY_EXP [split]
set -euo pipefail
cd "$(dirname "$0")/../.."
SEG="${1:?segmentation experiment name}"; BODY="${2:?bodycoords experiment name}"; SPLIT="${3:-test}"
.venv/bin/python -m model_fine_tuning.segmentation.evaluate --checkpoint "experiments/$SEG/checkpoints/best.pt" --split "$SPLIT"
.venv/bin/python -m model_fine_tuning.bodycoords.evaluate --checkpoint "experiments/$BODY/checkpoints/best.pt" \
    --seg-checkpoint "experiments/$SEG/checkpoints/best.pt" --split "$SPLIT"
