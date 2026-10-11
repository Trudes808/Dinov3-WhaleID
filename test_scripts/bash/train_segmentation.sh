#!/usr/bin/env bash
# Full segmentation fine-tune. Usage: train_segmentation.sh NAME [extra train.py args]
# e.g. train_segmentation.sh seg_vitb16_ft4 --unfreeze-blocks 4
set -euo pipefail
cd "$(dirname "$0")/../.."
NAME="${1:?experiment name}"; shift
mkdir -p experiments
.venv/bin/python -m model_fine_tuning.segmentation.train --name "$NAME" "$@" 2>&1 | tee "experiments/$NAME.stdout"
