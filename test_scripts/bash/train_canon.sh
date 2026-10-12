#!/usr/bin/env bash
# Canonical-surface fine-tune on run02. Usage: train_canon.sh NAME [extra train.py args]
set -euo pipefail
cd "$(dirname "$0")/../.."
NAME="${1:?experiment name}"; shift
RUN="${WHALE_RUN02_DIR:-$HOME/NOAA_whales/Whale-Image-Data-Factory/data/renders/run02}"
mkdir -p experiments
.venv/bin/python -m model_fine_tuning.canon.train --name "$NAME" --run-dir "$RUN" --split model_fine_tuning/splits/run02.json "$@" 2>&1 | tee "experiments/$NAME.stdout"
