#!/usr/bin/env bash
# Full body-coordinate + pose fine-tune. Usage: train_bodycoords.sh NAME [extra train.py args]
set -euo pipefail
cd "$(dirname "$0")/../.."
NAME="${1:?experiment name}"; shift
mkdir -p experiments
.venv/bin/python -m model_fine_tuning.bodycoords.train --name "$NAME" "$@" 2>&1 | tee "experiments/$NAME.stdout"
