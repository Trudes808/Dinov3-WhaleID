#!/usr/bin/env bash
# Bendable mesh fits of a canon model's predictions on the run04 test split (fit_mesh --synthetic), scored against the truth.
#   scripts/fit_synthetic.sh canon_r04_ft4        (env: FIT_MESH, FIT_SUFFIX, FIT_N 48)
set -uo pipefail
cd "$(dirname "$0")/.."
n="$1"; RUN=$HOME/NOAA_whales/Whale-Image-Data-Factory/data/renders/run04; out="experiments/$n/fit_synthetic${FIT_SUFFIX:-}"; rm -rf "$out"
.venv/bin/python -m model_fine_tuning.canon.fit_mesh --synthetic --checkpoint "experiments/$n/checkpoints/best.pt" --run-dir "$RUN" \
    --split-file model_fine_tuning/splits/run04.json --split test --max-images "${FIT_N:-48}" --overlays "${FIT_N:-48}" --workers 6 \
    --out "$out" ${FIT_MESH:+--mesh "$FIT_MESH"} > "$out.log" 2>&1
echo "synthetic fit $n done: $(date -u +%T)"
