#!/usr/bin/env bash
# A few optimisation steps of both trainers on a handful of images, into a scratch folder.
# Catches broken data paths / shapes / checkpoints before a long run. ~1 minute on GPU.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${1:-/tmp/whaleid_smoke}"
rm -rf "$OUT"
.venv/bin/python -m model_fine_tuning.segmentation.train --name smoke_seg --out-root "$OUT" --max-train 32 --max-val 8 --max-steps 12 --batch-size 4 --workers 2 --warmup-steps 2
.venv/bin/python -m model_fine_tuning.bodycoords.train --name smoke_body --out-root "$OUT" --max-train 32 --max-val 8 --max-steps 12 --batch-size 4 --workers 2 --warmup-steps 2
.venv/bin/python -m model_fine_tuning.segmentation.evaluate --checkpoint "$OUT/smoke_seg/checkpoints/best.pt" --split val --max-images 8 --overlays 4
.venv/bin/python -m model_fine_tuning.bodycoords.evaluate --checkpoint "$OUT/smoke_body/checkpoints/best.pt" --seg-checkpoint "$OUT/smoke_seg/checkpoints/best.pt" --split val --max-images 8 --overlays 4
echo "smoke OK -> $OUT"
