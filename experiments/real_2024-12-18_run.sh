#!/usr/bin/env bash
set -euo pipefail
cd /home/sat3737/NOAA_whales/Dinov3-WhaleID
SRC="/home/sat3737/NOAA_whales/test_data_do_not_use/2024-12-18"
OUT=experiments/real_2024-12-18
.venv/bin/python test_scripts/python/convert_raw.py "$SRC" $OUT/images --long-side 2560
echo "== predict"; date
.venv/bin/python -m model_fine_tuning.bodycoords.predict --seg-checkpoint experiments/seg_r02_ft4/checkpoints/best.pt \
    --checkpoint experiments/body_r02_ft4/checkpoints/best.pt --images $OUT/images --out $OUT/pred_r02 --max-side 1280
echo "== done"; date
