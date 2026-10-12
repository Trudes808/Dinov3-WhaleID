#!/usr/bin/env bash
set -uo pipefail
cd /home/sat3737/NOAA_whales/Dinov3-WhaleID
RUN=$HOME/NOAA_whales/Whale-Image-Data-Factory/data/renders/run02; SPLIT=model_fine_tuning/splits/run02.json
SEG=experiments/seg_r02_ft4/checkpoints/best.pt; CK=experiments/canon_r02_ft4/checkpoints/best.pt; PY=.venv/bin/python
echo "== evaluate v2"; date
$PY -m model_fine_tuning.canon.evaluate --checkpoint $CK --seg-checkpoint $SEG --run-dir $RUN --split-file $SPLIT --split test --out experiments/canon_r02_ft4/eval_test_v2 2>&1 | grep -v -i warn | tail -3
echo "== real photos v2"; date
$PY -m model_fine_tuning.canon.predict --seg-checkpoint $SEG --checkpoint $CK --images experiments/real_2024-12-18/images --out experiments/real_2024-12-18/canon_r02_ft4_v2 --max-side 1280 > experiments/real_2024-12-18/canon_r02_ft4_v2.log 2>&1
echo "== compare"; date
$PY test_scripts/python/compare_real_before_after.py experiments/real_2024-12-18/side_diag/real_side.json experiments/real_2024-12-18/canon_r02_ft4_v2 experiments/real_2024-12-18/canon_r02_ft4_v2_compare
echo "== done"; date
