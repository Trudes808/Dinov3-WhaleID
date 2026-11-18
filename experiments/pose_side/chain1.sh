#!/bin/bash
# pose_side first experiment, run under the shared GPU lock as ONE job:
#   setsid nohup flock /var/tmp/sat3737-parts-cache/gpu_train.lock bash experiments/pose_side/chain1.sh > /var/tmp/sat3737-parts-cache/pose_side/chain1.log 2>&1 &
# marker: /var/tmp/sat3737-parts-cache/pose_side/chain1.done
set -e
cd /home/sat3737/NOAA_whales/Dinov3-WhaleID
export PYTHONPATH=.
PY=.venv/bin/python
C=/var/tmp/sat3737-parts-cache/pose_side
O=experiments/pose_side
date
$PY -m model_fine_tuning.pose_side.train --out $C/A --steps 5000
$PY -m model_fine_tuning.pose_side.predict --head $C/A/head.pt --out $C/pred_A.json --keys clicks:2025-01-02,clicks:2025-03-18,sheet:2025-01-15:4,sheet:2025-02-27:4,sheet:2025-03-14:4
$PY -m model_fine_tuning.pose_side.train --out $C/B_0102 --init $C/A/head.pt --real-day 2025-01-02 --steps 800 --lr 3e-4 --n-synth-eval 200
$PY -m model_fine_tuning.pose_side.predict --head $C/B_0102/head.pt --out $C/pred_B_0102.json --keys clicks:2025-03-18
$PY -m model_fine_tuning.pose_side.train --out $C/B_0318 --init $C/A/head.pt --real-day 2025-03-18 --steps 800 --lr 3e-4 --n-synth-eval 200
$PY -m model_fine_tuning.pose_side.predict --head $C/B_0318/head.pt --out $C/pred_B_0318.json --keys clicks:2025-01-02
$PY -m model_fine_tuning.pose_side.train --out $C/B_both --init $C/A/head.pt --real-day 2025-01-02,2025-03-18 --steps 800 --lr 3e-4 --n-synth-eval 200
$PY -m model_fine_tuning.pose_side.predict --head $C/B_both/head.pt --out $C/pred_B_both.json --keys sheet:2025-01-15:4,sheet:2025-02-27:4,sheet:2025-03-14:4
date
touch $C/chain1.done
