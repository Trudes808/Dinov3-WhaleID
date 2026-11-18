#!/bin/bash
# second GPU round (prediction only, about 10 min): probability maps for CPU decoding of the fluke tips, and B_both on
# every instance of the validation days (for point-based fin cut-outs). Run as
#   setsid nohup flock /var/tmp/sat3737-parts-cache/gpu_train.lock bash experiments/pose_side/chain2.sh > /var/tmp/sat3737-parts-cache/pose_side/chain2.log 2>&1 &
set -e
cd /home/sat3737/NOAA_whales/Dinov3-WhaleID
export PYTHONPATH=.
PY=.venv/bin/python
C=/var/tmp/sat3737-parts-cache/pose_side
date
$PY -m model_fine_tuning.pose_side.predict --head $C/A/head.pt --out $C/pred2_A.json --keys clicks:2025-01-02,clicks:2025-03-18 --save-maps $C/maps_A
$PY -m model_fine_tuning.pose_side.predict --head $C/B_0102/head.pt --out $C/pred2_B_0102.json --keys clicks:2025-03-18 --save-maps $C/maps_Bx
$PY -m model_fine_tuning.pose_side.predict --head $C/B_0318/head.pt --out $C/pred2_B_0318.json --keys clicks:2025-01-02 --save-maps $C/maps_Bx
$PY -m model_fine_tuning.pose_side.predict --head $C/B_both/head.pt --out $C/pred2_Bboth_val.json --keys all:2025-01-15,all:2025-02-27,all:2025-03-14 --save-maps $C/maps_Bboth_val
date
touch $C/chain2.done
