#!/bin/bash
# CPU follow-up of chain1.sh (waits for its marker, at most 24 h): scores, region transfer, sheets.
#   setsid nohup bash experiments/pose_side/after_chain1.sh > /var/tmp/sat3737-parts-cache/pose_side/after_chain1.log 2>&1 &
cd /home/sat3737/NOAA_whales/Dinov3-WhaleID
export PYTHONPATH=.
PY=.venv/bin/python
C=/var/tmp/sat3737-parts-cache/pose_side
for i in $(seq 1 1440); do [ -f $C/chain1.done ] && break; sleep 60; done
[ -f $C/chain1.done ] || { echo "chain1 never finished"; exit 1; }
$PY -m model_fine_tuning.pose_side.evaluate --out experiments/pose_side/results_chain1.json A=$C/pred_A.json Bx=$C/pred_B_0102.json+$C/pred_B_0318.json > $C/evaluate_chain1.txt 2>&1
$PY -m model_fine_tuning.pose_side.sheets --pred $C/pred_A.json --name A_val
$PY -m model_fine_tuning.pose_side.sheets --pred $C/pred_B_both.json --name Bboth_val
$PY -m model_fine_tuning.pose_side.sheets --pred $C/pred_A.json --name A_clickdays --days 2025-01-02,2025-03-18
touch $C/after_chain1.done
