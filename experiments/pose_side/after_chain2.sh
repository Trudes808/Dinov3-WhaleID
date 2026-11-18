#!/bin/bash
# CPU follow-up of chain2.sh (waits for its marker, at most 24 h): fluke tips decoded as two peaks from the saved maps
# ('model' and 'image' naming), scored against the clicks and in the region transfer; point-based dorsal cut-outs
# (finP, fin + hump triangle) for the validation days from B_both.
#   setsid nohup bash experiments/pose_side/after_chain2.sh > /var/tmp/sat3737-parts-cache/pose_side/after_chain2.log 2>&1 &
cd /home/sat3737/NOAA_whales/Dinov3-WhaleID
export PYTHONPATH=.
PY=.venv/bin/python
C=/var/tmp/sat3737-parts-cache/pose_side
for i in $(seq 1 1440); do [ -f $C/chain2.done ] && break; sleep 60; done
[ -f $C/chain2.done ] || { echo "chain2 never finished"; exit 1; }
$PY -m model_fine_tuning.pose_side.decode_tips $C/maps_A $C/dec_A
$PY -m model_fine_tuning.pose_side.decode_tips $C/maps_Bx $C/dec_Bx
$PY -m model_fine_tuning.pose_side.evaluate --out experiments/pose_side/results_chain2.json \
    A2=$C/dec_A_model.json A2img=$C/dec_A_image.json Bx2=$C/dec_Bx_model.json Bx2img=$C/dec_Bx_image.json > $C/evaluate_chain2.txt 2>&1
$PY -m model_fine_tuning.pose_side.make_finP $C/pred2_Bboth_val.json --days 2025-01-15,2025-02-27,2025-03-14 --set finP --workers 12
touch $C/after_chain2.done
