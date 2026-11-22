#!/usr/bin/env bash
# Render-side and synthetic-val scoring of W3c / W3d (now) and W3e (after w3e.done): w3e_scores.py pred + synth per
# model under the GPU lock, then the CPU scoring. Marker /var/tmp/sat3737-parts-cache/parts_weak/w3e_scores.done
set -u
REPO=/home/sat3737/NOAA_whales/Dinov3-WhaleID
cd "$REPO"
PY="$REPO/.venv/bin/python"
LOCK="flock /var/tmp/sat3737-parts-cache/gpu_train.lock"
export PYTHONPATH="$REPO"
C=/var/tmp/sat3737-parts-cache/parts_weak
S=experiments/parts_tri_expand/w3e_scores.py
one() {
  local NAME=$1
  local CK=$C/runs/$NAME/checkpoints/best.pt
  [ -f /home/sat3737/NOAA_whales/moonshot_cache/parts_tri_expand/render_parts_hr/$NAME/r06r_check21_windows.json ] || $LOCK $PY $S pred --model $NAME --checkpoint $CK || echo "pred $NAME failed"
  [ -f /home/sat3737/NOAA_whales/moonshot_cache/parts_tri_expand/synth_val/$NAME.json ] || $LOCK $PY $S synth --model $NAME --checkpoint $CK || echo "synth $NAME failed"
}
for N in parts_seg_r05_w3c parts_seg_r05_w3cs1 parts_seg_r05_w3d parts_seg_r05_w3ds1; do echo "$(date +%T) $N"; one $N; done
until [ -f $C/w3e.done ] || [ -f $C/w3e.failed ]; do sleep 120; done
for N in parts_seg_r05_w3e parts_seg_r05_w3es1; do echo "$(date +%T) $N"; [ -f $C/runs/$N/checkpoints/best.pt ] && one $N; done
$PY $S score --models parts_seg_r05_w3c parts_seg_r05_w3cs1 parts_seg_r05_w3d parts_seg_r05_w3ds1 parts_seg_r05_w3e parts_seg_r05_w3es1
date +%T > $C/w3e_scores.done
