#!/bin/bash
# Step 1: fin / fluke fingerprints of the existing 21-day students (and their init id_v4) on the 3 val days.
cd /home/sat3737/NOAA_whales/Dinov3-WhaleID
LOCK=/var/tmp/sat3737-parts-cache/gpu_train.lock
VAL=2025-01-15,2025-02-27,2025-03-14
for nc in G=pair_student_g R0s0=pair_student_v3_r0 R0s1=pair_student_v3_r0s1 R1=pair_student_v3_r1 RP6s0=pair_student_v3_rp6s0 RP6s1=pair_student_v3_rp6s1 idv4=id_v4; do
  n=${nc%%=*}; e=${nc#*=}
  [ -f /home/sat3737/NOAA_whales/moonshot_cache/student_curve/emb/$n/2025-03-14.npz ] && continue
  flock $LOCK env PYTHONPATH=. .venv/bin/python -W ignore experiments/moonshot_sizing/student_curve/embed.py --net $n=experiments/$e/checkpoints/best.pt --days $VAL
done
echo EXISTING_DONE $(date +%T)
