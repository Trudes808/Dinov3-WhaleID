#!/usr/bin/env bash
# CPU scoring of the pilot test with the unchanged W3c / W3e scripts on the OLD day root (pilot22_train/oldroot.py),
# outputs here under new names, then table.py.
#   bash experiments/pilot22_train/run_p22_compare.sh   (after p22_add.done; again after p22_only.done)
set -u
REPO=/home/sat3737/NOAA_whales/Dinov3-WhaleID
cd "$REPO"
PY="$REPO/.venv/bin/python"
export PYTHONPATH="$REPO"
C=/home/sat3737/NOAA_whales/moonshot_cache/pilot22/d
D=experiments/pilot22_train
M="parts_seg_r05_w3e parts_seg_r05_w3es1"
for N in parts_seg_p22add parts_seg_p22adds1 parts_seg_p22only parts_seg_p22onlys1; do
  [ -f /var/tmp/sat3737-parts-cache/pilot22/days_shadow/2025-01-15/$N/instances.json ] && M="$M $N"
done
echo "models: $M"
$PY -m model_fine_tuning.pilot22_train.oldroot compare --arms $M --out $D/compare_p22.json > $C/cmp_p22.out 2>&1 || echo "compare failed"
$PY -m model_fine_tuning.pilot22_train.oldroot finless --models $M --out $D/finless_p22.json > $C/fl_p22.out 2>&1 || echo "finless failed"
$PY -m model_fine_tuning.pilot22_train.oldroot predfill score --models $M --out ../pilot22_train/new_measures_p22.json > $C/nm_p22.out 2>&1 || echo "new measures failed"
$PY $D/table.py > $C/table_p22.out 2>&1 || echo "table failed"
date +%T > $C/p22_compare.done
