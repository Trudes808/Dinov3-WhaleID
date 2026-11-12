#!/usr/bin/env bash
# Instance-to-instance window scores and the cluster-pair table for every day whose window features exist and whose
# scores do not (the reference day uses its out-of-fold saliency: the where-to-look model was fitted on its boxes).
R=/home/sat3737/NOAA_whales/Dinov3-WhaleID; cd $R; export PYTHONPATH=$R OMP_NUM_THREADS=4
for d in $R/experiments/days_all/*/; do day=$(basename $d)
  [ -f $d/safe/windows.npz ] || continue
  if [ ! -f $d/safe/window_scores.npz ]; then
    if [ "$day" = "2025-01-02" ]; then $R/.venv/bin/python -m model_fine_tuning.cluster.link_scores scores --day-dir $d --saliency experiments/safe_clusters/where_to_look_oof_2025-01-02.npy
    else $R/.venv/bin/python -m model_fine_tuning.cluster.link_scores scores --day-dir $d; fi
  fi
  [ -f experiments/stage2_linker/tables/$day.pkl ] && [ experiments/stage2_linker/tables/$day.pkl -nt $d/safe/clusters.json ] || (cd experiments/stage2_linker && $R/.venv/bin/python tables.py $day 2>&1 | grep -v -i warn)
done
