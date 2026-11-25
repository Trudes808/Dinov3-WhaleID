#!/usr/bin/env bash
# The live chain after parts_hr, unchanged commands, on a split arm's day folder (split_arm.py made parts_hr):
# safe_clusters features (CPU), the two GPU commands (cluster_day channels with the student, link_scores instances) on a
# mini folder of the split photos only, in ONE lock hold per day (gpu_mini.py prep / merge with the shadow's rows for the
# other photos), then link_scores not-whale. The strict clustering itself is done by score.py. Skips finished steps.
#   experiments/fused_instances/arm_chain.sh ARM_DAYS [SRC_DAYS] -- DAY ...
set -uo pipefail
cd "$(dirname "$0")/../.."
R="$1"; shift; SRC=/home/sat3737/NOAA_whales/moonshot_cache/switch_root/days
[ "${1-}" != "--" ] && { SRC="$1"; shift; }; [ "${1-}" = "--" ] && shift
PY=.venv/bin/python; export PYTHONPATH=.; STUDENT=experiments/pair_student_g/checkpoints/best.pt
LOCK="flock /var/tmp/sat3737-parts-cache/gpu_train.lock"
for day in "$@"; do
  case "$day" in 2024-12-18|2024-12-26|2025-02-19|2025-03-21) [ "${FUSED_EVAL_OK-}" = 1 ] || { echo "eval day $day refused"; continue; };; esac
  D="$R/$day"; M="$R/_mini/$day"; t0=$(date +%s)
  [ -f "$D/safe/photos.json" ] || $PY -m model_fine_tuning.cluster.safe_clusters features --day-dir "$D" 2>&1 | grep -v -i warn | tail -1
  n=$($PY -c "import json;print(len(json.load(open('$D/split_info.json'))['split_stems']))")
  if [ ! -f "$D/safe/inst_feat.npz" ]; then
    if [ "$n" -gt 0 ]; then
      $PY experiments/fused_instances/gpu_mini.py prep --arm "$R" --day "$day"
      $LOCK bash -c "
        $PY -m model_fine_tuning.cluster.cluster_day channels --day-dir '$M' --out '$M/cluster_v3' --sources id:$STUDENT --local '' --candidates 0 2>&1 | grep -v -i warn | tail -1
        $PY -m model_fine_tuning.cluster.link_scores instances --day-dir '$M' 2>&1 | grep -v -i warn | tail -1"
    fi
    $PY experiments/fused_instances/gpu_mini.py merge --arm "$R" --day "$day" --src "$SRC"
  fi
  [ -f "$D/safe/not_whale.json" ] || $PY -m model_fine_tuning.cluster.link_scores not-whale --day-dir "$D" 2>&1 | grep -v -i warn | tail -1
  echo "CHAIN_DONE $day $(( $(date +%s) - t0 )) s"
done
