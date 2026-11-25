#!/usr/bin/env bash
# As arm_chain.sh, but for several split arms of ONE day in ONE GPU-lock hold (the mini GPU steps take seconds; the lock
# queue is the cost). FUSED_CPU=1 (default): the mini steps run on the CPU instead (cpu_steps.py), no lock at all. Per day: features + prep per arm (CPU), one lock hold for every arm's mini channels / instances,
# then merge + not-whale per arm.
#   experiments/fused_instances/arms_day.sh DAY ARM_DAYS [ARM_DAYS ...]
set -uo pipefail
cd "$(dirname "$0")/../.."
day="$1"; shift; PY=.venv/bin/python; export PYTHONPATH=.; STUDENT=experiments/pair_student_g/checkpoints/best.pt
SRC="${FUSED_SRC-/home/sat3737/NOAA_whales/moonshot_cache/switch_root/days}"
case "$day" in 2024-12-18|2024-12-26|2025-02-19|2025-03-21) [ "${FUSED_EVAL_OK-}" = 1 ] || { echo "eval day $day refused"; exit 1; };; esac
t0=$(date +%s); GPU=""
for R in "$@"; do
  D="$R/$day"; M="$R/_mini/$day"
  [ -f "$D/safe/photos.json" ] || $PY -m model_fine_tuning.cluster.safe_clusters features --day-dir "$D" 2>&1 | grep -v -i warn | tail -1
  n=$($PY -c "import json;print(len(json.load(open('$D/split_info.json'))['split_stems']))")
  if [ ! -f "$D/safe/inst_feat.npz" ] && [ "$n" -gt 0 ]; then
    rm -rf "$M"; $PY experiments/fused_instances/gpu_mini.py prep --arm "$R" --day "$day"
    if [ "${FUSED_CPU-1}" = 1 ]; then          # default since 2026-11-23 00:15: the mini steps on the CPU (cpu_steps.py), no lock
      CUDA_VISIBLE_DEVICES="" $PY experiments/fused_instances/cpu_steps.py "$M" 2>&1 | grep -v -i warn | tail -1
    else
      GPU="$GPU $PY -m model_fine_tuning.cluster.cluster_day channels --day-dir '$M' --out '$M/cluster_v3' --sources id:$STUDENT --local '' --candidates 0 2>&1 | grep -v -i warn | tail -1;"
      GPU="$GPU $PY -m model_fine_tuning.cluster.link_scores instances --day-dir '$M' 2>&1 | grep -v -i warn | tail -1;"
    fi
  fi
done
[ -n "$GPU" ] && flock /var/tmp/sat3737-parts-cache/gpu_train.lock bash -c "$GPU"
for R in "$@"; do
  D="$R/$day"
  [ -f "$D/safe/inst_feat.npz" ] || $PY experiments/fused_instances/gpu_mini.py merge --arm "$R" --day "$day" --src "$SRC"
  [ -f "$D/safe/not_whale.json" ] || $PY -m model_fine_tuning.cluster.link_scores not-whale --day-dir "$D" 2>&1 | grep -v -i warn | tail -1
done
echo "ARMS_DONE $day $(( $(date +%s) - t0 )) s"
