#!/usr/bin/env bash
# The live new-day chain for an unsorted test day, into an untracked day folder (no catalogue, no labels passed).
# Steps and arguments are the ones in the READMEs (Dinov3-WhaleID, Whale-Review-Web) and scripts/parts_*_all.sh;
# every GPU step runs under the shared lock. Skips a step whose output exists.
#   experiments/test_day/run_pipeline.sh /home/sat3737/NOAA_whales/moonshot_cache/test_days/2026-04-07/days/2026-04-07
# Before this: test_scripts/bash/process_day_v2.sh SRC OUT_ROOT (images, canon = seg_r04_ft4 mask, cse, parts, timestamps).
set -uo pipefail
cd "$(dirname "$0")/../.."
D="${1:?day dir}"; PY=.venv/bin/python; export PYTHONPATH=.; LOCK="${GPU_LOCK-flock /var/tmp/sat3737-parts-cache/gpu_train.lock}"   # GPU_LOCK="": the caller holds the lock
STUDENT=experiments/pair_student_g/checkpoints/best.pt
step() { echo "== $(date +%T) $1"; }
q() { grep -v -i "warn\|_check_multiprocessing\|nanmean"; }
[ -f "$D/parts_hr/index.jsonl" ] || { step parts_hr; $PY -m model_fine_tuning.cluster.parts --index "$D/images/index.jsonl" --canon-dir "$D/cse" --out "$D/parts_hr" --size 1536 --workers 12 --fluke-min-fraction 0.3 --fin-min-fraction 0.03 --instances 2>&1 | q | tail -2; }
[ -f "$D/cluster_v3/embeddings.npz" ] || { step "cluster_v3 student"; $LOCK $PY -m model_fine_tuning.cluster.cluster_day channels --day-dir "$D" --out "$D/cluster_v3" --sources id:$STUDENT --local "" --candidates 0 2>&1 | q | tail -2; }
[ -f "$D/safe/photos.json" ] || { step "safe_clusters features"; $PY -m model_fine_tuning.cluster.safe_clusters features --day-dir "$D" 2>&1 | q | tail -2; }
[ -f "$D/safe/inst_feat.npz" ] || { step "link_scores instances"; $LOCK $PY -m model_fine_tuning.cluster.link_scores instances --day-dir "$D" 2>&1 | q | tail -2; }
[ -f "$D/safe/not_whale.json" ] || { step "not-whale"; $PY -m model_fine_tuning.cluster.link_scores not-whale --day-dir "$D" 2>&1 | q | tail -2; }
[ -f "$D/safe/clusters.json" ] || { step "safe clusters"; $PY -m model_fine_tuning.cluster.safe_tracklets cluster --day-dir "$D" --model experiments/safe_clusters/safe_model.pkl --threshold 0.995 --strict --cost 1.5 --skip 3 --filter --guard-fused 2>&1 | q | tail -3; }
[ -f "$D/safe/view_feat.npz" ] || { step "view_class features"; $LOCK $PY -m model_fine_tuning.cluster.view_class features --day-dir "$D" 2>&1 | q | tail -2; }
[ -f "$D/safe/view_class.json" ] || { step "view_class predict"; $PY -m model_fine_tuning.cluster.view_class predict --day-dir "$D" 2>&1 | q | tail -2; }
[ -f "$D/safe/view_match.npz" ] || { step "view_match embed"; $PY -m model_fine_tuning.cluster.view_match embed --day-dir "$D" 2>&1 | q | tail -2; }
[ -f "$D/safe/view_match_hwb6.npz" ] || { step "fin_hw embed"; $LOCK env OMP_NUM_THREADS=8 $PY -m model_fine_tuning.cluster.fin_hw embed --day-dir "$D" --models hwb6 2>&1 | q | tail -2; }
[ -f "$D/parts_up/index.jsonl" ] || { step parts_up; $PY -m model_fine_tuning.cluster.parts --index "$D/images/index.jsonl" --canon-dir "$D/cse" --out "$D/parts_up" --size 1536 --workers 10 \
    --fluke-min-fraction 0.3 --fin-min-fraction 0.03 --instances --pose up --mirror side --level-max 30 --frame fin=fit,body=fit 2>&1 | q | tail -2; }
[ -f "$D/cluster_up/embeddings.npz" ] || { step cluster_up; $LOCK $PY -m model_fine_tuning.cluster.cluster_day channels --day-dir "$D" --out "$D/cluster_up" --parts-tag parts_up --sources id:$STUDENT --local "" --candidates 0 2>&1 | q | tail -2; }
[ -f "$D/safe/windows.npz" ] || { step "link_scores windows"; $LOCK $PY -m model_fine_tuning.cluster.link_scores windows --day-dir "$D" 2>&1 | q | tail -2; }
[ -f "$D/safe/window_scores.npz" ] || { step "link_scores scores"; $LOCK $PY -m model_fine_tuning.cluster.link_scores scores --day-dir "$D" 2>&1 | q | tail -2; }
[ -f "$D/safe/link_scores.json" ] || { step "link_scores pairs"; $PY -m model_fine_tuning.cluster.link_scores pairs --day-dir "$D" 2>&1 | q | tail -2; }
echo "PIPELINE_DONE $(date +%T)"
