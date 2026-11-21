#!/usr/bin/env bash
# Round 2: the test day through the live chain with the candidate mask (S6h) in place of seg_r04_ft4, into a new
# untracked day folder; then the candidate's own instance stage (instances_v2, join gap 0.35) on that folder.
#   experiments/test_day/run_s6.sh /home/sat3737/NOAA_whales/moonshot_cache/test_days/2026-04-07
# Images and timestamps are reused from round 1's day folder (symlink / copy); every GPU step under the shared lock.
set -uo pipefail
cd "$(dirname "$0")/../.."
R="${1:?test root}"; L=$R/days/2026-04-07; D=$R/round2/days_s6/2026-04-07
PY=.venv/bin/python; export PYTHONPATH=.; LOCK="flock /var/tmp/sat3737-parts-cache/gpu_train.lock"
CSE=experiments/cse_r04_ft2/checkpoints/best.pt
mkdir -p "$D"; [ -e "$D/images" ] || ln -s "$L/images" "$D/images"; [ -f "$D/timestamps.json" ] || cp "$L/timestamps.json" "$D/"
echo "== $(date +%T) canon (candidate mask)"; [ -f "$D/canon/points.jsonl" ] || $LOCK $PY experiments/test_day/s6_swap.py canon --images "$D/images" --out "$D/canon" > "$D/canon.log" 2>&1
echo "== $(date +%T) cse"; [ -f "$D/cse/points.jsonl" ] || $LOCK $PY -m model_fine_tuning.canon.predict_cse --checkpoint $CSE --canon-dir "$D/canon" --images "$D/images" --out "$D/cse" --overlays 0 --as-canon > "$D/cse.log" 2>&1
echo "== $(date +%T) parts"; [ -f "$D/parts/index.jsonl" ] || $PY -m model_fine_tuning.cluster.parts --index "$D/images/index.jsonl" --canon-dir "$D/cse" --out "$D/parts" --size 512 --workers 10 --fluke-min-fraction 0.3 --fin-min-fraction 0.03 2>&1 | grep -v -i "warn\|_check_multiprocessing" | tail -1
experiments/test_day/run_pipeline.sh "$D"
echo "== $(date +%T) instances_v2 (candidate, join gap 0.35)"; [ -f "$D/instances_v2/index.jsonl" ] || $LOCK $PY experiments/test_day/s6_swap.py v2 --day-dir "$D" > "$D/instances_v2.log" 2>&1
tail -2 "$D/instances_v2.log"
echo "S6_DONE $(date +%T)"
