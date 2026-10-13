#!/usr/bin/env bash
# Redo the run03 real-photo chain after the mask cleanup in canon.predict: predictions, outline metric,
# side-flip rate, atlas -> identity -> match -> clusters (swap_day_to_run03.sh), then restart the web app.
set -uo pipefail
cd /home/sat3737/NOAA_whales/Dinov3-WhaleID
PY=.venv/bin/python; export PYTHONPATH=.
R=experiments/real_2024-12-18; NAME=canon_r03_ft4; SEG=experiments/seg_r03_ft4/checkpoints/best.pt; CK=experiments/$NAME/checkpoints/best.pt
echo "== predict (cleaned masks)"; date
[ -d $R/canon_${NAME}_nocleanup ] || mv $R/canon_$NAME $R/canon_${NAME}_nocleanup
rm -rf $R/canon_$NAME
$PY -m model_fine_tuning.canon.predict --seg-checkpoint $SEG --checkpoint $CK --images $R/images --out $R/canon_$NAME --max-side 1280 > $R/canon_$NAME.log 2>&1
echo "== outline metric"; date
[ -f $R/outline_metric_${NAME}_nocleanup.json ] || mv $R/outline_metric_$NAME.json $R/outline_metric_${NAME}_nocleanup.json
rm -f $R/outline_metric_$NAME.json
$PY -m model_fine_tuning.canon.outline_metric --canon-dir $R/canon_$NAME --images $R/images --timestamps $R/timestamps.json --out $R/outline_metric_$NAME.json > $R/outline_metric_$NAME.log 2>&1
$PY test_scripts/python/compare_real_before_after.py $R/side_diag/real_side.json $R/canon_$NAME $R/compare_$NAME > $R/compare_$NAME.log 2>&1 || true
echo "== rebuild the matching chain"; date
rm -rf $R/atlas_v3 $R/atlas_v3_id $R/cluster_v3id
scripts/swap_day_to_run03.sh > $R/swap_run03.log 2>&1
echo "== restart the web app"; date
pid=$(ss -ltnp 'sport = :8765' | grep -o 'pid=[0-9]*' | cut -d= -f2 | head -1)
[ -n "$pid" ] && kill $pid && while kill -0 $pid 2>/dev/null; do sleep 1; done
cd /home/sat3737/NOAA_whales/Whale-Review-Web && nohup ./serve.sh --port 8765 > /tmp/whale_review_server.log 2>&1 < /dev/null &
sleep 20; curl -s http://127.0.0.1:8765/api/day | head -c 120; echo
$PY -c "print()" 2>/dev/null
cd /home/sat3737/NOAA_whales/Dinov3-WhaleID && $PY scripts/run03_report.py --name canon_r03_ft4 --seg seg_r03_ft4 --baseline canon_r02_ft4 --real $R --out experiments/run03_report.md > /dev/null
echo "== done"; date
