#!/usr/bin/env bash
# Switch prep (2026-11-21): the live days' safe/clusters.json carry each cluster's catalogue animals ("catalog"), because
# production ran `safe_tracklets cluster ... --labels labels_all_days.json`; run_pipeline.sh (written for the unlabelled
# test day) passes no labels. /bursts reads "catalog" for its review list. This re-runs the same clustering with the
# labels on every finished shadow day that has catalogue labels (labels annotate only: the check below requires the
# clusters to be identical apart from "catalog" and the report's metrics) and waits for days still being made.
#   experiments/test_day/shadow_catalog_fix.sh SHADOW_ROOT
set -uo pipefail
cd "$(dirname "$0")/../.."
SR="${1:?shadow root}"; PY=.venv/bin/python; export PYTHONPATH=.; LAB=model_fine_tuning/splits/labels_all_days.json
while true; do
  fin=0; grep -q "SHADOW_ROOT_DONE\|^STOP\|^FAIL" "$SR/logs/driver.log" && fin=1
  for M in "$SR"/markers/*.done; do
    DAY=$(basename "$M" .done); D="$SR/days/$DAY"; [ -f "$D/safe/clusters.json" ] || continue
    [ -f "$SR/markers/$DAY.catalog" ] && continue
    $PY -c "import json,sys; sys.exit(0 if json.load(open('$LAB'))['days'].get('$DAY',{}).get('photos') else 1)" || { touch "$SR/markers/$DAY.catalog"; continue; }
    cp "$D/safe/clusters.json" "$SR/logs/clusters_nolabels_$DAY.json"
    $PY -m model_fine_tuning.cluster.safe_tracklets cluster --day-dir "$D" --model experiments/safe_clusters/safe_model.pkl --threshold 0.995 --strict --cost 1.5 --skip 3 --filter --labels $LAB 2>&1 | tail -1
    $PY - "$SR/logs/clusters_nolabels_$DAY.json" "$D/safe/clusters.json" <<'PYEOF' && touch "$SR/markers/$DAY.catalog"
import json, sys
a, b = (json.load(open(f)) for f in sys.argv[1:3])
strip = lambda z: [{k: v for k, v in c.items() if k != "catalog"} for c in z["clusters"]]
same = strip(a) == strip(b) and a.get("suggestions") == b.get("suggestions")
print(sys.argv[2].split("/")[-3], "clusters identical apart from catalog:", same, "| clusters with catalog:", sum(1 for c in b["clusters"] if c.get("catalog")))
sys.exit(0 if same else 1)
PYEOF
  done
  [ $fin = 1 ] && break
  sleep 60
done
echo CATALOG_FIX_DONE $(date +%T)
