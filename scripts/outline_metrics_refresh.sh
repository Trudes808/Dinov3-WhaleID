#!/usr/bin/env bash
# Recompute the real-photo outline metrics (PnP and direct rotation) for the given canon prediction names with the
# full mesh (the silhouette used to be rendered with every third face).
set -uo pipefail
cd "$(dirname "$0")/.."
REAL=experiments/real_2024-12-18
for n in "$@"; do
  .venv/bin/python -m model_fine_tuning.canon.outline_metric --canon-dir "$REAL/canon_$n" --images "$REAL/images" --timestamps "$REAL/timestamps.json" --out "$REAL/outline_metric_$n.json" > "$REAL/outline_metric_$n.log" 2>&1
  .venv/bin/python -m model_fine_tuning.canon.outline_metric --canon-dir "$REAL/canon_$n" --rotation direct --images "$REAL/images" --timestamps "$REAL/timestamps.json" --out "$REAL/outline_metric_${n}_direct.json" > "$REAL/outline_metric_${n}_direct.log" 2>&1
  echo "outline $n refreshed"
done
