#!/usr/bin/env bash
# Bendable mesh fits on the real photos for the given canon models (fit_mesh), one after the other.
#   scripts/fit_real_photos.sh canon_r04_ft4 canon_r03_ft4        (env: FIT_MESH to use another mesh OBJ, FIT_SUFFIX for the output name)
set -uo pipefail
cd "$(dirname "$0")/.."
for pid in $(pgrep -f "canon.fit_mesh --canon-dir"); do [ "$pid" != "$$" ] && kill "$pid" 2>/dev/null; done; sleep 2
REAL=experiments/real_2024-12-18
for n in "$@"; do
  out="$REAL/fit_${n}${FIT_SUFFIX:-}"; rm -rf "$out"
  .venv/bin/python -m model_fine_tuning.canon.fit_mesh --canon-dir "$REAL/canon_$n" --images "$REAL/images" --timestamps "$REAL/timestamps.json" \
      --out "$out" ${FIT_MESH:+--mesh "$FIT_MESH"} --overlays 60 --workers 6 > "$out.log" 2>&1
  echo "fit $n done: $(date -u +%T)"
done
echo ALLDONE
