#!/usr/bin/env bash
# One command for a new raw day (rules_scene_v3.md S6 / S7): decode the RAW frames, S6h, the frozen scene model v3,
# one marker record per photo and a sheet of every marker call. Checks the frozen checkpoint / code hashes first.
#
#   model_fine_tuning/seg_multi/run_new_day.sh "RAW_DAY_FOLDER" OUT_ROOT [FREEZE.json]
#   e.g. .../run_new_day.sh "/home/sat3737/NOAA_whales/raw_test_days_unsorted/2026-4-7 C1 Harvey" /home/sat3737/NOAA_whales/moonshot_cache/sam3_labeller/v3/day_runs
#
# RAW_DAY_FOLDER: a folder of CR2 / CR3 files whose name starts with YYYY-M-D (only read). Output (keep it untracked):
#   OUT_ROOT/src/<YYYY-M-D>/<folder name>/  per-file symlinks (so image names match the other days)
#   OUT_ROOT/days/<ISO day>/images/          2560-px JPEGs + index.jsonl (test_scripts/python/convert_raw.py), timestamps.json
#   OUT_ROOT/days/<ISO day>/scene3/{scene,s6prob}/<stem>.png   scene map (our classes) and S6 P(whale) at 1280
#   OUT_ROOT/<ISO day>_markers.jsonl         one record per photo, time order
#   OUT_ROOT/<ISO day>_marker_calls_<page>.jpg   every marker call (shows crew: never into the repo)
#   OUT_ROOT/<ISO day>_removal.jsonl         the marker records + removal rule v2 (proposed / faint whale?), SAM 3 in days/<ISO day>/removal_sam3
set -euo pipefail
cd "$(dirname "$0")/../.."
RAW="${1:?raw day folder}"; ROOT="${2:?output root}"; FREEZE="${3:-/home/sat3737/NOAA_whales/moonshot_cache/sam3_labeller/v3/FREEZE.json}"
NAME="$(basename "$RAW")"; DATE="${NAME%% *}"; IFS=- read -r Y M D <<< "$DATE"; DAY=$(printf "%04d-%02d-%02d" "$((10#$Y))" "$((10#$M))" "$((10#$D))")
case "$DAY" in 2024-12-18|2024-12-26|2025-02-19|2025-03-21) echo "evaluation day: refused"; exit 2;; esac
PY=.venv/bin/python; export PYTHONPATH=.
LOCK=/var/tmp/sat3737-parts-cache/gpu_train.lock
# 1. the freeze: checkpoint, reader code and gate
read -r CKPT GATE < <($PY - "$FREEZE" <<'PYEOF'
import hashlib, json, sys
f = json.load(open(sys.argv[1]))
def h(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
bad = [p for p, v in {**f["code_sha256"], f["checkpoint"]: f["checkpoint_sha256"]}.items() if h(p) != v]
if bad:
    sys.exit("FREEZE MISMATCH: " + ", ".join(bad))
print(f["checkpoint"], f["gate"])
PYEOF
)
[ -n "${CKPT:-}" ] && [ -n "${GATE:-}" ] || { echo "freeze check failed: nothing run"; exit 3; }
echo "== $DAY: frozen checkpoint $CKPT, gate $GATE"
# 2. decode
SRC="$ROOT/src/$DATE/$NAME"; OUT="$ROOT/days/$DAY"; mkdir -p "$SRC" "$OUT"
for f in "$RAW"/*; do case "${f,,}" in *.cr2|*.cr3|*.nef|*.arw|*.dng|*.raf) ln -sfn "$f" "$SRC/$(basename "$f")";; esac; done
[ -f "$OUT/images/index.jsonl" ] || $PY test_scripts/python/convert_raw.py "$ROOT/src/$DATE" "$OUT/images" --long-side 2560 --workers "${RAW_WORKERS:-10}" > "$OUT/images.log" 2>&1
echo "== $DAY: $(wc -l < "$OUT/images/index.jsonl") photos decoded"
# 3. S6h + scene model + records + sheet (GPU under the shared lock)
flock "$LOCK" $PY -m model_fine_tuning.seg_multi.markers --day-dir "$OUT" --ckpt "$CKPT" --gate "$GATE" --cache "$OUT/scene3" \
    --out "$ROOT/${DAY}_markers.jsonl" --sheet "$ROOT/${DAY}_marker_calls.jpg"
# 4. removal rule v2 (frozen separately: experiments/removal_v2/README.md; its own hash check first). SAM 3 takes the
#    GPU lock inside removal.py; REMOVAL_ARGS=--sam3-cpu runs it on the CPU instead.
$PY experiments/removal_v2/freeze.py check
$PY -m model_fine_tuning.seg_multi.removal --day-dir "$OUT" --markers "$ROOT/${DAY}_markers.jsonl" --cache "$OUT/scene3" \
    --sam3-out "$OUT/removal_sam3" --out "$ROOT/${DAY}_removal.jsonl" ${REMOVAL_ARGS:-}
echo "== $DAY: done -> $ROOT/${DAY}_markers.jsonl, $ROOT/${DAY}_removal.jsonl"
