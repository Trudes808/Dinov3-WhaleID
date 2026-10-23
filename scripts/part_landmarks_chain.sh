#!/usr/bin/env bash
# Fin / fluke landmark detector, queued behind the factory review render (one GPU):
#   1. wait until the r05_individuals render has finished (all batches done, no editor running)
#   2. factory postprocess + dataset + individuals review sheet (CPU)
#   3. train landmarks/parts.py on run04 (GPU), then the val sheet and predictions on two val days
#   nohup setsid scripts/part_landmarks_chain.sh > experiments/part_landmarks/chain.log 2>&1 &
set -uo pipefail
DINO=/home/sat3737/NOAA_whales/Dinov3-WhaleID
FACT=/home/sat3737/NOAA_whales/Whale-Image-Data-Factory
REVIEW=$FACT/data/renders/r05_individuals
NAME="${PART_NAME:-parts_v1}"
cd "$DINO"
log() { echo "[chain $(date -u +%FT%TZ)] $*"; }

n_batches=$(ls "$REVIEW"/batches/batch_*.json | wc -l)
log "waiting for the review render ($n_batches batches)"
while true; do
    done_n=$(ls "$REVIEW"/raw/*/done 2>/dev/null | wc -l)
    editor=$(pgrep -f "^[^ ]*UnrealEditor" | wc -l)      # the editor binary itself (harness shells quote the name in their command lines)
    if [ "$done_n" -ge "$n_batches" ] && [ "$editor" -eq 0 ]; then break; fi
    if ! pgrep -f "tools/render_batch.sh" >/dev/null && [ "$editor" -eq 0 ]; then log "render_batch.sh is not running (done $done_n/$n_batches); continuing anyway"; break; fi
    sleep 60
done
log "render finished ($done_n/$n_batches batches)"

if [ ! -f "$REVIEW/individuals_sheet.png" ]; then
    (cd "$FACT" && PYTHONPATH="$FACT" python3 -m factory.postprocess "$REVIEW" --workers 8 && PYTHONPATH="$FACT" python3 -m factory.dataset "$REVIEW" \
        && PYTHONPATH="$FACT" python3 tools/individuals_sheet.py "$REVIEW" --cols 6 --tile 420) 2>&1 | grep -v -i "warn" | tail -20
    log "review sheet: $REVIEW/individuals_sheet.png"
fi

log "train $NAME"
.venv/bin/python -m model_fine_tuning.landmarks.parts train --name "$NAME" --run-dir "$FACT/data/renders/run04" --split model_fine_tuning/splits/run04.json \
    --epochs "${PART_EPOCHS:-8}" --batch-size 8 --workers 8 ${PART_EXTRA:-} 2>&1 | grep -v -i "warn" | grep -v "^$"
CK=experiments/part_landmarks/$NAME/best.pt
[ -f "$CK" ] || { log "no checkpoint; stop"; exit 1; }
.venv/bin/python -m model_fine_tuning.landmarks.parts sheet --ckpt "$CK" --run-dir "$FACT/data/renders/run04" --split model_fine_tuning/splits/run04.json --out "docs/part_landmarks_${NAME}_val.png" 2>&1 | grep -v -i warn
for day in 2025-02-27 2025-01-15; do
    .venv/bin/python -m model_fine_tuning.landmarks.parts predict --ckpt "$CK" --day "$day" --suffix "_$NAME" 2>&1 | grep -v -i warn
done
log "done"
