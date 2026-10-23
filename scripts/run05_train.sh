#!/usr/bin/env bash
# run05 training, queued behind tools/run05_finish.sh (waits for RUN/dataset/.finished), detached and resumable:
#   split (common.splits make)  ->  seg_r05_ft4 (30 ep)  ->  canon_r05_ft4 (40 ep, run04 "plain" recipe)
#   ->  parts_v2 (landmarks/parts.py on run05 + the refreshed run04: real fin tip, edge chains, spine_end)
#   nohup setsid scripts/run05_train.sh > experiments/run05_train.log 2>&1 &
set -uo pipefail
cd /home/sat3737/NOAA_whales/Dinov3-WhaleID
F=/home/sat3737/NOAA_whales/Whale-Image-Data-Factory/data/renders
RUN=$F/run05; SPLIT=model_fine_tuning/splits/run05.json; PY=.venv/bin/python
log() { echo "[run05_train $(date -u +%FT%TZ)] $*"; }
finished() { [ -f "experiments/$1/log.jsonl" ] && grep -q '"event": "done"' "experiments/$1/log.jsonl"; }
while [ ! -f "$RUN/dataset/.finished" ]; do sleep 120; done
export WHALE_RUN_DIR="$RUN"
log "split"
[ -f "$SPLIT" ] || $PY -m model_fine_tuning.common.splits make --run-dir "$RUN" --out "$SPLIT" --min-visible-px 800 --drop-flagged 2>&1 | grep -v -i warn | tail -3
log "seg_r05_ft4"
finished seg_r05_ft4 || $PY -m model_fine_tuning.segmentation.train --name seg_r05_ft4 --run-dir "$RUN" --split "$SPLIT" \
    --unfreeze-blocks 4 --epochs "${R05_EPOCHS_SEG:-30}" --batch-size 8 --workers 6 > experiments/seg_r05_ft4.stdout 2>&1
log "canon_r05_ft4"
finished canon_r05_ft4 || $PY -m model_fine_tuning.canon.train --name canon_r05_ft4 --run-dir "$RUN" --split "$SPLIT" \
    --unfreeze-blocks 4 --epochs "${R05_EPOCHS_CANON:-40}" --batch-size 8 --workers 6 --min-visible 300 > experiments/canon_r05_ft4.stdout 2>&1
log "parts_v2 (run05 + run04 refreshed)"
[ -f experiments/part_landmarks/parts_v2/best.pt ] || $PY -m model_fine_tuning.landmarks.parts train --name parts_v2 --run-dir "$RUN" --split "$SPLIT" \
    --extra-run-dir "$F/run04" --extra-split model_fine_tuning/splits/run04.json --epochs "${R05_EPOCHS_PARTS:-10}" --batch-size 8 --workers 8 \
    > experiments/part_landmarks/parts_v2.stdout 2>&1
CK=experiments/part_landmarks/parts_v2/best.pt
if [ -f "$CK" ]; then
    $PY -m model_fine_tuning.landmarks.parts sheet --ckpt "$CK" --run-dir "$RUN" --split "$SPLIT" --out docs/part_landmarks_parts_v2_val.png 2>&1 | grep -v -i warn | tail -1
    for day in 2025-02-27 2025-01-15; do $PY -m model_fine_tuning.landmarks.parts predict --ckpt "$CK" --day "$day" --suffix _parts_v2 2>&1 | grep -v -i warn | tail -1; done
fi
log "done"
