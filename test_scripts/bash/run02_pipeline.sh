#!/usr/bin/env bash
# Train and evaluate the run02 models (same recipes as run01) and cross-evaluate run01 models.
#   test_scripts/bash/run02_pipeline.sh [RUN_DIR]      default ../Whale-Image-Data-Factory/data/renders/run02
# Stages are idempotent: a finished experiment (log.jsonl ends with "done") is skipped.
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN="${1:-$HOME/NOAA_whales/Whale-Image-Data-Factory/data/renders/run02}"
SPLIT=model_fine_tuning/splits/run02.json
PY=.venv/bin/python
MIN_VISIBLE=${MIN_VISIBLE:-2000}

[ -f "$SPLIT" ] || $PY -m model_fine_tuning.common.splits make --run-dir "$RUN" --out "$SPLIT" --min-visible-px "$MIN_VISIBLE" --drop-flagged

done_() { [ -f "experiments/$1/log.jsonl" ] && grep -q '"event": "done"' "experiments/$1/log.jsonl"; }
train() {  # name module extra-args...
    local name=$1 mod=$2; shift 2
    if done_ "$name"; then echo "skip $name (done)"; return; fi
    nohup $PY -m "model_fine_tuning.$mod.train" --name "$name" --run-dir "$RUN" --split "$SPLIT" --workers 6 "$@" \
        > "experiments/$name.stdout" 2>&1 &
    echo "launched $name pid $!"
}
wait_all() { while pgrep -f "model_fine_tuning\.(segmentation|bodycoords)\.train" > /dev/null; do sleep 60; done; }

echo "== wave 1"; date
train seg_r02_ftall  segmentation --unfreeze-blocks -1 --epochs 30 --batch-size 8 --lr-backbone 2e-5
train seg_r02_frozen segmentation --unfreeze-blocks 0  --epochs 30 --batch-size 8
train body_r02_ft4   bodycoords   --unfreeze-blocks 4  --epochs 40 --batch-size 8
wait_all
echo "== wave 2"; date
train seg_r02_ft4    segmentation --unfreeze-blocks 4  --epochs 30 --batch-size 8
train body_r02_frozen bodycoords  --unfreeze-blocks 0  --epochs 40 --batch-size 8
wait_all
echo "== evaluation on the run02 test split"; date
for n in seg_r02_frozen seg_r02_ft4 seg_r02_ftall; do
    [ -f experiments/$n/eval_test/metrics.json ] || $PY -m model_fine_tuning.segmentation.evaluate --checkpoint experiments/$n/checkpoints/best.pt --run-dir "$RUN" --split-file "$SPLIT" --split test
done
for n in body_r02_ft4 body_r02_frozen; do
    [ -f experiments/$n/eval_test/metrics.json ] || $PY -m model_fine_tuning.bodycoords.evaluate --checkpoint experiments/$n/checkpoints/best.pt --seg-checkpoint experiments/seg_r02_ftall/checkpoints/best.pt --run-dir "$RUN" --split-file "$SPLIT" --split test
done
echo "== cross-evaluation: run01 models on the run02 test split"; date
for n in seg_vitb16_frozen seg_vitb16_ft4 seg_vitb16_ftall; do
    [ -f experiments/$n/eval_run02_test/metrics.json ] || $PY -m model_fine_tuning.segmentation.evaluate --checkpoint experiments/$n/checkpoints/best.pt --run-dir "$RUN" --split-file "$SPLIT" --split test --out experiments/$n/eval_run02_test
done
echo "== cross-evaluation: run02 models on the run01 test split"; date
for n in seg_r02_frozen seg_r02_ft4 seg_r02_ftall; do
    [ -f experiments/$n/eval_run01_test/metrics.json ] || $PY -m model_fine_tuning.segmentation.evaluate --checkpoint experiments/$n/checkpoints/best.pt --split test --out experiments/$n/eval_run01_test
done
echo "== done"; date
