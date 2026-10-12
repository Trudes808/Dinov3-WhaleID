#!/usr/bin/env bash
# Train the canonical-surface model on run02, evaluate it (oracle + seg crops), run it on the
# real photos. Usage: canon_pipeline.sh [NAME] (default canon_r02_ft4). Idempotent per stage.
set -euo pipefail
cd "$(dirname "$0")/../.."
NAME="${1:-canon_r02_ft4}"
RUN="${WHALE_RUN02_DIR:-$HOME/NOAA_whales/Whale-Image-Data-Factory/data/renders/run02}"
SPLIT=model_fine_tuning/splits/run02.json
SEG=experiments/seg_r02_ft4/checkpoints/best.pt
PY=.venv/bin/python
echo "== train $NAME"; date
if ! { [ -f "experiments/$NAME/log.jsonl" ] && grep -q '"event": "done"' "experiments/$NAME/log.jsonl"; }; then
    $PY -m model_fine_tuning.canon.train --name "$NAME" --run-dir "$RUN" --split "$SPLIT" --unfreeze-blocks 4 --epochs 40 --batch-size 8 --workers 6 > "experiments/$NAME.stdout" 2>&1
fi
echo "== evaluate (run02 test)"; date
[ -f "experiments/$NAME/eval_test/metrics.json" ] || $PY -m model_fine_tuning.canon.evaluate --checkpoint "experiments/$NAME/checkpoints/best.pt" --seg-checkpoint "$SEG" --run-dir "$RUN" --split-file "$SPLIT" --split test
echo "== real photos"; date
[ -f "experiments/real_2024-12-18/canon_$NAME/points.jsonl" ] || $PY -m model_fine_tuning.canon.predict --seg-checkpoint "$SEG" --checkpoint "experiments/$NAME/checkpoints/best.pt" --images experiments/real_2024-12-18/images --out "experiments/real_2024-12-18/canon_$NAME" --max-side 1280 > "experiments/real_2024-12-18/canon_$NAME.log" 2>&1
echo "== done"; date
