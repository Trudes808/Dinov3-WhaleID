#!/usr/bin/env bash
# run05 evaluation (mirror of run04_evaluate.sh): the run05 models on the run05 test split and on the real photos of
# 2024-12-18 (prediction, label-free outline metric both rotations, side-flip rate), the run04 models on the run05 test
# split as the baseline, then a one-line summary of the key numbers into experiments/run05_eval_summary.txt.
#   scripts/run05_evaluate.sh            (resumable: finished outputs are skipped)
set -uo pipefail
cd /home/sat3737/NOAA_whales/Dinov3-WhaleID
RUN=/home/sat3737/NOAA_whales/Whale-Image-Data-Factory/data/renders/run05
SPLIT=model_fine_tuning/splits/run05.json
REAL="${R05_REAL:-experiments/real_2024-12-18}"
PY=.venv/bin/python
export WHALE_RUN_DIR="$RUN"
log() { echo "[run05_evaluate $(date -u +%FT%TZ)] $*"; }
eval_models() {   # $1 seg experiment, $2 canon experiment, $3 eval dir name
    local SEGX="$1" NAME="$2" TAG="$3"
    local SEG="experiments/$1/checkpoints/best.pt" CANON="experiments/$2/checkpoints/best.pt"
    if [ ! -f "$SEG" ] || [ ! -f "$CANON" ]; then log "$NAME: no checkpoints, skipped"; return 0; fi
    log "$NAME: segmentation on the run05 test split"
    [ -f "experiments/$SEGX/$TAG/metrics.json" ] || $PY -m model_fine_tuning.segmentation.evaluate --checkpoint "$SEG" --run-dir "$RUN" \
        --split-file "$SPLIT" --split test --out "experiments/$SEGX/$TAG" > "experiments/$SEGX.$TAG.stdout" 2>&1
    log "$NAME: canonical surface on the run05 test split"
    [ -f "experiments/$NAME/$TAG/metrics.json" ] || $PY -m model_fine_tuning.canon.evaluate --checkpoint "$CANON" --seg-checkpoint "$SEG" \
        --run-dir "$RUN" --split-file "$SPLIT" --split test --out "experiments/$NAME/$TAG" > "experiments/$NAME.$TAG.stdout" 2>&1
}
eval_real() {     # $1 seg experiment, $2 canon experiment
    local SEG="experiments/$1/checkpoints/best.pt" CANON="experiments/$2/checkpoints/best.pt" NAME="$2"
    if [ ! -f "$SEG" ] || [ ! -f "$CANON" ]; then return 0; fi
    log "$NAME: real photos: predict"
    [ -f "$REAL/canon_$NAME/points.jsonl" ] || $PY -m model_fine_tuning.canon.predict --seg-checkpoint "$SEG" --checkpoint "$CANON" \
        --images "$REAL/images" --out "$REAL/canon_$NAME" --max-side 1280 > "$REAL/canon_$NAME.log" 2>&1
    log "$NAME: real photos: outline metric (PnP and direct rotation)"
    [ -f "$REAL/outline_metric_$NAME.json" ] || $PY -m model_fine_tuning.canon.outline_metric --canon-dir "$REAL/canon_$NAME" \
        --images "$REAL/images" --timestamps "$REAL/timestamps.json" --out "$REAL/outline_metric_$NAME.json" > "$REAL/outline_metric_$NAME.log" 2>&1
    [ -f "$REAL/outline_metric_${NAME}_direct.json" ] || $PY -m model_fine_tuning.canon.outline_metric --canon-dir "$REAL/canon_$NAME" --rotation direct \
        --images "$REAL/images" --timestamps "$REAL/timestamps.json" --out "$REAL/outline_metric_${NAME}_direct.json" > "$REAL/outline_metric_${NAME}_direct.log" 2>&1
    log "$NAME: real photos: side-flip rate"
    [ -f "$REAL/compare_$NAME/summary.json" ] || $PY test_scripts/python/compare_real_before_after.py "$REAL/side_diag/real_side.json" "$REAL/canon_$NAME" "$REAL/compare_$NAME" \
        > "$REAL/compare_$NAME.log" 2>&1 || true
}
eval_models seg_r05_ft4 canon_r05_ft4 eval_test
eval_real seg_r05_ft4 canon_r05_ft4
log "run04 models on the run05 test split (baseline)"
eval_models seg_r04_ft4 canon_r04_ft4 eval_run05_test
log "summary"
$PY - > experiments/run05_eval_summary.txt 2>&1 <<'PYEOF'
import json, pathlib
def load(p):
    p = pathlib.Path(p); return json.loads(p.read_text()) if p.exists() else {}
def pick(d, keys):
    return {k: d[k] for k in keys if k in d}
print("run05 evaluation summary")
for seg, canon, tag, label in (("seg_r04_ft4", "canon_r04_ft4", "eval_run05_test", "run04 models on run05 test"), ("seg_r05_ft4", "canon_r05_ft4", "eval_test", "run05 models on run05 test")):
    s = load(f"experiments/{seg}/{tag}/metrics.json"); c = load(f"experiments/{canon}/{tag}/metrics.json")
    print(f"- {label}: seg {json.dumps({k: round(v, 3) for k, v in s.items() if isinstance(v, (int, float))} )[:300]}")
    print(f"    canon {json.dumps({k: round(v, 3) for k, v in c.items() if isinstance(v, (int, float))})[:400]}")
real = pathlib.Path("experiments/real_2024-12-18")
for name in ("canon_r04_ft4", "canon_r05_ft4"):
    for suf in ("", "_direct"):
        m = load(real / f"outline_metric_{name}{suf}.json")
        if m: print(f"- real 2024-12-18 outline metric {name}{suf}: {json.dumps({k: round(v, 3) for k, v in m.items() if isinstance(v, (int, float))})[:300]}")
PYEOF
cat experiments/run05_eval_summary.txt
log "evaluate done"
