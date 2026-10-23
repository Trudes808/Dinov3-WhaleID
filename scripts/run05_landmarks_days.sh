#!/usr/bin/env bash
# After run05 training: the landmark detector parts_v2 on every split day, the shape channel from its landmarks, and the
# day clustering with and without that channel (val days tell whether it helps; eval days are reported, never tuned).
# Detached, resumable (finished predictions / channels are skipped):
#   nohup setsid scripts/run05_landmarks_days.sh > experiments/run05_landmarks_days.log 2>&1 &
set -uo pipefail
cd /home/sat3737/NOAA_whales/Dinov3-WhaleID
PY=.venv/bin/python; export PYTHONPATH=.
NAME="${PART_NAME:-parts_v2}"; CK="experiments/part_landmarks/$NAME/best.pt"
LAB=model_fine_tuning/splits/labels_all_days.json
DAYS="2025-01-15 2025-02-27 2025-03-14 2024-12-18 2024-12-26 2025-02-19 2025-03-21"
log() { echo "[landmarks_days $(date -u +%FT%TZ)] $*"; }
while ! grep -q "\[run05_train .*\] done" experiments/run05_train.log 2>/dev/null; do sleep 300; done
[ -f "$CK" ] || { log "no $CK; stop"; exit 1; }
mkdir -p docs/shape_pairs
for d in $DAYS; do
    D=experiments/days_all/$d
    n_charts=$(ls $D/parts_xr/charts/*.jpg 2>/dev/null | wc -l); n_pred=$(ls $D/part_landmarks_$NAME/*.json 2>/dev/null | wc -l)
    if [ "$n_pred" -lt "$n_charts" ]; then
        log "$d: predict $NAME on $n_charts charts"
        $PY -m model_fine_tuning.landmarks.parts predict --ckpt "$CK" --day "$d" --suffix "_$NAME" --sheet-count 48 2>&1 | grep -v -i warn | tail -1
    fi
    log "$d: shape channel"
    $PY -m model_fine_tuning.cluster.shape_channel --day-dir "$D" --name "$NAME" --labels $LAB --sheet "docs/shape_pairs/$d.png" 2>&1 | grep -v -i warn | tail -3
done
log "clustering with the shape channel"
scripts/cluster_v3_all.sh cluster "_student_shape" --use "id:pair_student,shape:$NAME" 2>&1 | tail -12
scripts/cluster_v3_all.sh cluster "_v4_shape" --use "id:id_v4,shape:$NAME" 2>&1 | tail -12
scripts/cluster_v3_all.sh cluster "_shape_only" --use "shape:$NAME" 2>&1 | tail -12
log "summary (reference runs: _student, _v4id)"
$PY scripts/v3_summary.py --runs student=_student v4id=_v4id student_shape=_student_shape v4_shape=_v4_shape shape_only=_shape_only 2>&1 | tail -25
$PY - <<'PYEOF'
import json, pathlib
print("shape channel pair AUC per day (cross-burst pairs; within-group = same catalog group, other animal):")
for d in "2025-01-15 2025-02-27 2025-03-14 2024-12-18 2024-12-26 2025-02-19 2025-03-21".split():
    f = pathlib.Path(f"experiments/days_all/{d}/cluster_v3/shape_parts_v2.json")
    if f.exists():
        r = json.loads(f.read_text())
        print(d, {p: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r[p].items()} for p in ("fin", "fluke")})
PYEOF
log "done"
