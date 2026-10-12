#!/usr/bin/env bash
# Train the identity embedding on the train days and evaluate on every held-out day.
#   nohup setsid test_scripts/bash/identity_pipeline.sh id_v1 > experiments/identity_pipeline.log 2>&1 &
# Needs OUT_ROOT/<day>/parts for the train/val/eval days (process_all_days.sh). Idempotent per stage.
set -uo pipefail
cd "$(dirname "$0")/../.."
NAME="${1:-id_v1}"; ROOT="${2:-experiments/days}"; PY=.venv/bin/python; export PYTHONPATH=.
LABELS=model_fine_tuning/splits/labels_all_days.json; DAYS=model_fine_tuning/splits/days.json
echo "== processed days: $(ls -d $ROOT/*/parts 2>/dev/null | wc -l)"; date
$PY - <<PYEOF
import json, os
d = json.load(open("$DAYS"))
for k in ("train", "val", "eval"):
    have = [x for x in d[k] if os.path.exists(f"$ROOT/{x}/parts/index.jsonl")]
    print(f"  {k}: {len(have)}/{len(d[k])} days processed; missing: {[x for x in d[k] if x not in have]}")
PYEOF
echo "== train $NAME"; date
if ! { [ -f experiments/$NAME/log.jsonl ] && grep -q '"event": "done"' experiments/$NAME/log.jsonl; }; then
    $PY -m model_fine_tuning.cluster.train_id --name $NAME --data-root $ROOT --labels $LABELS --days $DAYS --epochs 15 --batches-per-epoch 300 --P 8 --K 4 --size 448 --workers 6 > experiments/$NAME.stdout 2>&1
fi
echo "== evaluate on the held-out days"; date
for day in $($PY -c "import json; print(' '.join(json.load(open('$DAYS'))['eval']))"); do
    D=$ROOT/$day; [ -f $D/parts/index.jsonl ] || { echo "  $day not processed yet, skipping"; continue; }
    [ -f $D/cluster_$NAME/scores.npz ] || $PY -m model_fine_tuning.cluster.eval_id --checkpoint experiments/$NAME/checkpoints/best.pt --day-dir $D --labels $LABELS --out $D/cluster_$NAME 2>&1 | grep -v -i warn
    # per-day labels in the clustering script's format
    $PY - "$day" "$D" <<PYEOF
import json, sys
day, D = sys.argv[1], sys.argv[2]
L = json.load(open("$LABELS"))["days"][day]["photos"]; ts = json.load(open(f"{D}/timestamps.json"))
labels = {s: [L.get(s.split("__")[-1], "unlabelled")] for s in ts}
json.dump({"_note": "EVALUATION ONLY", "labels": labels}, open(f"{D}/labels.json", "w"), indent=1)
PYEOF
    for cfg in "id best 1.0" "id best 0.0" "id same 1.0"; do set -- $cfg
        $PY -m model_fine_tuning.cluster.cluster --scores $D/cluster_$NAME/scores.npz --labels $D/labels.json --out $D/cluster_$NAME --sources $1 --hypothesis $2 --time-weight $3 --tag ${day}_$1_$2_t$3 2>&1 | grep -v -i warn | grep "^\[" | cut -c1-230
    done
    $PY - "$D" <<PYEOF
import json, numpy as np, sys
sys.path.insert(0, ".")
from model_fine_tuning.cluster.cluster import cluster, pair_metrics
D = sys.argv[1]; z = np.load(f"{D}/cluster_$NAME/scores.npz"); stems = [str(s) for s in z["stems"]]
lab = json.load(open(f"{D}/labels.json"))["labels"]; gt = np.array([lab[s][0] for s in stems])
pred = cluster(z["time_term"].astype(float), 0.25); m = pair_metrics(pred, gt)
print(f"[time-only] clusters={m['clusters']} (individuals {m['individuals']}) ARI={m['ari']:.3f} pair P/R/F1={m['pair_precision']:.3f}/{m['pair_recall']:.3f}/{m['pair_f1']:.3f}")
PYEOF
done
echo "== done"; date
