#!/usr/bin/env bash
# After all days are processed: train the identity embedding on the train days' CSE charts (cluster/train_id.py), then
# cluster the eval and val days with it (cluster/cluster_bursts.py) and print the table next to the MegaDescriptor run.
#   nohup setsid scripts/train_id_and_cluster.sh > experiments/train_id_and_cluster.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; NAME="${ID_NAME:-id_v2}"; export PYTHONPATH=.
until grep -q ALLDONE experiments/process_all_days_v2.log; do sleep 300; done
echo "== train $NAME"; date
if ! { [ -f experiments/$NAME/log.jsonl ] && grep -q '"event": "done"' experiments/$NAME/log.jsonl; }; then
  $PY -m model_fine_tuning.cluster.train_id --name "$NAME" --data-root experiments/days_all --labels model_fine_tuning/splits/labels_all_days.json \
      --days model_fine_tuning/splits/days.json --epochs "${ID_EPOCHS:-15}" --batches-per-epoch 300 --unfreeze-blocks 4 --workers 6 ${ID_EXTRA:-} > experiments/$NAME.stdout 2>&1
fi
echo "== cluster eval + val days with $NAME"; date
for d in 2024-12-18 2024-12-26 2025-02-19 2025-03-21 2025-01-15 2025-02-27 2025-03-14; do
  [ -f experiments/days_all/$d/cluster_bursts_$NAME/metrics.json ] || $PY -m model_fine_tuning.cluster.cluster_bursts --day-dir experiments/days_all/$d \
      --labels model_fine_tuning/splits/labels_all_days.json --out experiments/days_all/$d/cluster_bursts_$NAME --embedder "id:experiments/$NAME/checkpoints/best.pt" --gap 1.5 ${CLUSTER_EXTRA:-} > experiments/days_all/$d/cluster_bursts_$NAME.log 2>&1
done
$PY - "$NAME" <<'PYEOF'
import json, sys
name=sys.argv[1]
split={d:"eval" for d in ("2024-12-18","2024-12-26","2025-02-19","2025-03-21")}; split.update({d:"val" for d in ("2025-01-15","2025-02-27","2025-03-14")})
print(f"{'day':11s} {'split':5s} {'animals':>7s} | MegaDescriptor: {'ARI':>6s} {'F1':>6s} | {name}: {'ARI':>6s} {'F1':>6s} {'purity':>6s} {'oracle':>6s} | groups {'ARI':>6s}")
for d in split:
    try:
        a=json.load(open(f"experiments/days_all/{d}/cluster_bursts_md/metrics.json")); b=json.load(open(f"experiments/days_all/{d}/cluster_bursts_{name}/metrics.json"))
        print(f"{d:11s} {split[d]:5s} {a['animals']['individuals']:7d} | {a['animals']['ari']:6.3f} {a['animals']['pair_f1']:6.3f} | {b['animals']['ari']:6.3f} {b['animals']['pair_f1']:6.3f} {b['animals']['cluster_purity']:6.3f} {max(s['ari'] for s in b['oracle_sweep_animals']):6.3f} | {b['groups']['ari']:6.3f}")
    except Exception as e: print(d, "missing", str(e)[:60])
PYEOF
echo "== done"; date
