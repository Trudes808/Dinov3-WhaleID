#!/usr/bin/env bash
set -uo pipefail
cd /home/sat3737/NOAA_whales/Dinov3-WhaleID
R=experiments/real_2024-12-18; PY=.venv/bin/python; export PYTHONPATH=.
until [ -f $R/atlas_v2/meta.json ]; do sleep 20; done
echo "== match"; date
$PY -m model_fine_tuning.cluster.match --atlas $R/atlas_v2 --timestamps $R/timestamps.json --out $R/cluster_v2 2>&1 | grep -v -i warn
echo "== cluster grid"; date
for src in "dino_1" "dino_0" "canon_1" "canon_0" "colour" "dino_1 colour" "dino_1 canon_1 colour"; do
  for hyp in best same; do
    for tw in 1.0 0.0; do
      $PY -m model_fine_tuning.cluster.cluster --scores $R/cluster_v2/scores.npz --labels $R/labels_from_lightroom.json --out $R/cluster_v2 --sources $src --hypothesis $hyp --time-weight $tw 2>&1 | grep -v -i warn
    done
  done
done
echo "== time only (appearance ignored)"
$PY - <<'PYEOF'
import json, numpy as np, sys
sys.path.insert(0, ".")
from model_fine_tuning.cluster.cluster import cluster, pair_metrics
R = "experiments/real_2024-12-18"
z = np.load(f"{R}/cluster_v2/scores.npz"); stems = [str(s) for s in z["stems"]]
lab = json.loads(open(f"{R}/labels_from_lightroom.json").read())["labels"]; gt = np.array([lab[s][0] for s in stems])
for thr in (0.25, 0.75, 1.25):
    pred = cluster(z["time_term"].astype(float), thr); m = pair_metrics(pred, gt)
    print(f"[time-only thr={thr}] clusters={m['clusters']} ARI={m['ari']:.3f} pair P/R/F1={m['pair_precision']:.3f}/{m['pair_recall']:.3f}/{m['pair_f1']:.3f}")
PYEOF
echo "== done"; date
