#!/usr/bin/env bash
set -uo pipefail
cd /home/sat3737/NOAA_whales/Dinov3-WhaleID; export PYTHONPATH=.; R=experiments/real_2024-12-18; PY=.venv/bin/python
echo "== identity head, all photos"; date
$PY -m model_fine_tuning.cluster.identity --atlas $R/atlas_v2 --timestamps $R/timestamps.json --out $R/atlas_v2_id 2>&1 | grep -v -i warn
echo "== identity head, trained on the 7D only (R5 photos held out)"; date
$PY -m model_fine_tuning.cluster.identity --atlas $R/atlas_v2 --timestamps $R/timestamps.json --out $R/atlas_v2_id7d --train-cameras "Canon EOS 7D Mark II" 2>&1 | grep -v -i warn
for tag in id id7d; do
  echo "== match + AUC ($tag)"; date
  $PY -m model_fine_tuning.cluster.match --atlas $R/atlas_v2_$tag --timestamps $R/timestamps.json --out $R/cluster_$tag 2>&1 | grep -v -i warn | grep "^id "
  $PY test_scripts/python/pair_auc.py $R/cluster_$tag/scores.npz $R/labels_from_lightroom.json 2>&1 | grep -v -i warn | grep -E "source|^id |^dino_1 |time"
done
echo "== cross-camera check: AUC on R5-R5 pairs only, head trained on 7D (true hold-out)"
$PY - <<'PYEOF'
import json, numpy as np
from sklearn.metrics import roc_auc_score
R = "experiments/real_2024-12-18"
for tag in ("id", "id7d"):
    z = np.load(f"{R}/cluster_{tag}/scores.npz"); stems = [str(s) for s in z["stems"]]
    ts = json.loads(open(f"{R}/timestamps.json").read()); lab = json.loads(open(f"{R}/labels_from_lightroom.json").read())["labels"]
    cam = np.array([ts[s]["camera"] for s in stems]); gt = np.array([lab[s][0] for s in stems]); grp = np.array([g[:3] for g in gt])
    iu = np.triu_indices(len(stems), 1); same = (gt[:, None] == gt[None, :])[iu]; sg = (grp[:, None] == grp[None, :])[iu]; dt = z["dt_s"][iu]
    r5 = ((cam == "Canon EOS R5")[:, None] & (cam == "Canon EOS R5")[None, :])[iu]
    a = z["id__same"][iu]; f = z["id__flip"][iu]; a = np.where(np.isnan(a), f, np.where(np.isnan(f), a, np.maximum(a, f))); ok = np.isfinite(a)
    for name, m in (("R5 pairs, all", r5), ("R5 pairs, same group 10s-1h", r5 & sg & (dt > 10) & (dt < 3600)), ("all pairs, same group 10s-1h", sg & (dt > 10) & (dt < 3600))):
        mm = ok & m; print(f"  [{tag}] {name:32s} n={mm.sum():5d} AUC={roc_auc_score(same[mm], a[mm]):.3f}" if mm.sum() > 20 and same[mm].any() and (~same[mm]).any() else f"  [{tag}] {name}: n/a")
PYEOF
echo "== clustering with the identity source"; date
for tag in id id7d; do for hyp in best same; do for tw in 1.0 0.5 0.0; do
  $PY -m model_fine_tuning.cluster.cluster --scores $R/cluster_$tag/scores.npz --labels $R/labels_from_lightroom.json --out $R/cluster_$tag --sources id --hypothesis $hyp --time-weight $tw --tag ${tag}_${hyp}_t${tw} 2>&1 | grep -v -i warn | grep "^\[" | cut -c1-250
done; done; done
echo "== explanations (id, best, t1)"; date
$PY -m model_fine_tuning.cluster.explain --atlas $R/atlas_v2_id --scores $R/cluster_id/scores.npz --clusters $R/cluster_id/clusters_id_best_t1.0.json --images $R/images --timestamps $R/timestamps.json --out $R/cluster_id/explain --source id --max-links 4 2>&1 | grep -v -i warn | tail -1
echo "== done"; date
