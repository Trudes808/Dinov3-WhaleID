#!/usr/bin/env bash
# Rebuild the 2024-12-18 matching chain on the run03 models (canon predictions already made by
# scripts/run03_evaluate.sh): atlas -> identity head -> pairwise match -> clusters, then the pair AUC
# against the Lightroom labels for a before/after. Idempotent per stage.
set -uo pipefail
cd "$(dirname "$0")/.."
R=experiments/real_2024-12-18; PY=.venv/bin/python; export PYTHONPATH=.
CANON=experiments/canon_r03_ft4/checkpoints/best.pt
echo "== atlas_v3"; date
[ -f $R/atlas_v3/meta.json ] || $PY -m model_fine_tuning.cluster.atlas --images $R/images --canon-dir $R/canon_canon_r03_ft4 --out $R/atlas_v3 --canon-checkpoint $CANON --k 512 --layers 8 11 > $R/atlas_v3.log 2>&1
echo "== identity head"; date
[ -f $R/atlas_v3_id/head.pt ] || $PY -m model_fine_tuning.cluster.identity --atlas $R/atlas_v3 --timestamps $R/timestamps.json --out $R/atlas_v3_id 2>&1 | grep -v -i warn | tail -3
echo "== match"; date
[ -f $R/cluster_v3id/scores.npz ] || $PY -m model_fine_tuning.cluster.match --atlas $R/atlas_v3_id --timestamps $R/timestamps.json --out $R/cluster_v3id 2>&1 | grep -v -i warn | grep "^id "
echo "== pair AUC (run03 chain vs the run02 chain)"; date
for tag in cluster_v3id cluster_id; do echo "-- $tag"; $PY test_scripts/python/pair_auc.py $R/$tag/scores.npz $R/labels_from_lightroom.json 2>&1 | grep -v -i warn | grep -E "source|^id |^dino_1 |time"; done
echo "== clusters"; date
for hyp in best same; do for tw in 1.0 0.5 0.0; do
  $PY -m model_fine_tuning.cluster.cluster --scores $R/cluster_v3id/scores.npz --labels $R/labels_from_lightroom.json --out $R/cluster_v3id --sources id --hypothesis $hyp --time-weight $tw --tag id_${hyp}_t${tw} 2>&1 | grep -v -i warn | grep "^\[" | cut -c1-250
done; done
echo "-- run02 chain, same setting:"; cat $R/cluster_id/metrics_id_best_t1.0.json 2>/dev/null | head -c 400; echo
echo "== done"; date
