#!/usr/bin/env bash
# After dino_hr features and the id_v4 chain exist: hi-res DINO global + local channels on the eval + val days, clustering
# with the hi-res head alone and with id_v4 + hi-res, the fusion report (per-channel pair AUC incl. hrlocal), tuning.
#   nohup setsid scripts/dino_hr_chain.sh > experiments/dino_hr_chain.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; export PYTHONPATH=.
until grep -q "DINO_HR_ALLDONE" experiments/dino_hr_train.log && grep -q "ID_V4_DONE" experiments/id_v4_chain.log; do sleep 60; done
echo "== hr channels"; date
scripts/cluster_v3_all.sh channels "hr:dino_hr,id:experiments/id_v4/checkpoints/best.pt" "hrlocal" 30 "hr:dino_hr,id:experiments/id_v4/checkpoints/best.pt"
echo "== fusion report (all channels, no indicators)"; date
scripts/cluster_v3_all.sh fuse fusion_D_check "" --no-missing
echo "== cluster: hr alone"; scripts/cluster_v3_all.sh cluster _hr --use hr:dino_hr --thr-offset 0.0 --cannot-link none --max-link-cost 1e9
echo "== cluster: id_v4 + hr"; scripts/cluster_v3_all.sh cluster _v4hr --use id:id_v4,hr:dino_hr --thr-offset 0.0 --cannot-link none --max-link-cost 1e9
echo "== cluster: id_v4, tracklets 1.5 + fin cannot-link"; scripts/cluster_v3_all.sh cluster _v4id_t --use id:id_v4 --thr-offset 0.0 --cannot-link fin --max-link-cost 1.5
echo "DINO_HR_CHAIN_DONE"; date
