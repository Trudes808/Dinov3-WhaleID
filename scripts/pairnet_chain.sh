#!/usr/bin/env bash
# Pair verifier (teacher) -> distilled embedding (student) -> pair AUC on identical pairs vs id_v4 (val, then eval days)
# -> the student as a day-clustering channel.   nohup setsid scripts/pairnet_chain.sh > experiments/pairnet_chain.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; export PYTHONPATH=.; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REF=experiments/id_v4/checkpoints/best.pt; SFX="${PAIR_SUFFIX:-}"
echo "== train teacher"; date
if ! { [ -f experiments/pair_teacher$SFX/log.jsonl ] && grep -q '"event": "done"' experiments/pair_teacher$SFX/log.jsonl; }; then
  $PY -m model_fine_tuning.cluster.pairnet train --name pair_teacher$SFX --data-root experiments/days_all --epochs "${PAIR_EPOCHS:-10}" --P 6 --workers 6 --reference $REF ${PAIR_EXTRA:-} > experiments/pair_teacher$SFX.stdout 2>&1
fi
echo "== distill student"; date
if ! { [ -f experiments/pair_student$SFX/log.jsonl ] && grep -q '"event": "done"' experiments/pair_student$SFX/log.jsonl; }; then
  $PY -m model_fine_tuning.cluster.pairnet distill --name pair_student$SFX --teacher experiments/pair_teacher$SFX/checkpoints/best.pt --data-root experiments/days_all \
      --epochs "${STUDENT_EPOCHS:-10}" --P 6 --workers 6 --init $REF --reference $REF ${STUDENT_EXTRA:-} > experiments/pair_student$SFX.stdout 2>&1
fi
echo "== eval on identical pairs (val, then eval days)"; date
for split in val eval; do
  $PY -m model_fine_tuning.cluster.pairnet eval --data-root experiments/days_all --split $split --teacher experiments/pair_teacher$SFX/checkpoints/best.pt \
      --student experiments/pair_student$SFX/checkpoints/best.pt --reference $REF --val-batches 80 --out experiments/pairnet_eval${SFX}_$split.json 2>&1 | grep -v -i warn | grep "auc_.*_mean\|pairs\"\|positives"
done
echo "== student as a day-clustering channel"; date
scripts/cluster_v3_all.sh channels "id:experiments/pair_student$SFX/checkpoints/best.pt" ""
scripts/cluster_v3_all.sh cluster _student$SFX --use id:pair_student$SFX --thr-offset 0.0 --cannot-link none --max-link-cost 1e9
echo "PAIRNET_CHAIN_DONE"; date
