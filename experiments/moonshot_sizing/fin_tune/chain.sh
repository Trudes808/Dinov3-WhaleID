#!/bin/bash
# fin_tune chain (2026-11-18). Each item = one or more lock holds of at most ~27 min: train (stops itself after
# MAXMIN minutes with a resume checkpoint, exit 3) -> embed the 8 test days -> (outside the lock) score on CPU.
# Items:  raw                 embed raw HW-b6 on the test days (reproduction check of this folder's pipeline)
#         s<N>_<seed>[:steps]  train on pool[:N] with that seed (default STEPS steps), embed, score
#   C=/home/sat3737/NOAA_whales/moonshot_cache/fin_tune
#   setsid nohup bash experiments/moonshot_sizing/fin_tune/chain.sh raw s5_0 s16_0 > $C/logs/chain_A.log 2>&1 &
set -uo pipefail
cd /home/sat3737/NOAA_whales/Dinov3-WhaleID/experiments/moonshot_sizing/fin_tune
PY=/home/sat3737/NOAA_whales/Dinov3-WhaleID/.venv/bin/python
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOCK=/var/tmp/sat3737-parts-cache/gpu_train.lock
C=/home/sat3737/NOAA_whales/moonshot_cache/fin_tune
M=$C/markers; L=$C/logs; mkdir -p $M $L
STEPS=${STEPS:-600}; MAXMIN=${MAXMIN:-22}

if [ "${1:-}" = --inlock ]; then          # one lock hold for one item
  item=$2; name=${item%%:*}; steps=${item#*:}; [ "$steps" = "$item" ] && steps=$STEPS
  t0=$(date +%s); echo "## $name lock taken $(date '+%F %T')"; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
  if [ "${RAW_FIRST:-}" = 1 ] && [ ! -f $M/emb_raw.done ]; then     # timing hold: the raw reproduction embed first
    $PY -W ignore embed.py --name raw >> $L/embed_raw.log 2>&1 && touch $M/emb_raw.done; echo "## raw embedded ($(( $(date +%s) - t0 )) s)"
  fi
  if [ "$name" = raw ]; then
    $PY -W ignore embed.py --name raw >> $L/embed_raw.log 2>&1 || exit 1
    touch $M/emb_raw.done; echo "## raw embedded ($(( $(date +%s) - t0 )) s)"; exit 0
  fi
  N=${name#[st]}; N=${N%%_*}; SEED=${name##*_}
  if [ ! -f $C/runs/$name/final.pt ]; then
    $PY -W ignore train.py --name $name --days $N --seed $SEED --steps $steps --max-minutes $MAXMIN >> $L/train_$name.log 2>&1
    rc=$?; echo "## $name train rc $rc ($(( $(date +%s) - t0 )) s)"
    [ $rc -eq 3 ] && exit 3
    [ $rc -ne 0 ] && exit 1
  fi
  if [ $(( $(date +%s) - t0 )) -gt $(( 25 * 60 )) ]; then echo "## $name: embed left for the next hold"; exit 3; fi
  $PY -W ignore embed.py --name $name --ckpt $C/runs/$name/final.pt >> $L/embed_$name.log 2>&1 || exit 1
  touch $M/emb_$name.done; echo "## $name embedded, lock held $(( $(date +%s) - t0 )) s $(date '+%F %T')"; exit 0
fi

echo $$ > $C/chain.pid
for item in "$@"; do
  name=${item%%:*}
  while [ ! -f $M/emb_$name.done ]; do
    echo "#### $name waiting for the lock $(date '+%F %T')"
    flock $LOCK bash "$0" --inlock "$item"; rc=$?
    [ $rc -eq 3 ] && continue
    [ $rc -ne 0 ] && { echo "#### $name FAILED rc $rc $(date '+%F %T')"; touch $M/$name.failed; exit 1; }
  done
  if [ "$name" != raw ] && [ ! -f $M/score_$name.done ]; then
    $PY -W ignore score.py $name > score_$name.log 2>&1 && touch $M/score_$name.done
    [[ $name == s* ]] && [ -f $M/score_$name.done ] && rm -f $C/runs/$name/resume.pt
  fi
  [ "$name" = raw ] && { $PY -W ignore score.py raw > score_raw.log 2>&1; }
  echo "#### $name finished $(date '+%F %T')"
done
touch $M/chain_$(echo "$@" | tr ' :' '_-').done; echo "CHAIN_DONE $(date '+%F %T')"
