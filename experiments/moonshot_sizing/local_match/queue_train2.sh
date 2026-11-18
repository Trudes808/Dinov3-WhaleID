#!/bin/bash
# continuation of queue_train.sh (whose first turn is already waiting): train days matched 2 x 2, packed into 20-min turns
cd "$(dirname "$0")"
P=../../../.venv/bin/python
TR="2024-12-23 2025-01-02 2025-01-08 2025-01-16 2025-01-17 2025-01-20 2025-01-23 2025-02-03 2025-02-12 2025-02-24 2025-02-28 2025-03-03 2025-03-06 2025-03-13 2025-03-18 2025-03-20 2025-03-24 2025-03-26 2025-04-01 2025-04-09 2025-04-11"
D_REST="2024-12-23 2025-01-08 2025-01-17 2025-01-20 2025-02-03 2025-02-12 2025-02-24 2025-02-28 2025-03-03 2025-03-06 2025-03-13 2025-03-20 2025-03-24 2025-03-26 2025-04-01 2025-04-09 2025-04-11"
run() { echo "== $(date +%T) $1"; ./gpu_job.sh "$1"; echo "== $(date +%T) exit $?"; }
until ! pgrep -f "lm_feats.py aliked1024 fluke 2025-01-15 2025-02-27 2025-03-14 --fallback" >/dev/null; do sleep 10; done
echo "== $(date +%T) first turn finished"
run "$P lm_feats.py aliked1024 fluke $TR && $P lm_feats.py aliked1024 fluke $TR --fallback && $P lm_match.py aliked1024 fluke $TR"
run "$P lm_match.py aliked1024 dorsal 2025-01-02 2025-01-16 2025-01-23 --no-mirror"
run "$P lm_match.py aliked1024 dorsal $D_REST 2025-03-18 --no-mirror"
run "$P lm_match.py aliked1024 fluke $TR && $P lm_match.py aliked1024 dorsal $TR --no-mirror"
echo "== $(date +%T) queue done"
