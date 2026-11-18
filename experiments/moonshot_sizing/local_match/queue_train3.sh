#!/bin/bash
# continuation of queue_train.sh, cut to two turns because of the GPU queue (2026-11-18 03:40):
# train flukes (cut-outs only, 2 x 2), then train dorsal fins 2 x 2 on 18 of 21 days (the three biggest days left out)
cd "$(dirname "$0")"
P=../../../.venv/bin/python
TR="2024-12-23 2025-01-02 2025-01-08 2025-01-16 2025-01-17 2025-01-20 2025-01-23 2025-02-03 2025-02-12 2025-02-24 2025-02-28 2025-03-03 2025-03-06 2025-03-13 2025-03-18 2025-03-20 2025-03-24 2025-03-26 2025-04-01 2025-04-09 2025-04-11"
D18="2024-12-23 2025-01-08 2025-01-17 2025-01-20 2025-01-23 2025-02-03 2025-02-12 2025-02-24 2025-02-28 2025-03-03 2025-03-06 2025-03-13 2025-03-20 2025-03-24 2025-03-26 2025-04-01 2025-04-09 2025-04-11"
run() { echo "== $(date +%T) $1"; ./gpu_job.sh "$1"; echo "== $(date +%T) exit $?"; }
until ! pgrep -f "lm_feats.py aliked1024 fluke 2025-01-15 2025-02-27 2025-03-14 --fallback" >/dev/null; do sleep 10; done
echo "== $(date +%T) first turn finished"
run "$P lm_feats.py aliked1024 fluke $TR && $P lm_match.py aliked1024 fluke $TR"
run "$P lm_match.py aliked1024 dorsal $D18 --no-mirror"
echo "== $(date +%T) queue done"
