#!/bin/bash
# the GPU work of local_match after the pilot, one flock turn per line (each <= 20 min)
cd "$(dirname "$0")"
P=../../../.venv/bin/python
V="2025-01-15 2025-02-27 2025-03-14"
TR="2024-12-23 2025-01-02 2025-01-08 2025-01-16 2025-01-17 2025-01-20 2025-01-23 2025-02-03 2025-02-12 2025-02-24 2025-02-28 2025-03-03 2025-03-06 2025-03-13 2025-03-18 2025-03-20 2025-03-24 2025-03-26 2025-04-01 2025-04-09 2025-04-11"
FL_A="2024-12-23 2025-01-02 2025-01-08 2025-01-16 2025-01-17 2025-01-20 2025-01-23 2025-02-03 2025-02-12 2025-02-24 2025-02-28 2025-03-03 2025-03-06 2025-03-13 2025-03-20 2025-03-24 2025-03-26 2025-04-01 2025-04-09 2025-04-11"
D_REST="2024-12-23 2025-01-08 2025-01-17 2025-01-20 2025-02-03 2025-02-12 2025-02-24 2025-02-28 2025-03-03 2025-03-06 2025-03-24 2025-03-26 2025-04-09 2025-04-11"
run() { echo "== $(date +%T) $1"; ./gpu_job.sh "$1"; echo "== $(date +%T) exit $?"; }
run "$P lm_feats.py aliked1024 fluke $V --fallback && $P lm_match.py aliked1024 dorsal $V && $P lm_match.py aliked1024 fluke $V && $P lm_feats.py aliked1024 dorsal $TR --no-mirror"
run "$P lm_feats.py aliked1024 fluke $TR && $P lm_feats.py aliked1024 fluke $TR --fallback && $P lm_match.py aliked1024 fluke $FL_A"
run "$P lm_match.py aliked1024 fluke 2025-03-18 && $P lm_match.py aliked1024 dorsal 2025-01-16 2025-01-23 2025-03-20 --no-mirror"
run "$P lm_match.py aliked1024 dorsal 2025-01-02 --no-mirror"
run "$P lm_match.py aliked1024 dorsal 2025-03-18 2025-03-13 2025-04-01 --no-mirror"
run "$P lm_match.py aliked1024 dorsal $D_REST --no-mirror"
echo "== $(date +%T) queue done"
