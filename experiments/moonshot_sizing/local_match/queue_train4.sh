#!/bin/bash
# one packed GPU turn (2026-11-18 04:25, the lock queue left time for one or two turns): train flukes on 18 days,
# train dorsal fins on the 14 smallest days (2 x 2, no mirror); a second turn for the next dorsal days if time allows
cd "$(dirname "$0")"
P=../../../.venv/bin/python
FL="2024-12-23 2025-01-08 2025-01-16 2025-01-17 2025-01-20 2025-01-23 2025-02-03 2025-02-12 2025-02-24 2025-02-28 2025-03-03 2025-03-06 2025-03-13 2025-03-20 2025-03-24 2025-04-01 2025-04-09 2025-04-11"
DS="2024-12-23 2025-01-08 2025-01-17 2025-01-20 2025-02-03 2025-02-12 2025-02-24 2025-02-28 2025-03-03 2025-03-06 2025-03-24 2025-03-26 2025-04-09 2025-04-11"
DM="2025-03-13 2025-04-01 2025-03-20 2025-01-23 2025-01-16"
run() { echo "== $(date +%T) $1"; ./gpu_job.sh "$1"; echo "== $(date +%T) exit $?"; }
run "$P lm_feats.py aliked1024 fluke $FL && $P lm_match.py aliked1024 fluke $FL && $P lm_match.py aliked1024 dorsal $DS --no-mirror"
run "$P lm_match.py aliked1024 dorsal $DM --no-mirror"
echo "== $(date +%T) queue done"
