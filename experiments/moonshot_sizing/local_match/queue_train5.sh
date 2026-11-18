#!/bin/bash
# last turn: the dorsal days that queue_train4's first turn skipped (it stopped at a fluke day with no pairs)
cd "$(dirname "$0")"
P=../../../.venv/bin/python
DS="2024-12-23 2025-01-08 2025-01-17 2025-01-20 2025-02-03 2025-02-12 2025-02-24 2025-02-28 2025-03-03 2025-03-06 2025-03-24 2025-03-26 2025-04-09 2025-04-11"
echo "== $(date +%T) dorsal DS"; ./gpu_job.sh "$P lm_match.py aliked1024 dorsal $DS --no-mirror"; echo "== $(date +%T) exit $?"
