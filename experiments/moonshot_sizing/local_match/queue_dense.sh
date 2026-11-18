#!/bin/bash
cd "$(dirname "$0")"
P=../../../.venv/bin/python
V="2025-01-15 2025-02-27 2025-03-14"
echo "== $(date +%T) dense val"; ./gpu_job.sh "$P lm_dense.py 640 2 fluke $V && $P lm_dense.py 640 2 dorsal $V --no-mirror"; echo "== $(date +%T) exit $?"
