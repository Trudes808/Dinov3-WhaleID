#!/usr/bin/env bash
# Unit tests: geometry conventions, dataset invariants, model shapes. ~1 minute.
set -euo pipefail
cd "$(dirname "$0")/../.."
.venv/bin/python -m unittest discover -s test_scripts/python -t . -v
