#!/usr/bin/env bash
# Compare clustering-rule variants of cluster_day.py on the val days (eval for the record) from saved channels.
#   scripts/cluster_v3_variants.sh "<common args>" "<variant 1 args>" "<variant 2 args>" ...
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; export PYTHONPATH=.; LAB=model_fine_tuning/splits/labels_all_days.json; TAG="${V3_TAG:-cluster_v3}"
COMMON="$1"; shift
VAL="2025-01-15 2025-02-27 2025-03-14"; EVAL="2024-12-18 2024-12-26 2025-02-19 2025-03-21"
for v in "$@"; do
  line=""; vs=0; es=0
  for d in $VAL $EVAL; do
    m=$($PY -m model_fine_tuning.cluster.cluster_day cluster --day-dir experiments/days_all/$d --out experiments/days_all/$d/$TAG --labels $LAB --tag _var $COMMON $v 2>&1 | grep "animals:" | sed -E 's/.*ARI ([0-9.]+).*oracle ARI ([0-9.]+)/\1 \2/')
    a=${m%% *}; line="$line $d:${a:0:5}"
  done
  $PY - "$v" $VAL -- $EVAL <<'PYEOF'
import json, sys, numpy as np
v = sys.argv[1]; days = sys.argv[2:]; val = days[:days.index("--")]; ev = days[days.index("--") + 1:]
def ari(d):
    m = json.load(open(f"experiments/days_all/{d}/cluster_v3/metrics_var.json")); return m["animals"]["ari"], max(s["ari"] for s in m["oracle_sweep_animals"])
va = [ari(d) for d in val]; ea = [ari(d) for d in ev]
print(f"{v:55s} val ARI {np.mean([a for a, _ in va]):.3f} (oracle {np.mean([o for _, o in va]):.3f}) | eval ARI {np.mean([a for a, _ in ea]):.3f} (oracle {np.mean([o for _, o in ea]):.3f})  val per day {[round(a, 3) for a, _ in va]}")
PYEOF
done
