#!/usr/bin/env bash
# v3 day clustering over the eval + val days (cluster/cluster_day.py), incremental: channels accumulate per day.
#   scripts/cluster_v3_all.sh channels "<sources>" "<local>" [candidates] [candidate_sources]   e.g. channels "md,id:experiments/id_v3/checkpoints/best.pt" ""
#   scripts/cluster_v3_all.sh fuse <fusion name> [channels]           fits experiments/<fusion name>.json on the val days
#   scripts/cluster_v3_all.sh cluster <score tag> [--fusion F | --use S] [extra cluster_day args]   clusters all days, prints the table, tunes on val
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; export PYTHONPATH=.
LAB=model_fine_tuning/splits/labels_all_days.json; DAYS_JSON=model_fine_tuning/splits/days.json; TAG="${V3_TAG:-cluster_v3}"
DAYS="2024-12-18 2024-12-26 2025-02-19 2025-03-21 2025-01-15 2025-02-27 2025-03-14"
cmd="$1"; shift
case "$cmd" in
  channels)
    SRC="$1"; LOCAL="${2:-}"; CAND="${3:-30}"; CSRC="${4:-}"; LPARTS="${5:-}"
    for d in $DAYS; do
      echo "== $d channels [$SRC] local [$LOCAL]"; date
      $PY -m model_fine_tuning.cluster.cluster_day channels --day-dir experiments/days_all/$d --out experiments/days_all/$d/$TAG --sources "$SRC" --local "$LOCAL" --candidates "$CAND" \
          ${CSRC:+--candidate-sources "$CSRC"} ${LPARTS:+--local-parts "$LPARTS"} 2>&1 | grep -v -i "warn\|nanmean" | tail -3
    done ;;
  fuse)
    NAME="$1"; CH="${2:-}"; shift; [ $# -gt 0 ] && shift
    $PY -m model_fine_tuning.cluster.fuse --root experiments/days_all --tag "$TAG" --days $DAYS_JSON --labels $LAB --out experiments/$NAME.json ${CH:+--channels "$CH"} "$@" 2>&1 | grep -v -i "warn" ;;
  cluster)
    STAG="$1"; shift
    for d in $DAYS; do
      echo "== $d"; $PY -m model_fine_tuning.cluster.cluster_day cluster --day-dir experiments/days_all/$d --out experiments/days_all/$d/$TAG --labels $LAB --tag "$STAG" "$@" 2>&1 | grep -v -i "warn\|nanmean\|instances from\|using channels"
    done
    $PY - "$TAG" "$STAG" <<'PYEOF'
import json, sys, numpy as np
tag, stag = sys.argv[1], sys.argv[2]
split = {d: "eval" for d in ("2024-12-18", "2024-12-26", "2025-02-19", "2025-03-21")}; split.update({d: "val" for d in ("2025-01-15", "2025-02-27", "2025-03-14")})
rows = {}
print(f"{'day':11s} split photos animals clusters    ARI  pairF1 purity oracleARI | groups   gARI")
for d, sp in split.items():
    try:
        m = json.load(open(f"experiments/days_all/{d}/{tag}/metrics{stag}.json")); a = m["animals"]; g = m["groups"]
        print(f"{d:11s} {sp:5s} {m['photos']:6d} {a['individuals']:7d} {a['clusters']:8d} {a['ari']:6.3f} {a['pair_f1']:7.3f} {a['cluster_purity']:6.3f} {max(s['ari'] for s in m['oracle_sweep_animals']):9.3f} | {g['individuals']:6d} {g['ari']:6.3f}")
        rows.setdefault(sp, []).append(a["ari"])
    except Exception as e:
        print(d, "missing", str(e)[:50])
print("mean ARI:", {k: round(float(np.mean(v)), 3) for k, v in rows.items()})
PYEOF
    echo "== tune on val (eval for the record)"; $PY -m model_fine_tuning.cluster.tune_v3 --root experiments/days_all --tag "$TAG" --days $DAYS_JSON --labels $LAB --score-tag "$STAG" 2>&1 | grep -v -i warn ;;
esac
