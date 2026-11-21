#!/usr/bin/env bash
# switch_apptests.sh ENVDIR PORTBASE : the app's test suite in a scratch env (days root = ENVDIR/Dinov3-WhaleID/experiments/days_all,
# state = ENVDIR/webcopy/data). One log per test in ENVDIR/tests_out/, a summary line per test on stdout.
E=$1; p=$2; cd $E/webcopy || exit 1; PY=/home/sat3737/NOAA_whales/Dinov3-WhaleID/.venv/bin/python; O=$E/tests_out; mkdir -p $O
unset WHALE_DAYS_ROOT WHALE_LABELING_STATE WHALE_PAIR_LABELS WHALE_TRIP_LOG WHALE_VIEW_MATCH_FILE WHALE_HINT_FIN WHALE_HINT_FIN_RULE
run() { n=$1; shift; s=$(date +%s); timeout 1500 "$@" > $O/$n.log 2>&1; rc=$?; ok=$(grep -c "^ *ok" $O/$n.log); bad=$(grep -c -E "^ *FAIL|Traceback" $O/$n.log)
        echo "$n exit=$rc ok=$ok fail=$bad $(( $(date +%s) - s ))s | $(grep -E "FAIL|Error|passed|failed|checks" $O/$n.log | tail -2 | tr '\n' ' ' | cut -c1-200)"; }
run labeling_e2e $PY tests/labeling_e2e.py --day 2025-02-27 --port $((p+1))
run labeling_panes_e2e $PY tests/labeling_panes_e2e.py frontend 2025-02-27 $((p+2))
run interpret_models_e2e $PY tests/interpret_models_e2e.py --day 2024-12-23 --port $((p+3))
run bursts_e2e $PY tests/bursts_e2e.py --day 2025-01-02 --port $((p+4))
export WHALE_TRIP_LOG=1 WHALE_VIEW_MATCH_FILE=view_match_hwb6.npz WHALE_HINT_FIN=mean WHALE_HINT_FIN_RULE=merge_hwb6
run labeling_triplog_e2e $PY tests/labeling_triplog_e2e.py --day 2025-01-15 --port $((p+5))
run labeling_triplog_cards_e2e $PY tests/labeling_triplog_cards_e2e.py --port $((p+6))
run labeling_sketch_e2e $PY tests/labeling_sketch_e2e.py --day 2025-03-14 --bare 2025-01-15 --port $((p+7))
run interpret_e2e $PY tests/interpret_e2e.py --day 2025-02-27 --day2 2025-01-15 --port $((p+8))
run library_e2e $PY tests/library_e2e.py --day 2025-02-27 --day2 2024-12-23 --port $((p+9))
run labeling_light_e2e $PY tests/labeling_light_e2e.py frontend 2025-01-15 $((p+10))
run labeling_replay $PY tests/labeling_replay.py --days val --port $((p+11)) --json $O/replay.json
run trip_log_placement $PY tests/trip_log_placement.py --days val --json $O/placement.json
echo TESTS_DONE
