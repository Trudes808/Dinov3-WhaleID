#!/usr/bin/env bash
# Watch the transfer target and process every day folder as soon as it has been quiet for a
# while (no file changed in the last QUIET minutes), writing to OUT_ROOT/<ISO day>/.
#   nohup setsid test_scripts/bash/process_all_days.sh /datasets/NOAA_whales_data experiments/days > experiments/process_all_days.log 2>&1 &
# Stops when every day folder is processed and the source has been quiet for 2 h, or on a STOP file in OUT_ROOT.
set -uo pipefail
cd "$(dirname "$0")/../.."
SRC_ROOT="${1:?source root with one folder per day}"; OUT_ROOT="${2:-experiments/days}"; QUIET="${QUIET_MIN:-15}"
mkdir -p "$OUT_ROOT"
idle_since=""
while true; do
    [ -f "$OUT_ROOT/STOP" ] && { echo "STOP file found"; exit 0; }
    did=0
    for day_dir in "$SRC_ROOT"/*/; do
        [ -d "$day_dir" ] || continue
        name="$(basename "$day_dir")"
        [[ "$name" =~ ^[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}$ ]] || continue
        IFS=- read -r Y M D <<< "$name"; iso=$(printf "%04d-%02d-%02d" "$Y" "$M" "$D")
        [ -f "$OUT_ROOT/$iso/timestamps.json" ] && continue                       # done
        n_raw=$(find "$day_dir" -type f \( -iname '*.cr2' -o -iname '*.cr3' \) | wc -l)
        [ "$n_raw" -gt 0 ] || continue
        recent=$(find "$day_dir" -type f -mmin -"$QUIET" | head -1)
        [ -n "$recent" ] && { echo "$(date -u +%FT%TZ) $name still receiving files ($n_raw RAW so far)"; continue; }
        echo "$(date -u +%FT%TZ) processing $name ($n_raw RAW files)"
        test_scripts/bash/process_day.sh "$day_dir" "$OUT_ROOT" 2>&1 | grep -v -i "warn\|_check_multiprocessing" | sed "s/^/  /"
        did=1
    done
    if [ "$did" = 0 ]; then
        recent=$(find "$SRC_ROOT" -type f -mmin -120 2>/dev/null | head -1)
        if [ -z "$recent" ] && [ -n "$(ls -A "$SRC_ROOT" 2>/dev/null)" ]; then
            pending=$(for d in "$SRC_ROOT"/*/; do n=$(basename "$d"); [[ "$n" =~ ^[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}$ ]] || continue; IFS=- read -r Y M D <<< "$n"; iso=$(printf "%04d-%02d-%02d" "$Y" "$M" "$D"); [ -f "$OUT_ROOT/$iso/timestamps.json" ] || echo "$n"; done | wc -l)
            [ "$pending" = 0 ] && { echo "$(date -u +%FT%TZ) all days processed and source quiet for 2 h; exiting"; exit 0; }
        fi
        sleep 300
    fi
done
