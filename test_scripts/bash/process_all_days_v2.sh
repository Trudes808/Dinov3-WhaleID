#!/usr/bin/env bash
# Process every day folder of the upload with process_day_v2.sh: held-out eval days first, then val, then train
# (model_fine_tuning/splits/days.json); folders without a leading date or without RAW files are skipped.
#   nohup setsid test_scripts/bash/process_all_days_v2.sh "/datasets/NOAA_whales_data/Lightroom images/Topside research/2025 Research" experiments/days_all > experiments/process_all_days_v2.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
SRC_ROOT="${1:?source root with one folder per day}"; OUT_ROOT="${2:-experiments/days_all}"
order=$(python3 -c "
import json; d=json.load(open('model_fine_tuning/splits/days.json')); print(' '.join(d['eval']+d['val']+d['train']+d.get('excluded_small_days',[])))")
declare -A dir_of
for day_dir in "$SRC_ROOT"/*/; do
    name="$(basename "$day_dir")"; date="${name%% *}"
    [[ "$date" =~ ^[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}$ ]] || { echo "skip $name (no date)"; continue; }
    IFS=- read -r Y M D <<< "$date"; iso=$(printf "%04d-%02d-%02d" "$((10#$Y))" "$((10#$M))" "$((10#$D))"); dir_of[$iso]="$day_dir"
done
for iso in $order; do
    day_dir="${dir_of[$iso]:-}"; [ -n "$day_dir" ] || { echo "no folder for $iso"; continue; }
    [ -f "$OUT_ROOT/$iso/timestamps.json" ] && { echo "$iso already done"; continue; }
    n_raw=$(find "$day_dir" -type f \( -iname '*.cr2' -o -iname '*.cr3' -o -iname '*.dng' \) | wc -l)
    [ "$n_raw" -gt 0 ] || { echo "skip $iso (no RAW)"; continue; }
    echo "$(date -u +%FT%TZ) processing $iso ($n_raw RAW files)"
    test_scripts/bash/process_day_v2.sh "$day_dir" "$OUT_ROOT" 2>&1 | sed "s/^/  /"
done
echo "$(date -u +%FT%TZ) ALLDONE"
