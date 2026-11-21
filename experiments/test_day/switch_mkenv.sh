#!/usr/bin/env bash
# switch_mkenv.sh ENVDIR DAYSROOT STATE_DIR : a scratch copy of the live app (backend, frontend, tests, serve.sh) whose
# ../Dinov3-WhaleID/experiments/days_all is DAYSROOT and whose data/ holds the trip log, gps.json and the saved state
# files from STATE_DIR (a folder of <day>/{labeling.json,pair_marks.json,review_extra.json,marks.json}). Never writes live files.
set -euo pipefail
E=$1; ROOT=$2; ST=$3; W=/home/sat3737/NOAA_whales/Whale-Review-Web; R=/home/sat3737/NOAA_whales/Dinov3-WhaleID
case "$E" in */scratchpad/*) ;; *) echo "refusing: not in scratch"; exit 1;; esac
mkdir -p $E/Dinov3-WhaleID/experiments $E/webcopy/data
ln -sfn $R/model_fine_tuning $E/Dinov3-WhaleID/model_fine_tuning; ln -sfn $R/.venv $E/Dinov3-WhaleID/.venv
ln -sfn $ROOT $E/Dinov3-WhaleID/experiments/days_all; ln -sfn /home/sat3737/NOAA_whales/Whale-Image-Data-Factory $E/Whale-Image-Data-Factory
for x in experiments/pair_student_g experiments/safe_clusters; do [ -e $R/$x ] && ln -sfn $R/$x $E/Dinov3-WhaleID/$x; done
cp -r $W/backend $W/frontend $W/tests $W/serve.sh $E/webcopy/; cp -r $W/frontend_next $E/webcopy/; rm -rf $E/webcopy/backend/__pycache__ $E/webcopy/tests/__pycache__
cp -r $W/data/trip_log $E/webcopy/data/
for d in $W/data/20*/; do b=$(basename $d); mkdir -p $E/webcopy/data/$b; [ -f $d/gps.json ] && cp $d/gps.json $E/webcopy/data/$b/; done
for d in $ST/20*/; do b=$(basename $d); for f in labeling.json pair_marks.json review_extra.json marks.json; do [ -f $d/$f ] && cp $d/$f $E/webcopy/data/$b/; done; done
echo "env $E -> $ROOT (state from $ST)"
