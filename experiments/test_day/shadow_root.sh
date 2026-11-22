#!/usr/bin/env bash
# Switch prep (2026-11-20): the shadow day root for the mask switch. Every day the app serves, rebuilt beside the live
# root with the candidate mask S6h (seg_splash_s6 last.pt, hysteresis 0.6 / 0.1) and rule F (canon.predict
# --use-fallback) at the source, the live chain behind it unchanged. Nothing live is written: live day folders are read
# only (images are linked, timestamps / EXIF caches copied), no model or threshold changes.
#
#   experiments/test_day/shadow_root.sh SHADOW_ROOT DAY ...        (DAY = a live day under experiments/days_all, or "testday")
#
# SHADOW_ROOT holds days/<day> (the day root the app would read), testday/<day> (the unculled test day, a root of its own
# until it is registered), markers/ (one file per finished day, resumable: a day
# whose marker exists is skipped; a day that was started but not finished is wiped and redone), logs/, _code/ (a copy
# of model_fine_tuning whose canon/predict.py has patch_canon_predict_s6h.diff applied; everything else symlinked to
# this repo), _notes_rank/ (a copy of the log_cards readers cache, so a cache miss can never refit the live one).
# GPU steps run under the shared lock, at most one day per hold. Stops before a day when /home would fall under
# MIN_FREE_GB (default 25) free.
set -uo pipefail
cd "$(dirname "$0")/../.."
REPO=$PWD; SR="${1:?shadow root}"; shift
PY=$REPO/.venv/bin/python; export PYTHONPATH=.; LOCK="${GPU_LOCK-flock /var/tmp/sat3737-parts-cache/gpu_train.lock}"   # GPU_LOCK="": the caller holds the lock
SEG=$REPO/experiments/seg_splash_s6/checkpoints/last.pt; CANON=$REPO/experiments/canon_r0304_ft4/checkpoints/best.pt; CSE=experiments/cse_r04_ft2/checkpoints/best.pt
LAB=model_fine_tuning/splits/labels_all_days.json
TESTDAY=2026-04-07; TESTLIVE=/home/sat3737/NOAA_whales/moonshot_cache/test_days/2026-04-07/days/2026-04-07
MIN_FREE_GB="${MIN_FREE_GB:-25}"
mkdir -p "$SR/days" "$SR/markers" "$SR/logs"
[ -f "$SR/_code/model_fine_tuning/canon/predict.py" ] || { echo "no patched code overlay in $SR/_code"; exit 2; }
grep -q -- "--use-fallback" "$SR/_code/model_fine_tuning/canon/predict.py" || { echo "overlay predict.py is not patched"; exit 2; }
export WHALE_NOTES_RANK="$SR/_notes_rank"
[ -d "$WHALE_NOTES_RANK" ] || cp -r ../moonshot_cache/notes_rank "$WHALE_NOTES_RANK"
t() { date +%s; }

for ARG in "$@"; do
  if [ "$ARG" = testday ]; then DAY=$TESTDAY; L=$TESTLIVE; DR="$SR/testday"; else DAY=$ARG; L=$(realpath experiments/days_all/$DAY); DR="$SR/days"; fi
  D="$DR/$DAY"; M="$SR/markers/$DAY"; mkdir -p "$DR"
  [ -f "$M.done" ] && { echo "== $DAY done before"; continue; }
  [ -d "$L/images" ] || { echo "== $DAY: no live day folder $L"; continue; }
  n=$(wc -l < "$L/images/index.jsonl")
  free=$(df --output=avail -BG /home | tail -1 | tr -dc 0-9); need=$(( (n * 5 + 1023) / 1024 + 1 ))
  if [ $((free - need)) -lt "$MIN_FREE_GB" ]; then echo "STOP: /home has ${free} GB free, $DAY needs ~${need} GB, floor ${MIN_FREE_GB} GB"; exit 3; fi
  if [ -f "$M.started" ]; then echo "== $DAY was interrupted: wiping $D"; rm -rf "$D"; fi
  date > "$M.started"; T0=$(t); echo "== $(date +%T) $DAY ($n photos, live $L, /home ${free} GB free)"
  mkdir -p "$D"
  # mask-independent inputs: decoded photos linked (hard links when on the same file system, else a symlink to the
  # live folder), capture times and the RAW EXIF cache copied
  if [ ! -e "$D/images" ]; then
    if [ "$(stat -c %d "$L/images")" = "$(stat -c %d "$DR")" ]; then cp -al "$(realpath "$L/images")" "$D/images"; else ln -s "$(realpath "$L/images")" "$D/images"; fi
  fi
  cp "$L/timestamps.json" "$D/"; [ -f "$L/images.log" ] && cp "$L/images.log" "$D/"
  echo "== $(date +%T) $DAY canon (S6h + rule F)"; s=$(t)
  [ -f "$D/canon/points.jsonl" ] || ( cd "$SR/_code" && $LOCK $PY -m model_fine_tuning.canon.predict --seg-checkpoint $SEG --checkpoint $CANON --images "$D/images" --out "$D/canon" --max-side 1280 --hysteresis 0.6,0.1 --use-fallback > "$D/canon.log" 2>&1 )
  [ -f "$D/canon/points.jsonl" ] || { echo "FAIL $DAY canon"; exit 4; }; echo "   canon $(( $(t) - s )) s"
  echo "== $(date +%T) $DAY cse"; s=$(t)
  [ -f "$D/cse/points.jsonl" ] || $LOCK $PY -m model_fine_tuning.canon.predict_cse --checkpoint $CSE --canon-dir "$D/canon" --images "$D/images" --out "$D/cse" --overlays 0 --as-canon > "$D/cse.log" 2>&1
  [ -f "$D/cse/points.jsonl" ] || { echo "FAIL $DAY cse"; exit 4; }; echo "   cse $(( $(t) - s )) s"
  echo "== $(date +%T) $DAY parts"; s=$(t)
  [ -f "$D/parts/index.jsonl" ] || $PY -m model_fine_tuning.cluster.parts --index "$D/images/index.jsonl" --canon-dir "$D/cse" --out "$D/parts" --size 512 --workers 10 --fluke-min-fraction 0.3 --fin-min-fraction 0.03 2>&1 | grep -v -i "warn\|_check_multiprocessing" | tail -1
  echo "   parts $(( $(t) - s )) s"
  if [ -f "$L/parts_hr/index.jsonl" ]; then
    mkdir -p "$D/safe"; [ -f "$L/safe/exif.json" ] && [ ! -f "$D/safe/exif.json" ] && cp "$L/safe/exif.json" "$D/safe/"
    s=$(t); experiments/test_day/run_pipeline.sh "$D"; echo "   run_pipeline $(( $(t) - s )) s"
    [ -f "$D/safe/link_scores.json" ] || { echo "FAIL $DAY run_pipeline"; exit 4; }
    if [ -f "$L/safe/fin_hw/hwb6_s16_0.npz" ]; then          # the fine-tuned fin models (off by default in the app: WHALE_VIEW_MATCH_FILE / WHALE_HINT_FIN_FILE)
      $LOCK env OMP_NUM_THREADS=8 $PY -m model_fine_tuning.cluster.fin_hw embed --day-dir "$D" --models hwb6_s16_0,hwb6_s16_1 2>&1 | grep -v -i warn | tail -1
    fi
    if [ -f "$L/cluster_v3/scores_student.npz" ]; then          # the /pairs queue order of the 7 split days (scripts/pairnet_chain.sh's last two steps)
      echo "== $(date +%T) $DAY pairs-page scores (_student)"; s=$(t); TMP="$D/_student_tmp"
      $LOCK $PY -m model_fine_tuning.cluster.cluster_day channels --day-dir "$D" --out "$TMP" --sources id:experiments/pair_student/checkpoints/best.pt --local "" --candidates 0 2>&1 | grep -v -i "warn\|nanmean" | tail -1
      $PY -m model_fine_tuning.cluster.cluster_day cluster --day-dir "$D" --out "$TMP" --labels $LAB --tag _student --use id:pair_student --thr-offset 0.0 --cannot-link none --max-link-cost 1e9 2>&1 | grep -v -i "warn\|nanmean\|instances from\|using channels" | tail -2
      for f in scores_student.npz metrics_student.json tracklets_student.json; do [ -f "$TMP/$f" ] && mv "$TMP/$f" "$D/cluster_v3/"; done; rm -rf "$TMP"
      echo "   pairs scores $(( $(t) - s )) s"
    fi
    echo "== $(date +%T) $DAY fin_sketch points / draw"; s=$(t)
    $LOCK $PY -m model_fine_tuning.cluster.fin_sketch points --root "$DR" --days $DAY --predict-all 2>&1 | grep -v -i warn | tail -1
    $PY -m model_fine_tuning.cluster.fin_sketch draw --root "$DR" --days $DAY --workers 6 2>&1 | grep -v -i "warn\|nanmedian" | tail -1
    echo "   fin_sketch $(( $(t) - s )) s"
    $PY -m model_fine_tuning.cluster.log_cards write --day-dir "$D" 2>&1 | tail -1
  fi
  rm -f "$M.started"; echo "$(date '+%F %T') $(( $(t) - T0 )) s $n photos" > "$M.done"
  echo "DAY_DONE $DAY $(( $(t) - T0 )) s ($n photos) $(du -sh "$D" | cut -f1)"
done
echo "SHADOW_ROOT_DONE $(date +%T)"
