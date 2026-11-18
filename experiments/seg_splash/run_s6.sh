#!/bin/bash
# S6 = S5 with labels that keep thin structures and spare the waterline (why_drop_train.json: thin blobs had 2.7 ring
# negatives within 8 px per positive pixel, thick 0.12; S5 drops 56 % of < 6 px structure pixels and 13 % of the bottom
# fifth of the whale against 3 % of the top):
#   - pixels of a sure whale blob whose local thickness (2 x the largest inscribed radius within 15 px) is < 12 px are
#     positive without erosion (S5: eroded by 1 px everywhere)
#   - the background ring (5-12 px outside a sure whale) only where r04 P < 0.05 (S5: < 0.25), and on the waterline
#     side (below the blob's lower edge in that column) it starts at 10 px instead of 5
# Same recipe and seed as S5 (seed 0, init seg_r05_ft4 best, 8 epochs, real share 0.3). Every GPU step under the lock.
# Then scored like S5: blob tables + views (s6, plain 0.5 and s6h = hysteresis 0.6/0.1), end to end with h0.6_0.1 and
# fluke-join gap 0.35, clicks per day, new blobs, lost instances -> experiments/seg_splash/s6_tables.md, logs; marker.
set -u
cd /home/sat3737/NOAA_whales/Dinov3-WhaleID
LOCK=/var/tmp/sat3737-parts-cache/gpu_train.lock; C=/var/tmp/sat3737-parts-cache/seg_splash; LOGS=$C/logs; PY=.venv/bin/python; export PYTHONPATH=.
SP=/tmp/claude-1072903897/-home-sat3737-NOAA-whales/4fbba7ff-3ba5-4b50-9cb7-970c81250409/scratchpad/seg_splash
rm -f $C/s6.done
echo "$(date +%T) build real_s6"
[ -f $C/real_s6/index.json ] || $PY -m model_fine_tuning.seg_real.data build --strict --ring-in 5 --erode 1 --thin 12 --ring-p 0.05 --below-in 10 --out real_s6 > $LOGS/build_s6.log 2>&1 || { echo build failed; exit 1; }
echo "$(date +%T) train seg_splash_s6"
[ -f experiments/seg_splash_s6/checkpoints/last.pt ] || flock $LOCK $PY -m model_fine_tuning.seg_real.train --name seg_splash_s6 --seed 0 --real-p 0.3 --real-dir $C/real_s6 \
  --init experiments/seg_r05_ft4/checkpoints/best.pt --epochs 8 --eval-every 4 --workers 6 --batch-size 8 > $LOGS/train_s6.log 2>&1 || { echo "$(date +%T) train failed"; exit 1; }
echo "$(date +%T) predict s6"; flock $LOCK $PY -m model_fine_tuning.seg_real.predict --checkpoint experiments/seg_splash_s6/checkpoints/last.pt --name s6 > $LOGS/pred_s6.log 2>&1
$PY - <<'PYEOF'
import cv2, numpy as np
from multiprocessing import Pool
from model_fine_tuning.seg_real.predict import PRED, load_prob
from model_fine_tuning.seg_real.followup import mask_of
def f(p):
    day, stem = p.parent.name, p.stem; o = PRED / 's6h' / day / f'{stem}.png'; o.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(o), (mask_of(load_prob('s6', day, stem), 'h0.6_0.1') * 255).astype(np.uint8)); return 1
with Pool(8) as P: print('s6h masks', sum(P.map(f, sorted(PRED.glob('s6/*/*.png')), chunksize=16)))
PYEOF
echo "$(date +%T) score"
$PY -m model_fine_tuning.seg_real.score --names s6,s6h > $LOGS/score_s6.log 2>&1
$PY -m model_fine_tuning.seg_real.extra --names s6,s6h >> $LOGS/score_s6.log 2>&1
$PY -m model_fine_tuning.seg_real.why drop --names s5,s6,s1 > $LOGS/why_s6.log 2>&1
echo "$(date +%T) e2e s6 h0.6_0.1"; flock $LOCK $PY -m model_fine_tuning.seg_real.followup e2e --name s6 --setting h0.6_0.1 --checkpoint experiments/seg_splash_s6/checkpoints/last.pt > $LOGS/e2e_s6h.log 2>&1
$PY -m model_fine_tuning.seg_real.followup join --name s6_h0.6_0.1 --gaps 0.35 > $LOGS/join_s6.log 2>&1
$PY -m model_fine_tuning.seg_real.e2e score --names s6_h0.6_0.1_g0.35 > $LOGS/e2e_score_s6.log 2>&1
$PY $SP/clicks_by_day.py ref,s5_h0.6_0.1_g0.35,s6_h0.6_0.1_g0.35 > $LOGS/clicks_s6.log 2>&1
$PY -m model_fine_tuning.seg_real.new_blobs find --names s6_h0.6_0.1_g0.35 > $LOGS/nb_s6.log 2>&1
$PY -m model_fine_tuning.seg_real.lost find --names s6_h0.6_0.1_g0.35 > $LOGS/lost_s6.log 2>&1
$PY -m model_fine_tuning.seg_real.sheets --new s6h --sets lowiou,easy,splash > $LOGS/sheets_s6.log 2>&1
$PY -m model_fine_tuning.seg_real.report --names r04,s1,s5,s5h,s6,s6h --out s6_tables.md > /dev/null 2>&1
{ cat $LOGS/e2e_score_s6.log; cat $LOGS/clicks_s6.log; cat $LOGS/nb_s6.log $LOGS/lost_s6.log; grep '"val"' $LOGS/train_s6.log; } >> experiments/seg_splash/s6_tables.md
touch $C/s6.done; echo "$(date +%T) s6 done"
