# Unsorted test day: can the pipeline tell whale from crew, rail, island and empty water?

Scripts for the first unculled day, 2026-04-07 (C1, 427 RAW frames). Text only lives here. Photos, sheets, labels and every
per-day output go to the untracked folder `moonshot_cache/test_days/2026-04-07/` or to a session scratch folder. Photos are
named by their 3-digit sequence number in anything written here. Nothing was fitted or tuned on this day.
The open-vocabulary prompts and rules (`prompts.json`) were written before any output on it was looked at.

    R=/home/sat3737/NOAA_whales/moonshot_cache/test_days/2026-04-07; D=$R/days/2026-04-07
    # a source folder of per-file symlinks to the read-only RAW folder, so the image names match the other days
    flock /var/tmp/sat3737-parts-cache/gpu_train.lock test_scripts/bash/process_day_v2.sh $R/src/2026-4-7 $R/days   # images, live mask, CSE, charts
    experiments/test_day/run_pipeline.sh $D               # the rest of the live new-day chain (parts_hr ... link_scores pairs)
    .venv/bin/python experiments/test_day/decode_fullres.py --day-dir $D --out $R       # full-res JPEGs + exif.tsv
    .venv/bin/python experiments/test_day/sheets.py plain --day-dir $D --out SCRATCH/plain   # blind sheets for the by-eye pass
    .venv/bin/python experiments/test_day/build_truth.py --root $R                      # truth_raw.txt + truth_overrides.txt -> truth.tsv
    flock ... .venv/bin/python experiments/test_day/openvocab.py --day-dir $D --out $R/openvocab.json   # CLIP / OWLv2 (public weights)
    .venv/bin/python experiments/test_day/nw_probe.py --sub $R/nw_subclass_train.json --day-dir $D --out $R/nw_probe.json   # fitted on train days only
    flock ... env PYTHONPATH=. .venv/bin/python experiments/test_day/candidate_mask.py --day-dir $D --out $R/cand_s6         # candidate mask, not live
    .venv/bin/python experiments/test_day/analyze.py --root $R --keys INST_KEYS.json    # photos.tsv, instances.tsv, measures.json
    .venv/bin/python experiments/test_day/result_sheets.py --root $R --keys INST_KEYS.json --out SHEETS
    .venv/bin/python experiments/test_day/timeline.py --root $R --triplog ../Whale-Review-Web/data/trip_log/2026-04-07.json --out SHEETS/testday_timeline.jpg
    .venv/bin/python experiments/test_day/marker_sheet.py $D $R SHEETS/testday_markers.jpg
    .venv/bin/python experiments/test_day/s6_sheet.py $R SHEETS/testday_seg_candidate_s6.jpg SEQ ...
    .venv/bin/python experiments/test_day/run06_examples.py $D SCRATCH/run06_examples

`INST_KEYS.json` lists the day's instance keys in photo order, and `inst_eye.txt` labels them by index. The `labeling_day.json` the
timeline reads comes from `backend.labeling.load_day` run in a scratch copy of Whale-Review-Web/backend (WHALE_DAYS_ROOT,
WHALE_RAW_ROOT, WHALE_LABELING_STATE pointed at the test folders), so the live app and its data folder are untouched.
The results are in `moonshot_cache/test_days/2026-04-07/tables.md`.

## Round 2 (2026-11-20): candidate mask end to end, ready-made classes, part candidate, tape rule

The rules were registered before any round-2 model ran: `round2_rules.md`. The models (S6h, Mask2Former Swin-L, W3c) were chosen after
round 1's results on this day, which weakens the day as a test for them. Results are in the untracked `moonshot_cache/test_days/2026-04-07/round2/tables.md`.

    R=/home/sat3737/NOAA_whales/moonshot_cache/test_days/2026-04-07; R2=$R/round2; LOCK="flock /var/tmp/sat3737-parts-cache/gpu_train.lock"
    experiments/test_day/run_s6.sh $R          # S6h swapped into canon.predict -> R2/days_s6/<day> (live chain behind it), then instances_v2 (gap 0.35)
    # Labeling Assistant load in a scratch copy of Whale-Review-Web (backend/, frontend/, data/trip_log/), serve.sh's defaults:
    WHALE_TRIP_LOG=1 WHALE_VIEW_MATCH_FILE=view_match_hwb6.npz WHALE_HINT_FIN=mean WHALE_HINT_FIN_RULE=merge_hwb6 \
      WHALE_DAYS_ROOT=$R2/days_s6 WHALE_RAW_ROOT=$R/src WHALE_LABELING_STATE=WEBCOPY/state \
      .venv/bin/python experiments/test_day/labeling_load.py --webcopy WEBCOPY --day 2026-04-07 --out $R2/labeling_day_s6.json
    PYTHONPATH=. .venv/bin/python experiments/test_day/s6_eye.py prep --root $R --sheets SCRATCH/eye   # tiles for the by-eye pass -> s6_eye.txt, s6_labels.json
    .venv/bin/python experiments/test_day/analyze_s6.py --root $R                                     # s6_photos.tsv, s6_instances.tsv, s6_measures.json
    .venv/bin/python experiments/test_day/s6_e2e_sheet.py --root $R --out SHEETS/testday_s6_end_to_end.jpg
    $LOCK .venv/bin/python experiments/test_day/readymade.py --day-dir $R/days/2026-04-07 --out $R2/readymade   # Mask2Former COCO / ADE / COCO-instance
    .venv/bin/python experiments/test_day/readymade_analyze.py --root $R --rm $R2/readymade --out $R2
    .venv/bin/python experiments/test_day/readymade_sheets.py --root $R --out SHEETS
    $LOCK env PYTHONPATH=. .venv/bin/python experiments/test_day/parts_w3c.py --arm live --day-dir $R/days/2026-04-07 --shadow $R2/w3c_live
    $LOCK env PYTHONPATH=. .venv/bin/python experiments/test_day/parts_w3c.py --arm s6 --day-dir $R2/days_s6/2026-04-07 --shadow $R2/w3c_s6
    .venv/bin/python experiments/test_day/parts_analyze.py --root $R; .venv/bin/python experiments/test_day/parts_sheet.py --root $R --out SHEETS/testday_parts_on_nonwhale.jpg
    .venv/bin/python experiments/test_day/tape_rule.py --root $R --rm $R2/readymade --out $R2/tape_rule.tsv   # registered rule, run once

## Switch prep (2026-11-20): the candidate mask on the val days, loss kinds, "why is this photo empty"

Rules: untracked `moonshot_cache/test_days/switch_prep/rules_switch.md` (written before the runs, addendum for rule F before
its one test-day run). Results: `switch_prep/tables.md`. Nothing live is written; eval days are refused by every script.

    SP=/home/sat3737/NOAA_whales/moonshot_cache/test_days/switch_prep
    experiments/test_day/switch_val.sh $SP/days_s6 full 2025-01-15 2025-02-27 2025-03-14   # S6-src: whole live chain behind the candidate mask
    experiments/test_day/switch_val.sh $SP/days_s6 inst 2025-01-02 2025-03-18              # click days: canon + cse + parts_hr
    PYTHONPATH=. .venv/bin/python experiments/test_day/switch_eval.py days --s6 $SP/days_s6 --out $SP/eval_s6.json      # live vs S6 with the catalogue
    PYTHONPATH=. .venv/bin/python experiments/test_day/switch_eval.py clicks --s6 $SP/days_s6 --out $SP/eval_clicks.json # flukes / main whales (needs the filter on the click days)
    # Labeling Assistant load in a scratch web copy (serve.sh defaults, WHALE_DAYS_ROOT / WHALE_LABELING_STATE in scratch), then saved labels:
    .venv/bin/python experiments/test_day/switch_load.py --webcopy WEBCOPY --day DAY --out $SP/load/ARM_DAY.json
    .venv/bin/python experiments/test_day/switch_attach.py LIVE.json CAND.json [STATE.json]
    .venv/bin/python experiments/test_day/fallback_use.py --src S6DAY --dst ARMDAY; experiments/test_day/switch_fallback.sh S6ROOT ARMROOT DAY   # rule F
    flock LOCK .venv/bin/python experiments/test_day/readymade_day.py --day-dir DAY --out $SP/readymade/DAY   # ADE20K maps by row
    .venv/bin/python experiments/test_day/reasons.py --day-dir ARMDAY --live-dir LIVEDAY --ade ADE [--ade-key row|seq] --out OUT.json
    PYTHONPATH=. .venv/bin/python experiments/test_day/switch_sheets.py lost|gained|tiles ...   # sheets -> scratch only
