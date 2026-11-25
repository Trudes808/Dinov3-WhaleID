# Removal rule v2: propose a photo for removal only when nothing sees a whale (2026-11-22)

The Labeling Assistant shows, on an unculled day, a tray of photos "proposed for removal" (the researcher's cull: marker
close-ups, crew, plain water, land). The researcher confirms each; nothing is removed by the page. Rule v1
(`experiments/test_day/reasons.py`) proposed every photo where our whale mask kept no instance. The user looked at it on
the development day 2026-04-07 and said: "The water only removal candidates seem to sometimes be just water but a lot of
them have a bit of whale." A photo with a bit of whale must not be proposed: a single frame can link two stacks, and a
blow or a footprint tells the researcher an animal was there.

Text only lives here (this repo is public). Photos, sheets, per-photo truth and readings are in the untracked
`moonshot_cache/removal_v2/` and `moonshot_cache/test_days/2026-04-07/`; photos are named by their 3-digit sequence number.

## The rule in words (`model_fine_tuning/seg_multi/removal.py`, frozen in `moonshot_cache/sam3_labeller/v3/removal_FREEZE.json`)

1. A marker photo (rail + tape, bollard, land; from the marker records) is proposed as "marker", as before.
2. A photo where our whale mask finds a whale (S6h hysteresis 0.6 / 0.1 paints >= 50 px at 1280 px, or the live chain
   keeps an instance) is never proposed.
3. Every other photo is read by several independent readings. It is PROPOSED (reason crew if a person covers >= 0.5 %
   of the frame, land if land covers >= 25 %, else water; shares from the scene model) only when ALL of them see no
   trace of a whale:
   - S6h at a low threshold: P(whale) >= 0.25 on >= 20 px
   - the old live mask (seg_r04_ft4) keeps a whale-scored instance (only where that chain was run)
   - SAM 3 "whale" / "humpback whale" / "marine mammal": any instance with score >= 0.10
   - SAM 3 "whale tail" / "whale fluke" / "dorsal fin": >= 0.20
   - SAM 3 "whale blow" / "whale spout" / "whale splash" / "splash": >= 0.20
   - SAM 3 "whale under water" / "submerged whale": >= 0.40
   - white water: foam (pixels much brighter than their surroundings and pale) outside crew / boat / land >= 1 % of the frame
   - glow under water: a turquoise patch brighter than the water around it (a white flipper or belly under the surface) >= 0.02 % of the frame
4. If one or more readings see a trace (or SAM 3 could not be read), the photo goes to the group "faint whale?": shown
   on the page in its own strip, never proposed, no "remove all" button; the researcher can still remove one photo.

Thresholds were set on the development day 2026-04-07 against the second by-eye truth below, at the lowest level
where each reading stays silent on all 30 of that day's photos with nothing of a whale (markers included; white water
on its 7 non-marker ones, since a land marker's coastal surf reaches 2.4 % of the frame and markers are proposed anyway). The SAM 3 phrases were written before any
output of the new phrases was looked at (`prompts_trace.json`).

## Truth on the development day (second by-eye pass)

`moonshot_cache/test_days/2026-04-07/truth_whale_trace.tsv` (beside `truth.tsv`, which is unchanged): every photo of the
day with a whale-trace class (above / awash or just under the surface / blow, splash or footprint with no body / none),
a sureness and the first-pass class. 103 photos were looked at again (every photo without a kept S6h instance on the
live root, the 12 "unsure" of rule v1, every first-pass water / under / crew_nowhale / whale_crew photo, and 077), at
full resolution and with the exposure raised and lowered and local contrast up (`eye_labels.tsv`; the view tool is in
the session scratch folder). The other 324 photos keep the first pass (whale). The truth is one helper's eye, not the
researcher's; the labeller saw no model output.

The first pass had called 12 photos "water only"; all 12 hold a whale trace on the second look (3 above / awash
backs seen with the exposure lowered inside the splash line, 9 footprints, splashes or bodies just under the surface).
25 of the 103 changed class (`removal2_truth_changes.jpg` in the scratch sheets); 13 flipped between whale and no whale
(12 water -> trace, 277 whale -> trace, unsure). After the pass, the day has 30 photos with nothing of a whale:
the 23 markers and 7 crew photos (drone launch / recovery, a head close-up). No open-water photo is empty.

## Results (measured; the truth is a helper's eye)

Development day 2026-04-07, 427 photos, rule v1 (the live file before 2026-11-22 23:59) against v2:

| | v1 | v2 |
|---|---|---|
| proposed | 54 (marker 23, crew 14, water 17) | 30 (marker 23, crew 7) |
| proposals holding any whale trace | **24** (crew 7, water 17) | **0** |
| photos with nothing of a whale that are proposed | 30 of 30 | 30 of 30 |
| "unsure" / "faint whale?" (shown, not proposed) | 12, all whale traces | 33, all whale traces (above 4, awash 15, blow / splash / footprint 14; by-eye sure 14, likely 16, unsure 3) |
| markers | 23 | 23, unchanged |

(077, 300, 321 left both groups: the live chain keeps a whale instance there since rule F.) Every "faint whale?" photo
is seen by at least two readings; which readings fire on the 33: SAM 3 blow / splash 32, white water 33, SAM 3 whale 15,
SAM 3 under water 15, SAM 3 fluke / fin 11, old live mask 10.

Culled catalogue days (every photo kept by the researcher, so every proposal is a false one). Thresholds were not set
on these days, but their candidate photos were looked at while the rule was written (the four rule-v1 false proposals of
val days 2025-01-15 and 2025-03-14 led to the "glow" reading and to the SAM 3 trace phrases): a weaker check than the
held-out day.

| day | photos | no S6h whale | v1 proposed | v2 proposed | v2 faint whale? |
|---|---|---|---|---|---|
| 2025-01-15 (val) | 166 | 3 | 2 | 0 | 3 |
| 2025-02-27 (val) | 329 | 0 | 0 | 0 | 0 |
| 2025-03-14 (val) | 389 | 4 | 4 | 0 | 4 |
| 2025-01-02 (train) | 981 | 8 | - | 0 | 8 |
| 2025-01-17 (train) | 333 | 2 | - | 0 | 2 |
| 2025-02-12 (train) | 378 | 1 | - | 0 | 1 |
| 2025-03-06 (train) | 291 | 0 | - | 0 | 0 |
| 2025-03-18 (train) | 754 | 1 | - | 0 | 1 |
| total | 3 621 | 19 | 6 of 884 (val) | **0** | **19** (val 7, train 12) |

By eye the 19 are whale photos (breaches, blows, backs in wash, whales under the bow), one is crew over dark water.
On the train days the scene model is v3a, which saw these days in training (it only gives the crew / land shares and
the mask for the foam and glow readings). SAM 3 was run on the CPU for this work (float32, `run_sam3_cpu.py`): the shared
GPU lock was held for over an hour by a hung job of another thread; on the phrases cached from the earlier GPU run
(fp16) the CPU scores agree to about 0.01.

Not measured: the researcher's own cull. The held-out day 2026-03-30 with its catalogue is the test (`score_cull.py`).

## Live (2026-11-22 / 23)

`Whale-Review-Web/data/markers/2026-04-07/markers.json` re-made with `markers_adapter.py v2` (additive: `faint`,
`removal_v1`, `removal_rule`; markers and their removal unchanged). The Labeling Assistant's removal tray shows a
separate strip "faint whale? · not proposed" (no remove-all; one remove button per photo; the readings that saw a
trace on each card). Backend: `read_markers` passes `faint`; a removed photo may carry reason "faint". Test:
`tests/labeling_markers_e2e.py` (48 checks, 9 new).

## Commands

    M=/home/sat3737/NOAA_whales/moonshot_cache; R=$M/removal_v2
    # readings of the candidate photos (no S6h whale) of a day already in the live root, from cached maps
    .venv/bin/python experiments/removal_v2/readings.py --day DAY --day-dir $M/switch_root/days/DAY --s6prob $M/sam3_labeller/s6prob/DAY \
        --scene SCENE_MAPS/DAY --kept-day $M/switch_root/days/DAY [--old-kept-day OLD_LIVE/DAY] [--sam3 CACHED --sam3-key stem|seq] \
        --stems $R/sam3lists/DAY.stems --out $R/readings/DAY.jsonl
    # SAM 3 trace phrases on the same photos (GPU, shared lock, Underwater-WhaleID venv read-only)
    flock /var/tmp/sat3737-parts-cache/gpu_train.lock /home/sat3737/NOAA_whales/Underwater-WhaleID/.venv/bin/python \
        experiments/sam3_compare/run_sam3.py --list $R/sam3lists/DAY.tsv --prompts experiments/removal_v2/prompts_trace.json --set DAY --out $R/sam3
    # old rule against v2 on the development day, and v2 on the culled days
    .venv/bin/python experiments/removal_v2/analyze.py --root $R --truth $M/test_days/2026-04-07/truth_whale_trace.tsv \
        --old $R/live_backup_*/markers_2026-04-07.json --days 2025-01-15 2025-02-27 2025-03-14 2025-01-02 2025-01-17 2025-02-12 2025-03-06 2025-03-18 --out $R/analysis.json
    # the rule itself on a day (all photos; SAM 3 on the photos without a whale is run when missing)
    .venv/bin/python -m model_fine_tuning.seg_multi.removal --day-dir DAYDIR [--markers DAY_markers.jsonl] --cache CACHE \
        --sam3-out SAM3DIR --kept-day LIVEDAY [--old-kept-day OLDDAY] --out DAY_removal.jsonl
    # into the page's records (additive fields; markers and their removal unchanged)
    $PY $M/labeling_markers/markers_adapter.py v2 --day DAY --day-dir DAYDIR --base markers.json --removal DAY_removal.jsonl --freeze FREEZE --out markers.json
    # freeze / check
    .venv/bin/python experiments/removal_v2/freeze.py write|check

### A new raw day (e.g. the held-out day), in order

    model_fine_tuning/seg_multi/run_new_day.sh "RAW_DAY_FOLDER" OUT_ROOT            # decode, S6h, scene model, marker records
    .venv/bin/python experiments/removal_v2/freeze.py check                          # removal rule unchanged since the freeze
    .venv/bin/python -m model_fine_tuning.seg_multi.removal --day-dir OUT_ROOT/days/DAY --markers OUT_ROOT/DAY_markers.jsonl \
        --cache OUT_ROOT/days/DAY/scene3 --sam3-out OUT_ROOT/days/DAY/removal_sam3 --out OUT_ROOT/DAY_removal.jsonl
    .venv/bin/python experiments/removal_v2/score_cull.py --raw-dir "RAW_DAY_FOLDER" --lrcat CATALOG.lrcat --day DAY \
        --records OUT_ROOT/DAY_removal.jsonl --out OUT_ROOT/DAY_cull_score.json --list OUT_ROOT/DAY_cull_frames.tsv

`score_cull.py`: kept frames = raw frames whose camera counter (the file name's last token, e.g. 184A1234 / HR5_3180) is
among the catalogue's photos of that day (capture date, or a folder named for the day; the raw folder's own name picks
the camera folder when the catalogue has one so named); the catalogue is opened read-only (`mode=ro&immutable=1`).
It writes, for the frames she culled, how many we propose and by which reason, and how many are "faint whale?"; for
the frames she kept, how many we wrongly propose and how many are "faint whale?". Written without opening any
held-out file; `--selftest` runs it on a made-up fixture (2025-01-15 C3 from Test.lrcat as the kept list, 40 kept +
12 invented culled frames, made-up records) and checks every count.
