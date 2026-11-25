# SAM 3 as an offline labeller for our own multi-class segmentation (option 2 of sam3_compare), 2026-11-21/22

Question: if SAM 3 labels real training-day photos offline (person, rail, tape, deck incl. bollard, land) and our whale
mask S6h gives the whale, does OUR DINOv3 model trained on those labels segment the extra classes, keep the whale, and
find the marker photos? Text only here; label maps, per-photo tables and sheets are in the untracked
`moonshot_cache/sam3_labeller/` and the session scratch folder. Test-day photos are named by sequence number only.

Rules: `rules_labeller.md` (parts A + C 2026-11-21 05:36 before any SAM 3 output was opened; B 07:13 choice, before the
scan; B5 09:35 after the 60-label look, before training; B6 after the val check of version 1, before the test day;
D record). Hashes per part: `moonshot_cache/sam3_labeller/rules.sha256`.

**Caveats.** The test day 2026-04-07 is a weak test: it shaped which models and phrasings were tried before, and the
three chosen phrasings for rail, tape and land are the ones the comparison found AFTER seeing it (they won again here on
renders and other days, by a rule written first, from the fixed candidate list; see rules B3). Its pixel truth was drawn
with SAM 2.1, which favours SAM-family outlines. A fresh unculled day is the real test. One seed for the primary model.

## Commands
    O=/home/sat3737/NOAA_whales/moonshot_cache/sam3_labeller; LOCK=/var/tmp/sat3737-parts-cache/gpu_train.lock
    PY3=/home/sat3737/NOAA_whales/Underwater-WhaleID/.venv/bin/python   # SAM 3, used read-only
    # dev: SAM 3 (31 phrasings) on the 53 boat renders; S6 P(whale) on all train + val photos
    flock $LOCK $PY3 experiments/sam3_compare/run_sam3.py --list $O/lists/renders.tsv --prompts experiments/sam3_compare/prompts_all.json --set renders --out $O/raw
    flock $LOCK env PYTHONPATH=. .venv/bin/python experiments/sam3_labeller/s6_predict.py --days train,val
    .venv/bin/python experiments/sam3_labeller/dev_pools.py; .venv/bin/python experiments/sam3_labeller/dev_sheets.py   # blind tag sheets -> $O/dev/tags.txt by eye
    .venv/bin/python experiments/sam3_labeller/dev_choose.py                        # rule A3 -> $O/dev/choice.json, choice_table.md
    .venv/bin/python experiments/sam3_labeller/live_minus_s6.py                     # scan list helper (rule A4 / B4)
    flock $LOCK $PY3 experiments/sam3_compare/run_sam3.py --list $O/lists/scan_train.tsv --prompts $O/scan_prompts.json --set train --out $O/raw
    .venv/bin/python experiments/sam3_labeller/build_labels.py select|pseudo|s6mc|val|test   # v2 labels: THIN_BAND=1 PSEUDO_DIR=pseudo_v2 ... pseudo
    flock $LOCK env PYTHONPATH=. .venv/bin/python -m model_fine_tuning.seg_multi.train --name seg_scene_sam3v2_s0 --seed 0 --balanced \
        --pseudo-dir $O/pseudo_v2 --real-dir $O/real_s6mc            # v1: --name seg_scene_sam3_s{0,1} --pseudo-dir $O/pseudo, no --balanced
    flock $LOCK env PYTHONPATH=. .venv/bin/python experiments/sam3_labeller/predict_scene.py --ckpt CKPT --name NAME --days val,2026-04-07
    flock $LOCK env PYTHONPATH=. .venv/bin/python experiments/sam3_labeller/predict_scene.py --cost --ckpt CKPT
    .venv/bin/python experiments/sam3_labeller/score_val.py --names NAMES --out $O/score_val.json
    .venv/bin/python experiments/test_day/pixtruth_score.py --root R --out $O/pixtruth_scores --ours v2s0=$O/pixtruth_in/seg_scene_sam3v2_s0 ...
    .venv/bin/python experiments/sam3_labeller/markers.py --name NAME [--src labeller] --out $O/markers_NAME.json
    .venv/bin/python experiments/sam3_labeller/sheets.py pseudo|truth|markers ...

Code: `experiments/sam3_labeller/` (labeller, labels, scoring, sheets), `model_fine_tuning/seg_multi/` (7-class model:
the S6 recipe, head initialised from the binary whale head, CE + Dice, resumable). Checkpoints (best.pt only, 353 MB each):
`experiments/seg_scene_sam3v2_s0/`, `experiments/seg_scene_sam3_s0/`, `experiments/seg_scene_sam3_s1/` (checkpoints/ is git-ignored).

## 1. Labeller set-up on other days (rule A3; renders = exact labels, real = by-eye photo tags on 297 val / nw photos)
| class | chosen phrasing @ thr | render IoU (n renders) | real recall (tagged yes) | real false (tagged no) | the comparison's primary on the same data |
|---|---|---|---|---|---|
| rail | "metal pole" @ 0.3 | 0.753 (36) | 0.90 (58) | 0.07 (239) | "railing" @ 0.5: 0.069, 0.88 |
| tape | "colored tape" + "colored band" @ 0.3 | 0.942 (20) | 0.76 (37) | 0.02 (260) | "colored tape" @ 0.5 |
| deck incl. bollard | "boat deck" @ 0.3 | 0.080 (9) | 0.95 (38) | 0.01 (259) | bollard phrasings: render IoU 0.03 |
| land | "land" + "mountain" @ 0.5 | 0.968 (13) | 0.98 (111) | 0.12 (186) | "island" @ 0.5: 0.000, 0.21 |
Person "person" @ 0.3 (kept from the comparison's dev). Real tape here = small bands on bow stanchions in crew photos;
the culled days hold no deliberate rail + tape marker. Deck is weak on renders for every phrasing.

## 2. Pseudo-labels (training days only; 2 500 scanned, rule B4)
Label map: whale = S6h (P >= 0.1 halo and SAM 3 whale outside S6h ignored), extra classes from SAM 3, any-boat (pontoon,
hull, other vessel) and gear ignored, unsure SAM 3 ignored, inside S6h a SAM 3 class wins only at score >= 0.5 (B5).
Training photos holding each class (pixel-truth classes, of 2 200 pseudo-labelled photos, version 2):

| whale | person | rail | tape | deck incl. bollard | land |
|---|---|---|---|---|---|
| 2 157 | 248 | 230 | 57 (61 k px in all; median 465 px per photo at 1280) | 64 | 705 |
Plus S6's 1 151 real partial whale labels (SAM 3 regions and non-water blobs ignored) and the 6 440 run05 renders.
No synthetic rail / tape / bollard render was used for training.

By eye on 60 random labels (first build, sam3lab_pseudolabels_firstbuild.jpg): 56 without a visible wrong region; a
whole fluke painted person (1); gear poles painted rail (3: a jab pole, two out-of-focus poles, a strap). Land right on all
16 tiles that show land; whale outlines S6h's, with awash parts ignored. After rule B5: maps with person over >= 30 % of
the S6h whale 25 -> 11 of 2 200 (the fluke is now ignore). Poles as rail stay (no rule separates them).

## 3. Results

### 3a. Whale on the val days (883 catalogue photos; S6h = S6 last.pt + hysteresis 0.6 / 0.1, same 1280 grid)
| model | photos with no whale pixel (S6h: 7) | pooled IoU with S6h | median per-photo IoU | photos under 0.8 | S6h pixels ours calls water |
|---|---|---|---|---|---|
| v2 s0 (primary) | 6 (all 6 also empty in S6h) | 0.955 | 0.957 | 4.9 % | 1.5 % |
| v1 s0 | 3 | 0.957 | 0.957 | 5.2 % | 1.6 % |
| v1 s1 | 5 | 0.951 | 0.956 | 5.6 % | 1.5 % |
Synthetic run05 val whale IoU: 0.926 / 0.928 / 0.927 (S6 0.931). Tape on val (rule B6): v1 never predicts tape (0 photos),
v2 on all 17 photos where the labeller paints tape (20 photos in all).

### 3b. Test day pixel truth (116 photos, freeze 2; scored once; IoU / recall / precision)
| class (truth photos) | ours v2 s0 | ours v1 s0 | ours v1 s1 | labeller (S6h + SAM 3 chosen) | SAM 3 primary (comparison) | M2F-ADE | M2F-COCO |
|---|---|---|---|---|---|---|---|
| person (49) | 0.879 / 0.98 / 0.90 | 0.841 | 0.862 | 0.885 | 0.91 | 0.880 | 0.851 |
| rail (36) | 0.857 / 0.97 / 0.88 | 0.668 | 0.782 | 0.862 | - | - | - |
| rail + tape (36) | 0.839 / 0.96 / 0.87 | 0.660 | 0.769 | 0.861 | 0.74 ("railing") | 0.282 (as railing etc.) | 0.000 |
| tape (36) | 0.346 / 0.57 / 0.47 | 0.000 | 0.000 | 0.806 | 0.43 ("colored tape") | no class | no class |
| deck incl. cleat (23) | 0.475 / 0.71 / 0.59 | 0.234 | 0.201 | 0.098 | 0.07 | in "boat" | in "boat" |
| our boat 3-6 (43) | 0.800 / 0.84 / 0.94 | 0.460 | 0.426 | 0.351 | - | 0.848 (boat + vessel) | 0.283 |
| land (29) | 0.854 / 0.93 / 0.91 | 0.786 | 0.851 | 0.857 | 0.005 ("island") | 0.618 | 0.691 |
| water / sky (116) | 0.976 | 0.959 | 0.947 | 0.951 | - | 0.973 | 0.933 |
| whale, all 116 | 0.385 / 0.89 / 0.40 | 0.256 | 0.278 | 0.717 | 0.861 | - | - |
| whale, 72 whale photos | 0.733 | 0.764 | 0.761 | 0.717 | | S6h 0.720 | |
| no-whale photos painted whale (of 44) | 26 | 30 | 20 | 0 | | S6h 0 | |
SAM 3's post-hoc phrasings on this day (not a test): rail + tape 0.88 ("metal pole"), tape 0.85 ("colored band"), land 0.57.
Where v2's whale pixels fall: 40 % truth whale, 46 % truth deck, 10 % water, 3 % rail / tape. Truth tape -> v2: tape 57 %,
whale 33 %, rail 5 %. Truth deck -> v2: deck 71 %, whale 21 %.

### 3c. Markers on the test day (rule C3 fixed beforehand; gate = our own whale < 50 px at 1280)
| model | rail + tape (13) | bollard (7) | land (3) | false markers (404) | val days false (884) | tape colour right (13) |
|---|---|---|---|---|---|---|
| v2 s0 (primary) | 1 | 0 | 3 | 0 | 1 (bollard) | 10 (green read blue; one pink "other"; one "none") |
| v1 s0 / v1 s1 | 0 / 1 (wrong kind) | 0 / 0 | 3 / 3 | 0 / 0 | 0 / 1 | 0 / 0 |
| labeller (S6h gate) | 13 | 3 | 2 | 0 | 0 | 10 |
| post hoc, NOT a test: v2 classes with S6h as the whale gate | 12 | 7 | 3 | 0 | 1 (bollard) | as above |
Why v2 misses with its own gate: its whale class paints a few thousand pixels on 11 of the 13 rail markers (they pass the
rail / tube / tape sizes) and 130-260 k px of the bow deck on all 7 bollard shots (deck class itself 61-77 % of the frame).
The only rail marker missed by the sizes is 027 (tape band 0.03 %, half hidden). False colour calls on other photos: 30
(stanchion bands in crew photos; the marker gate would not reach them).

### 3d. Cost (100 val photos, bf16, long side 1280, incl. softmax / hysteresis; RTX 4000 Ada, shared)
S6h 0.055 s / photo, 727 MB peak; ours 0.072 s / photo, 753 MB peak; both 88.2 M parameters (one 7-channel output layer more).

## 4. Reading (judged)
- Our model learns person, rail, land and our boat from SAM 3's labels on 2 200 photos and matches or beats SAM 3 and
  Mask2Former on them (rail 0.86, land 0.85, person 0.88, our boat 0.80), at S6h's speed. Deck (0.48) ends up better than
  its own labeller (0.10). Tape is weak (0.35): 57 training photos with tiny stanchion bands, none a marker close-up;
  version 1 never learned it, version 2 only after class-balanced sampling.
- The whale holds on whale photos and on the culled val days, but NOT on unculled no-whale photos: it paints the bow deck
  and parts of rail markers as whale (26 of 44 no-whale photos). The training labels left the boat mostly as ignore
  (SAM 3 "boat" region, weak "boat deck"), and the culled training days hold almost no deck to teach "deck is not whale";
  S6h never paints it. Until fixed, keep S6h as the whale and use the scene model only for the other classes; with that
  split (post hoc) the marker rule finds 22 of 23 markers with the right kind and 0 false.

## 5. What this route needs to be trusted
- A fresh unculled day with pixel truth or at least marker truth, phrasings and rules unchanged.
- Labels for the boat itself: deck and bitt on real photos (here 64 training photos, labeller render IoU 0.08), and the
  boat region as "not whale" rather than ignore. A few hundred unculled photos (crew, bow, markers) would give this; the
  culled catalogue days barely contain them.
- Tape and bollard: real examples are almost only the culled marker photos (training photos: tape 57 with tiny bands,
  bollard / deck 64, rail markers 0). Synthetic rail / tape / bollard renders (exact labels, r06r_boat20*) are still the
  cheapest source for exactly those classes, or the markers of further unculled days labelled by SAM 3.
- Two or more seeds of the primary recipe (only v2 seed 0 exists; v1 seeds differ by 0.11 rail IoU on the test day).

Not verified: the test day is the only unculled day; one seed of v2; by-eye tags and the 60-label look are one person's.

# Scene model v3 and the marker reader (2026-11-22/23)

Rules: `rules_scene_v3.md` (part S 2026-11-22 20:42 before any v3 label, training or dev score; S6 addendum 21:05;
record R 2026-11-23 ~03:30 before the test-day run). Hashes: `moonshot_cache/sam3_labeller/v3/rules.sha256`. Freeze:
`moonshot_cache/sam3_labeller/v3/FREEZE.json` (checkpoint, code, gate; one amendment: S6 P is always cached, see below).
Code: `v3_tools.py` (ADE labels, v3 label maps, render sets, predictions, dev scores), `v3_markers_eval.py` (gate choice),
`v3_sheets.py`, `v3_testday.py` (the single test-day scoring), `model_fine_tuning/seg_multi/{model.py (NOT_WHALE loss),
data.py / train.py (render source), markers.py (reader), run_new_day.sh (one command for a new raw day)}`.

## Commands
    V3=/home/sat3737/NOAA_whales/moonshot_cache/sam3_labeller/v3; LOCK=/var/tmp/sat3737-parts-cache/gpu_train.lock; V=.venv/bin/python
    flock $LOCK env PYTHONPATH=. $V experiments/sam3_labeller/v3_tools.py ade                  # Mask2Former ADE20K on 2 347 photos
    $V experiments/sam3_labeller/v3_tools.py labels                                           # -> V3/pseudo_v3, V3/real_s6mc_v3 (dev photos held out)
    $V experiments/sam3_labeller/v3_tools.py renders r06r_boat20b_look r06r_boat20_look r06r_boat22_markers   # train 291 / hold 62
    flock $LOCK env PYTHONPATH=. $V -m model_fine_tuning.seg_multi.train --name seg_scene_v3b_s0 --seed 0 --balanced \
        --pseudo-dir $V3/pseudo_v3 --real-dir $V3/real_s6mc_v3 --render-dir $V3/renders/train --p-render 0.1   # v3a: no --render-dir
    flock $LOCK env PYTHONPATH=. $V experiments/sam3_labeller/v3_tools.py predict --ckpt CKPT --name NAME --set dev,renders,val,train5
    flock $LOCK env PYTHONPATH=. $V experiments/sam3_labeller/v3_tools.py s6renders
    $V experiments/sam3_labeller/v3_tools.py devscore --names v2s0,NAME; $V experiments/sam3_labeller/score_val.py --names ../v3/pred/NAME --out ...
    $V experiments/sam3_labeller/v3_markers_eval.py --name NAME                               # culled-day false markers, render markers, gate
    model_fine_tuning/seg_multi/run_new_day.sh "RAW_DAY_FOLDER" OUT_ROOT                      # decode, S6h, scene v3, records, sheet, removal v2
    $V experiments/sam3_labeller/v3_testday.py --run OUT_ROOT --name v3b_s0 --extra ... --records-extra ...
Renders: config copy of the live run06 config + round-20h boat keys + force_plan (rail red 38 / green 38 / blue 38 /
yellow 36, deck 90, land 60), seed 2202, 700 shots generated and cut to the 300 marker shots
(`Whale-Image-Data-Factory/data/renders/r06r_boat22_markers`, raw frames dropped; live config not written).

## What changed in the labels (v3, rule S2)
- NOT WHALE (code 254, loss -log(1 - P(whale))) where v2 left ignore and a boat labeller is sure (ADE20K boat group,
  SAM 3 "boat" @ 0.5, SAM 3 rail), outside the whale's reach. 340 of 2 176 pseudo photos, 212 of 1 135 real_s6mc photos.
- Deck on real photos from ADE boat / ship touching the frame edge: deck photos 64 -> 80.
- Boat renders (v3b only): exact labels, rail / tape / deck incl. cleat / land; pontoon = not whale; p 0.1 of samples.
- The 31 dev photos are held out of all v3 training.

## Development measures (no test day)
Real dev photos (31 crew / rail photos on 2025-03-14 val, 2025-01-17 and 2025-03-18 train) and 62 held-out renders
(58 marker shots: rail 28, deck 15, land 15):

| model | real: photos with >= 50 whale px on the boat (px) | renders: marker shots with whale (of 58) | renders: whale px on boat | render IoU rail / tape / deck / land |
|---|---|---|---|---|
| v2 s0 | 6 (2 047) | 46 | 1 324 890 | 0.904 / 0.512 / 0.792 / 0.543 |
| v3a s0 (labels only) | 12 (17 205) | 50 | 892 164 | 0.868 / 0.248 / 0.825 / 0.525 |
| v3b0 s0 (labels + 43 look renders; early, not a candidate) | 5 (775) | 0 | 0 | 0.972 / 0.948 / 0.999 / 0.973 |
| v3b s0 (labels + 291 renders; chosen) | 6 (1 160) | 0 | 0 | 0.973 / 0.951 / 0.999 / 0.973 |
Fault 1 reproduces on the renders (v2 paints whale on 46 of 58 marker shots), not on the culled real crew photos (v2
paints little there: 2 047 px on 3.2 M boat px; most "whale on boat" px there are whale outlines spilling over a rail
in front of a real whale). The label change alone (v3a) does not fix it; the renders do.

Val days (883 catalogue photos, whale against S6h; extra classes against the labeller's val maps):
| model | pooled IoU | photos without whale (S6h 7) | under 0.8 | person / rail / tape / deck / land vs labeller |
|---|---|---|---|---|
| v2 s0 | 0.955 | 6 | 4.9 % | 0.969 / 0.847 / 0.876 / 0.988 / 0.822 |
| v3a s0 | 0.949 | 4 | 5.8 % | 0.969 / 0.906 / 0.876 / 0.977 / 0.792 |
| v3b s0 | 0.951 | 4 | 5.5 % | 0.970 / 0.896 / 0.866 / 0.973 / 0.767 |
Choice (S5): v3b (6 vs 12 real dev photos); it does not beat v2 on the real dev photos (6 = 6); seed 1 trained after.

## Marker reader off the test day (gate choice, S6.1; v3b s0)
Culled days (3 val + 5 train, 3 621 photos): 1 false call for every gate (a crew member at the bow over a large deck,
person 8 %, deck 12 %: "bollard"). Held-out render markers missed / wrong: S6h gate 5 (S6h paints 4 white render decks),
own 1 (a hazy land shot), both 1. Gate = own (tie own / both broken by the script's order, written before output).
Tape colour on the 28 held-out render rails: 26 right.

## The test day 2026-04-07, once (S7, 2026-11-23 ~04:00)
This is the third scored reading of this day for this line of work (v1 / v2, the post-hoc S6h gate, now v3). The faults
fixed here were found on this day, and the marker definition and colour rule come from it. Only a fresh unculled day is
a real test. One runner call stopped in the removal step (no S6 cache with gate own); markers.py then always caches S6 P
(marker logic unchanged; the second call's 427 records are identical to the first's).

Pixel truth (116 photos; IoU / recall / precision; earlier rows in 3b above):
| class (truth photos) | v2 s0 | v3a s0 | **v3b s0 (frozen)** | v3b s1 | SAM 3 primary | M2F-ADE |
|---|---|---|---|---|---|---|
| person (49) | 0.879 | 0.895 | **0.899** / 0.98 / 0.92 | 0.894 | 0.91 | 0.880 |
| rail (36) | 0.857 | 0.869 | **0.851** / 0.98 / 0.87 | 0.769 | - | - |
| rail + tape (36) | 0.839 | 0.845 | **0.850** / 0.98 / 0.86 | 0.767 | 0.74 | 0.282 |
| tape (36) | 0.346 | 0.193 | **0.672** / 0.95 / 0.70 | 0.589 | 0.43 | no class |
| deck incl. cleat (23) | 0.475 | 0.535 | **0.638** / 0.95 / 0.66 | 0.593 | 0.07 | in "boat" |
| our boat 3-6 (43) | 0.800 | 0.852 | **0.923** / 0.97 / 0.95 | 0.933 | - | 0.848 |
| land (29) | 0.854 | 0.861 | **0.864** / 0.93 / 0.92 | 0.875 | 0.005 | 0.618 |
| water / sky (116) | 0.976 | 0.977 | **0.976** | 0.978 | - | 0.973 |
| whale, all 116 | 0.385 | 0.466 | **0.761** / 0.91 / 0.82 | 0.741 | 0.861 | - |
| whale, 72 whale photos | 0.733 | 0.752 | **0.765** | 0.742 | | S6h 0.720 |
| no-whale photos painted whale (of 44) | 26 | 29 | **6** | 7 | | S6h 0 |
v3b s0's 6: three crew photos (85-502 px), two "water" photos (6 k / 24 k px) and one whale_crew photo; the second by-eye
pass (truth_whale_trace.tsv, removal_v2) found whale traces in every first-pass "water" photo, so some of these may be
right (not checked here).

Markers (rule C3 kinds, gate own; truth markers.tsv):
| model | rail + tape (13) | bollard (7) | land (3) | false (404 others) | tape colour right (13) | refers_to exact (markers called, rail + bollard) |
|---|---|---|---|---|---|---|
| **v3b s0 (frozen)** | **13** | **7** | **3** | **1** (389, rail) | **10** | **18 of 20** (mean Jaccard 0.98) |
| v3b s1 | 12 | 7 | 3 | 0 | 10 | 16 of 19 |
| v3a s0 | 1 | 0 | 3 | 0 | 6 | 1 of 1 |
| v2 s0 (rules_labeller C3, own whale) | 1 | 0 | 3 | 0 | 10 | - |
Colour misses (v3b s0): 027 red read "other" (half-hidden pink-red band), 068 red read blue, 078 green read blue (the
same two blue / green confusions as before: the rule is unchanged). The false marker 389: crew at the bow rail over an
awash whale (the scene model paints 48 whale px, just under the 50-px gate; S6h 0); refers_to misses: one extra photo
in each of two series.
Removal v2 with these records in front (runner's last step): 31 proposed (24 marker, 7 crew), 32 faint; one proposal
holds a whale trace (389, the false rail marker; truth_whale_trace: awash). Land marker 050 comes out a land marker.

## Cost and use per day
Runner on 427 RAW: decode ~0.5 min (10 workers), S6 + scene model ~0.15 s / photo, SAM 3 (removal) on the 63 photos
without a whale ~1 s / photo; about 3 min of compute per day plus waits for the shared GPU lock. Output per day: 450 MB
of JPEGs (the day folder) + 8 MB maps + 80 MB SAM 3.

## What stays weak / not verified
- Tape and bollard / deck come from the renders (no real tape close-up or bollard shot in training); the colour rule
  still reads green as blue on dark-water rails; real marker photos from further unculled days would replace the renders.
- Gate own puts the whale decision on the scene model; a crew photo over an awash whale (389) passes as a marker. The
  removal rule then proposes it.
- One seed chosen; seed 1 is weaker on rail (0.77) but has no false marker. The test day was read three times; one person's
  by-eye truth; the renders' look faults (black sea behind rails, clean white deck) remain.
