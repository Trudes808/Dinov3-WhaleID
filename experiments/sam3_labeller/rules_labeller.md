# SAM 3 as an offline labeller for our multi-class segmentation: rules

Part A written 2026-11-21 05:36 -0600, before any SAM 3 output was opened in this experiment (the comparison's raw SAM 3
outputs on the val days and the nw photos exist from `experiments/sam3_compare/`; I have read its README and tables,
which already report test-day numbers for some phrasings, see A0). sha256 of each part in
`moonshot_cache/sam3_labeller/rules.sha256`. Test day 2026-04-07: nothing fitted, chosen or looked at before the one
scored run (part C). Eval days 2024-12-18, 2024-12-26, 2025-02-19, 2025-03-21: never read. Val days 2025-01-15,
2025-02-27, 2025-03-14: dev for the labeller (A2) and the whale check (C1), never in training.

## A0. What traces back to the test day already (stated up front)
- The idea and the classes (person, rail, tape, deck incl. bollard, land) come from the test day's markers.
- The comparison's README reports test-day IoU for every phrasing of `prompts_all.json` ("metal pole" 0.85-0.88 for
  rail + tape, "colored band" 0.85 for tape, "land" / "mountain" 0.57, "island" 0.005). I know these numbers. They are
  allowed only as candidates; a phrasing is used only if it wins by rule A3 on non-test data.
- The candidate list is exactly the 31 phrasings of `experiments/sam3_compare/prompts_all.json` (written before any
  SAM 3 output on any day). No new phrasing is added (in particular no colour words like "pink", which I would only
  add because of what the test day showed).
- The marker definition (rail >= 4 %, tube >= 5.5 %, tape band >= 0.12 %) was proposed from the test day's truth
  (pixel_truth/tables.md); it is used as written there, not re-tuned.

## A1. SAM 3 as run
As `experiments/sam3_compare/run_sam3.py` (reused unchanged): `facebook/sam3`, transformers 5.20.1 in the
Underwater-WhaleID venv (read-only), fp16, image mode, the day folder's 2560-px JPEG (renders: their PNG), squashed to
1008 x 1008 by the processor, masks at 288 x 288 upsampled bilinearly, mask >= 0.5. Instance score = class prob x presence
prob; instances with score >= 0.1 are stored. A phrasing's class mask at threshold t = union of its instances with
score >= t. A union of phrasings = union of their class masks.

## A2. Dev data (no test day)
1. Boat renders (exact labels, read-only): `r06r_boat20b_look` (28) and `r06r_boat20_look` (25), label layer
   `dataset/scene/<id>.png` (palette: 3 rail, 4 tape, 5 deck, 6 cleat, 7 pontoon, 8 land; 1 / 2 whale). SAM 3 is run
   on them with all 31 phrasings. Pixel score per class pooled over the renders (on the label grid, 2560 x 1707).
2. Real photos, non-test days, with SAM 3 raw outputs already made by the comparison: the val days (lists/val.tsv,
   884 photos) and the nw photos (lists/nw.tsv, 128 training-day photos with by-eye non-whale instances).
   By-eye photo-level tags, made on plain thumbnails (no model overlay), per class: person, rail (a boat rail tube or
   stanchion in frame), tape (a coloured band on a rail), deck (our boat's deck or bow bollard), land (any land),
   land_big (land >= about a quarter of the frame). Which photos get tagged: for each class, the photos where the union
   of that class's candidate phrasings at 0.3 paints >= 0.05 % of the frame (pool P_c, flagged blind to which phrasing
   painted), at most 80 drawn with random.Random(20261121) if larger, plus 40 random photos outside P_c (misses).
   Person is not re-tagged: its choice from the comparison (dev on the nw subclasses: "person" @ 0.3) is kept.

## A3. Choice per class (phrasing or union of the group's phrasings x threshold 0.3 / 0.4 / 0.5)
Candidates per class: each single phrasing of the group (prompts_all.json groups; deck = deck + bollard groups merged:
"boat deck", "cleat", "boat bow", "bollard", "mooring bollard"), and the union of every pair within the group.
Render class sets: rail -> {3}, tape -> {4}, deck -> {5, 6}, land -> {8}.
- IoU_r = pooled pixel IoU on the renders that hold the class (>= 0.1 % of frame); FP_r = share of the renders without
  the class where the candidate paints >= 0.5 % of the frame (tape: >= 0.05 %).
- Real: area rule a_c = 0.05 % of the frame (tape), 0.5 % (rail, deck, land). Rc = tagged-yes photos (A2) where the
  candidate paints >= a_c / tagged-yes photos; Fc = tagged-no photos where it paints >= a_c / tagged-no photos.
  When fewer than 5 tagged-yes photos exist for a class, Rc is not used (reported only) and J uses the renders alone.
- J = IoU_r - FP_r + (Rc - Fc), maximised. Ties within 0.02: higher threshold, then single phrasing over union, then
  the order in prompts_all.json.
- Land: the class is "land"; the land marker uses the land class share (rule C3).
- Person "person" @ 0.3 (kept). Whale for the labeller is NOT SAM 3: it is S6h (A5). SAM 3's "humpback whale" @ 0.5
  (the comparison's dev choice) is used only to mark ignore regions.
- Also fixed now (not chosen): any-boat region = "boat" @ 0.5 (pontoon / hull / other vessel go to ignore through it),
  gear region = union of "drone", "camera", "pole", "crossbow" @ 0.3 (the comparison's dev choice).
- Unsure SAM 3 region of a class = instances of its chosen phrasings with 0.15 <= score < the chosen threshold.
Part B adds the chosen set and its numbers with a new time stamp before the training-day scan.

## A4. Which training-day photos get pseudo-labels
SAM 3 is run with the chosen phrasings (and the fixed ones in A3) on every photo of the 21 training days
(model_fine_tuning/splits/days.json "train", 8 060 photos). A photo is "extra" when any extra class (person, rail, tape,
deck, land) paints >= 0.1 % of the frame at its chosen threshold, or our live mask (seg_r04_ft4, canon mask) paints
>= 0.5 % of the frame outside S6h (crew / boat the live mask is known to paint). Pseudo-label set = every extra photo
(capped at 1 500 by random.Random(20261121).sample if larger) + random plain photos to a total of 2 200
(random.Random(20261122)). Val days are labelled the same way for checks only (never trained on).

## A5. Label map per photo (long side 1280, as S6's real crops; classes ours: 0 water / sky, 1 whale, 2 person,
3 rail, 4 tape, 5 deck incl. bollard / cleat, 6 land, 255 ignore)
S6 = experiments/seg_splash_s6/checkpoints/last.pt, P = its whale probability at 1280; S6h = hysteresis 0.6 / 0.1
(as everywhere). In this order (later steps overwrite earlier ones):
1. everything 0;
2. ignore: P >= 0.1 outside S6h; SAM 3 "humpback whale" @ 0.5 outside S6h (submerged bodies, backs S6h misses);
   any-boat region; gear region; every extra class's unsure region;
3. land (sure) -> 6;
4. S6h -> 1, then a 2-px band either side of the S6h edge -> ignore;
5. deck -> 5, rail -> 3, tape -> 4, person -> 2 (sure SAM 3 classes win over whale and land; person last, so a hand
   on a rail is person);
6. a 3-px band either side of every boundary of a SAM 3 class (steps 3 and 5) -> ignore (SAM 3's mask grid is about
   4.4 px of the 1280 frame).
Pontoon, other vessel and gear are not classes: ignore (no usable labeller: pontoon phrasings painted nothing in the
comparison, other-vessel phrasings paint our own bow, gear phrasings paint rails; and they are few at this data size).

## A6. Training (ours)
Recipe of seg_splash_s6 (model_fine_tuning/seg_real/train.py: ViT-B/16 DINOv3, last 4 blocks, crop 512, scale 0.5-1.5,
photometric 1.0, AdamW lr head 3e-4 / backbone 3e-5, layer decay 0.8, wd 0.05, warm-up 200 then cosine, batch 8, bf16,
8 epochs of len(run05 train), init experiments/seg_r05_ft4/checkpoints/best.pt) with a 7-channel output (whale channel
initialised from the binary head, extra-class biases -3), loss = cross-entropy with ignore + mean soft Dice over the
classes present in the batch. Each sample: run05 synthetic crop (labels 0 / 1 / ignore as S6) with p 0.5; S6's real
partial-label crop (real_s6, by-eye whale blobs) with p 0.2, where every SAM 3 region of A5 step 2-5 other than
whale (any extra class, sure or unsure, any-boat, gear) is set to ignore and every non-water background blob of
blob_labels (shore, boat, person, pole, drone, bird, other) is ignore; pseudo-label crop with p 0.3, centred on an
extra-class region with p 0.6 when the photo has one, else on a whale or a random point. Seeds 0 and 1 if time allows.
The checkpoint kept = last (as S6h uses last.pt), saved as best.pt only.

## A7. Prediction
Long side 1280, softmax. Whale = hysteresis 0.6 / 0.1 on P(whale) (as S6h). Other pixels = argmax over the non-whale
classes (water / sky, person, rail, tape, deck, land). Output in the truth's classes: 0 -> 0, 1 -> 1, 2 -> 2, 3 -> 3,
4 -> 4, 5 -> 5, 6 -> 8.

## C. Scoring (written with part A, 2026-11-21 05:36, before any training)
C1. Whale, val days (883 catalogue photos, labels_all_days.json): our whale (A7) against S6h computed the same way
    (S6 last.pt at 1280, hysteresis 0.6 / 0.1): catalogue photos with no whale pixel (each arm); pooled IoU of the two
    masks at 1280; per-photo IoU median and the share of photos under 0.8; S6h pixels our model gives to each other class.
C2. Test day pixel truth (116 photos, freeze 2), ONCE: `pixtruth_score.py --root R --ours NAME=DIR` for each seed and for
    the labeller itself (A5's map on the test photos: S6h whale + SAM 3 chosen phrasings, ignore -> 255), output to
    moonshot_cache/sam3_labeller/pixtruth_scores (not the shared scores folder). Beside them the existing SAM 3
    (primary and post-hoc) and Mask2Former rows from the comparison's tables.
C3. Markers on the test day (427 photos), ONCE, with ours (A7 at 1280, shares of the frame):
    gate: no whale = fewer than 50 whale px at 1280 (about 0.005 % of the frame);
    1. rail marker = (proposed definition, pixel_truth/tables.md) rail + tape >= 4 % of the frame, widest tube
       = 2 x the largest distance-transform value inside rail + tape >= 5.5 % of the frame width, largest 8-connected
       tape component >= 0.12 % of the frame;
    2. else land marker = land >= 25 % of the frame;
    3. else bollard marker = deck >= 10 % of the frame and person <= 10 % (a deck / bitt close-up, not crew at the bow);
    4. else none. Photos passing rule 1 with a whale are counted and reported, not markers.
    Scored against markers.tsv: found per kind (any marker call) and right kind; false markers on the other 404; the same
    rule on the 884 val photos (false calls only).
    Tape colour: rule B4 of sam3_compare (unchanged: HSV on the 2560-px photo; S >= 0.25, V >= 0.20; < 200 px -> none;
    circular mean hue: red < 20 or >= 320, yellow 20-70, green 70-170, blue 170-260, other 260-320) applied inside the
    largest tape component of our map, upsampled nearest to 2560 and eroded by max(1, round(0.002 x 2560)) = 5 px.
    Right = the first word of tape_colours.tsv; reported on the 13 rail markers and as false colour calls elsewhere.
C4. Cost: seconds per photo (GPU forward incl. softmax / hysteresis at 1280, no file IO) and peak GPU memory, on 100
    val photos, ours against S6, same script, bf16.

# Part B, written 2026-11-21 07:13 -0600 after the dev choice (dev_choose.py) and before the training-day scan

## B1. Dev data as used
- Renders: 53 (rail in 36, tape in 20, deck / cleat in 9, land in 13; the rest count for FP_r).
- By-eye tags: 297 real photos (val days + nw photos; pools rail 111 -> 80, tape 33, deck 47, land 265 -> 80, + 40 outside
  each), plain 384 x 256 thumbnails, then 640-px sheets of every rail-tagged or tape-pool photo for the tape tag. Tagged
  yes: person 79, rail 58, tape 37, deck 38, land 111, land_big 28. Tape here = the small bands on the bow stanchions in
  crew photos; there is no deliberate rail + tape close-up (marker) among them (catalogue photos are culled).
- Deviation: for a pair of phrasings the real "paints >= a_c" test is an OR of the two phrasings' shares (the stored
  per-photo shares are per phrasing); the union can only be larger, so this undercounts a pair's Rc and Fc slightly.

## B2. Chosen (J = IoU_r - FP_r + Rc - Fc; dev/choice_table.md has every row)
| class | chosen | thr | render IoU | render FP | real Rc (n yes) | real Fc (n no) | J | primary of sam3_compare, same J table |
|---|---|---|---|---|---|---|---|---|
| rail | "metal pole" | 0.3 | 0.753 | 0.00 | 0.90 (58) | 0.07 (239) | 1.579 | "railing" @ 0.5: IoU 0.069, Rc 0.88, J 0.944 |
| tape | "colored tape" + "colored band" | 0.3 | 0.942 | 0.00 | 0.76 (37) | 0.02 (260) | 1.680 | "colored tape" @ 0.5 |
| deck incl. bollard | "boat deck" | 0.3 | 0.080 | 0.00 | 0.95 (38) | 0.01 (259) | 1.019 | bollard phrasings: render IoU 0.03, Rc 0.03-0.29 |
| land | "land" + "mountain" | 0.5 | 0.968 | 0.03 | 0.98 (111) | 0.12 (186) | 1.807 | "island" @ 0.5: IoU 0.000, Rc 0.21, J 0.207 |
Person "person" @ 0.3, SAM 3 whale "humpback whale" @ 0.5 (ignore only), any-boat "boat" @ 0.5, gear union @ 0.3 (part A).
Exact settings: moonshot_cache/sam3_labeller/labeller.json; scan phrasings: scan_prompts.json (13).

## B3. What traces back to the test day
The three winners for rail, tape and land are the phrasings the comparison found best AFTER seeing the test day
("metal pole", "colored band", "land" / "mountain"). Here they won on renders and on other days' photos by a rule
written before (A3), from the fixed candidate list of prompts_all.json. They were in that list before any test-day
output, but my attention to them was not blind. Deck ("boat deck" @ 0.3) is weak on the renders (IoU 0.08: it paints
little of the bow structure); it wins on the real crew photos' photo-level test only.

## B4. Change to A4 made now (GPU contention: the shared lock gave one hold in about 80 minutes)
SAM 3 scans 2 500 training-day photos instead of all 8 060: every photo where the live mask paints >= 0.5 % outside
S6h (649), every photo of the three days missing from the shadow root (2025-04-01 / 04-09 / 04-11, 538), every photo of
S6's real labels (real_s6, 1 151, needed for A6), every train-day photo of nw_labels positives; together 2 016; plus 484
drawn with random.Random(20261123). The pseudo-label set is then chosen from these by A4 (extra photos, cap 1 500, plain
to 2 200). Land photos are therefore found among the scanned photos only.

## B5. Change after the 60-label look (written 2026-11-21 09:34 -0600, before any training; test day untouched)
By eye on the 60 random training labels (sheet sam3lab_pseudolabels, first build): a whole fluke painted "person" (1 of
60), gear poles painted rail (3 of 60: a jab pole, two out-of-focus poles, a strap); 25 of the 2 200 maps had person over
>= 30 % of the S6h whale. Change to A5 step 5: inside S6h a SAM 3 class wins only where its instance score is >= 0.5;
inside S6h with score below 0.5 the pixel is ignore; outside S6h unchanged. Poles painted rail are left as they are
(the gear phrasings paint the rail too, so no rule separates them at this size); reported as a labeller error.

## B6. Version 2, after the val check of version 1 (written 2026-11-21 11:47 -0600; test day still untouched)
Version 1 (seeds 0 and 1 trained) never predicts tape on the val days (0 photos; the labeller paints tape on 17), though
it learns person, rail, deck and land there. Cause in the labels: tape totals 43 k pixels over 57 of the 2 200 training
photos (median 274 px per photo at 1280; the 3-px ignore band erases much of a stanchion band), and pseudo crops are
centred on a random extra pixel, so a tape crop comes up about 10 times per epoch. Version 2 (seed 0 only; time):
- labels pseudo_v2: the boundary ignore band is 1 px for rail and tape (3 px for the others), else as B5;
- pseudo crops: with p 0.6 an extra class is drawn uniformly among the five, then a photo holding it, then one of its
  stored points (tape in about 3 of 10 pseudo crops instead of under 1 in 100).
Primary model for the test-day run, fixed now from the val days: version 2 seed 0, if on the val days it predicts tape
on at least 5 of the 17 photos where the labeller paints tape and keeps the whale (pooled IoU with S6h >= 0.93);
otherwise version 1 seed 0. All three trained models are scored on the test day and reported; only the primary's
numbers answer the question.

# Part D (record, 2026-11-22 20:29 -0600)
Primary by rule B6 (val days): version 2 seed 0 (tape on 17 of 17 val photos where the labeller paints tape; pooled IoU
with S6h 0.955). The single test-day run (pixel truth, markers) was made 2026-11-22 20:2x for v2 s0, v1 s0, v1 s1 and the
labeller; nothing was changed after it. One post-hoc variant (marker rule with S6h as the whale gate) is reported as such.
