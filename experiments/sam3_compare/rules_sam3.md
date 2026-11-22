# SAM 3 against our whale masks: rules

Part A written 2026-11-21 00:05 -0600, before any SAM 3 output on any of our photos was looked at. Part B (the test-day
settings) is added after the dev run and before the one test-day run; its time and sha256 go into
`moonshot_cache/sam3_compare/rules_sam3.sha256`. The test day 2026-04-07 is a TEST day: nothing is fitted, tuned or
chosen on it. Eval days 2024-12-18, 2024-12-26, 2025-02-19, 2025-03-21 are never read.

## A1. Model and settings that are not chosen
- `facebook/sam3` (local HF cache), transformers 5.20.1 `Sam3Model` / `Sam3Processor` in the Underwater-WhaleID venv
  (used read-only), image mode (no video / tracker), float16, one pass per photo on the day folder's 2560 x 1709 JPEG.
  The processor resizes every photo to 1008 x 1008 (squashed, aspect not kept); masks come out at 288 x 288 over the
  whole photo and are upsampled bilinearly to the photo, binarised at 0.5 (the post-processing default).
- Instance score = sigmoid(class logit) x sigmoid(presence logit), as `post_process_instance_segmentation` computes.
  Raw outputs are stored for score >= 0.1, so thresholds are applied later.
- Every phrasing in `prompts_all.json` is run on every photo of a set (click days: the whale and part phrasings only).
- Overlapping instances of one phrasing: the union is the class mask; for instance scoring each pixel goes to the
  highest-scoring instance that covers it.

## A2. Dev data (other days only)
- Val days 2025-01-15, 2025-02-27, 2025-03-14 (884 photos, 883 catalogue photos; every catalogue photo holds a whale).
  "Sure whale" pixels = live seg_r04_ft4 mask AND S6h mask (the shadow root moonshot_cache/switch_root/days, S6h + rule F; switch_prep/days_s6 was moved there), on a 640 x 427 grid.
- Non-whale instances: the positives of `experiments/stage2_linker/nw_labels.json` (live parts_hr instance masks), on the
  val day 2025-03-14 (34) and on training days (the 128 photos of `lists/nw.tsv`); the 150 training-day positives with
  a subclass by eye in `test_days/2026-04-07/nw_subclass_train.json` (person / gear / boat / other).

## A3. Whale phrasing and threshold (chosen on dev by this rule)
Candidates: phrasings "whale", "humpback whale", "marine mammal" x score thresholds 0.3 (transformers default), 0.4,
0.5 (the model card's example).
- R = catalogue photos on the val days with >= 1 instance / 883.
- F = non-whale instances (all nw_labels positives with a mask) with >= 25 % of their mask painted by the phrasing's
  union / their number.
- J = R - F. The highest J wins; candidates within 0.005 of the best are tied, ties go to the higher threshold, then to
  the order "whale", "humpback whale", "marine mammal".
Reported for all nine: R, F, sure-whale pixel recall, SAM 3 whale pixels outside live OR S6h (pooled share).

## A4. Extra classes
- Person, gear, rail (dev = nw subclasses person / gear / boat): for each phrasing (gear: also the union of its four)
  and threshold 0.3 / 0.4 / 0.5: Rc = subclass instances with >= 25 % of their mask covered / their number;
  Fc = val photos where the phrasing covers >= 25 % of the photo's sure-whale pixels / val photos with >= 500 sure-whale
  pixels. J = Rc - Fc, same tie rule (rail: "boat" subclass, n small; reported as such).
- Tape, deck / cleat, pontoon, other vessel, land: no labelled dev data on other days (catalogue photos are culled:
  no markers, few close-ups of our boat). Primary = the first phrasing of the group in prompts_all.json at 0.5 (the
  model card's example threshold). The others are run and reported as sensitivity only.
- Dev check for those classes (descriptive, not a choice): pooled share of sure-whale pixels on val days each phrasing
  paints at 0.5, and val photos where it paints >= 1 % of the frame.


# Part B, written 2026-11-21 03:48 -0600 after the dev run (dev.json) and before the one test-day run

## B1. Choices made by the part A rules (dev = val days + nw_labels instances; numbers in the README)
- Whale: J = R - F over 9 candidates: best "humpback whale" @ 0.3 (J 0.9649). Tied within 0.005: "humpback whale"
  @ 0.4 / 0.5, "whale" @ 0.3, "marine mammal" @ 0.4 / 0.5. Ties to the higher threshold, then the order:
  **"humpback whale" @ 0.5** (the model card's example threshold).
- Person: **"person" @ 0.3** (the transformers default; J 0.930, next 0.922). Gear: **union of "drone", "camera",
  "pole", "crossbow" @ 0.3** (J 0.892). Rail (dev = 9 "boat" crops, weak): "railing" and "metal pole" tie at 0.778 at
  every threshold -> **"railing" @ 0.5**.
- No dev labels, first phrasing @ 0.5 (model card): tape "colored tape", deck "boat deck", pontoon "inflatable boat
  tube", other vessel "boat", land "island"; bollard = union of "cleat" and "bollard" @ 0.5. Parts @ 0.5.
- The exact settings file: `settings_partB.json` (same text in moonshot_cache/sam3_compare/).
- Every other phrasing is run on the test day too and reported as sensitivity, never used to choose.
- Change of input made before this part: S6h on the val / click days is read from the shadow root
  moonshot_cache/switch_root/days (S6h + rule F), because switch_prep/days_s6 had been moved there.

## B2. Scoring on the test day (fixed before its run)
- Photo level, whale (all 427, round 1's truth.tsv): SAM 3 "says whale" = >= 1 instance of the chosen whale phrasing
  at the chosen threshold. Also reported: an instance with >= 30 % of its pixels on live OR S6h (proxy for "on the
  whale"; the rest looked at by eye). Our arms: round 2's chain numbers (>= 1 kept instance) quoted, and raw mask
  "any pixel" recomputed here from the same files.
- Pixel level (frozen pixel truth, 116 photos, truth grid 2748 x 1835, 255 left out): pooled IoU / recall / precision
  for whale = class 1; SAM 3's whale mask = union of its whale instances. Confusion row = share of each arm's whale
  pixels on each truth class. Extra classes: each class's SAM 3 mask (union of the chosen phrasing's instances at its
  threshold) against its truth class, plus the grouped pairs (rail + tape, our boat 3-6, any boat 3-7) where they
  compare with Mask2Former's pairs. Every phrasing of every group is scored too (sensitivity, not a choice).
- Whale + crew fusion (the 35 whale + crew photos, all in the pixel sample): an instance is fused when >= 10 % of its
  scored pixels and >= 200 px are truth whale AND >= 10 % and >= 200 px are truth person. SAM 3 instances
  de-overlapped by score; our masks as 8-connected components (our real instances are parts_hr's; their by-eye fused
  counts from round 1 / 2 are quoted).
- Photo level, extra classes: class pixels >= 0.5 % of the frame (round 2's rule); land as subject >= 25 %; against
  the truth.tsv tags person, rail, deck, gear, vessel, land_subj, land_bg (value exactly "1").

## B3. Marker rule (fixed from first principles; val days give the false-call check only, they hold no markers)
Two gates, both reported: (S) SAM 3 alone: no whale instance; (O) our S6h mask beside SAM 3: S6h paints 0 px.
The user's definition (relayed 2026-11-21 00:15, before any test-day output): a rail marker is a deliberate close-up,
the rail fills the frame, the tape is prominent, no whale; rail or tape small in the corner of a busy photo is a correct
class detection, NOT a marker. Facts from the truth given with it: tape 0.2-1.0 % of the frame on the 13 markers,
0.01-0.06 % on bow stanchions in crew photos; tube width at the tape 5-11 % of the frame width.
On a gated photo, in this order:
1. land: land class >= 25 % of the frame -> land marker;
2. bollard (bow-cleat close-up, the user's "bollard shot": marks an opportunistic fluke of another animal): the union
   of the phrasings "cleat" and "bollard" (added 00:28 from the user's word, before any SAM 3 run on any day; "mooring
   bollard" run as sensitivity) at score >= 0.5 covers >= 3 % of the frame -> bollard marker;
3. rail: tape class >= 0.1 % of the frame AND rail + tape classes >= 3 % of the frame -> rail marker
   (variant reported beside it, not primary: rail + tape >= 5 % without the tape condition, for when the tape is missed);
4. person class >= 1 % -> crew; else water.
Person is last on purpose (a hand or a leg in a marker photo must not hide the marker).
Class detection of rail and tape anywhere (photo level, pixel level) is reported separately from markers.
Scored on the 13 rail markers, 7 bow-cleat shots, 3 land markers (found = any marker call; right kind = the call
matches), false markers on the other 404 photos, and the same rule on the 884 val photos.

## B4. Tape colour rule (fixed before the run)
On the full-resolution decoded photo: tape mask = union of the tape class's instances at its threshold, upsampled,
eroded by max(1, round(0.002 x width)) px. Coloured pixels = HSV saturation >= 0.25 and value >= 0.20. Fewer than
200 coloured pixels -> "none". Otherwise the circular mean hue of the coloured pixels: red < 20 deg or >= 320,
yellow 20-70, green 70-170, blue 170-260, other 260-320. Right = the first word of tape_colours.tsv. Calls on the 414
other photos are false colour calls.

## B5. Hard kinds (selection from round 1's truth only; counts by eye on the sheet)
From truth.tsv, whale / whale + crew / under-water photos: thin backs awash = note matches awash|wash|thin;
splash or blow = tag blow or splash = 1; just under the surface = class under; far small = tag far = 1; white fluke
underside = note matches underside; two whales = n_whales 2 or 2?. Measured per kind: photos with a SAM 3 instance,
with an S6h pixel, median IoU SAM 3 vs S6h. Sheet: 6 photos per kind drawn with random.Random(20261121).sample(sorted
(kind), 6); per tile by eye: SAM 3 better / same / worse than S6h on the whale.

## B6. Sheets
sam3_whale_differs: the 12 whale-class photos with the lowest IoU(SAM 3, S6h) among photos where either paints
>= 0.5 % of the frame, the 6 no-whale or under-water photos with the largest SAM 3 whale share, and the 6 whale
photos with SAM 3 instances and no S6h pixel or the reverse (largest painted share first).

## B7. Pixel truth corrections
The pixel truth is being corrected (rails in the crew photos re-drawn along the tube, the whale of one photo). Pixel
scores are made against the freeze that exists at run time (its time is written beside the numbers) and re-run once for
rail, tape and whale after the re-freeze. Photo- and instance-level parts do not use the pixel truth.
