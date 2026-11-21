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
  "Sure whale" pixels = live seg_r04_ft4 mask AND S6h mask (switch_prep/days_s6), on a 640 x 427 grid.
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

Part B follows.
