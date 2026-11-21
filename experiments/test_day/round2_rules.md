# Test day 2026-04-07, round 2: rules fixed before any round-2 model ran on the day

Written 2026-11-20, before any round-2 output on the test day existed (file time stamp = registration time; a copy with
the same text is in the untracked day folder, round2/rules_round2.md). Nothing below was chosen from this day's outputs.

Declared plainly: the CHOICE of models for this round (candidate mask S6h, Mask2Former, W3c) was made by the main session
after round 1's results on this day were known. That weakens this day as a test of those models; the settings are not
taken from the day, but the selection is.

Known before writing (from round 1 on this day, so not "blind"): the marker photos and their tape colours, that the red
tape reads desaturated pink and the sea saturated blue, that S6 paints no crew / rail. The tape rule in section 4 is
therefore NOT a clean test (see there).

## 1. Candidate mask end to end (spec from experiments/LOG.md, 2026-11-17 18:20 "Candidate now")

Candidate = `experiments/seg_splash_s6/checkpoints/last.pt` + hysteresis 0.6 / 0.1 (components of P >= 0.1 that hold a
pixel with P >= 0.6, on P at long side 1280, then nearest-neighbour to the 2560 frame, exactly as
`seg_real.followup e2e` does) + instances_v2 fluke-join gap 0.35. All other settings live / defaults.

Arms:
- L (live): round 1's day folder as it is (seg_r04_ft4, canon keep_main_components, split_instances, not-whale 2.0, ...).
- S6-src: the live chain with segment() swapped for the candidate mask inside `canon.predict` (keep_main_components 25 %,
  min 200 px, canon crop, CSE, parts, parts_hr --instances, cluster_v3, safe features, link_scores instances,
  not-whale filter threshold 2.0, safe clusters strict 0.995 cost 1.5 skip 3 --filter, view_class, view_match, fin_hw,
  parts_up, cluster_up, link_scores windows / scores / pairs: run_pipeline.sh unchanged), then the Labeling Assistant's
  load_day in a scratch copy of the backend with the same environment as round 1. This arm feeds every downstream table.
- S6-v2 (the candidate exactly as the LOG specifies it): instances_v2 `candidates` with the candidate mask (raw, no 25 %
  rule) and `assemble` with its DEFAULTS except join_gap 0.35 (blob gate blob_gate.pkl, not-whale 2.0), run on the S6-src
  day folder. No downstream consumer of instances_v2 exists, so this arm stops at instances and filter decisions.

"Whale in photo" = at least one kept instance (as round 1). Also reported: at least one kept instance that sits on a
whale by eye (W or F).

By-eye labels for the S6 instances (same letters as round 1's inst_eye.txt: W P R G S F L V): an S6 instance whose box has
IoU >= 0.7 with a round-1 instance labelled W inherits W; every other S6 instance is labelled by eye on the full photo
with its box and mask drawn, before its filter score is looked at.

## 2. Ready-made segmentation with fixed classes

Models (public weights, Hugging Face transformers, Swin-L backbones):
- M2F-COCO: facebook/mask2former-swin-large-coco-panoptic (133 classes)
- M2F-ADE: facebook/mask2former-swin-large-ade-semantic (150 classes)
- M2F-COCO-inst: facebook/mask2former-swin-large-coco-instance, person class only, post_process_instance_segmentation
  default threshold 0.5 (mask_threshold 0.5, overlap 0.8).
Input size: the published test sizes of the Mask2Former configs (the HF preprocessor's 384 x 384 is not the published
test setting): COCO shortest edge 800 (max 1333), ADE20K shortest edge 640 (max 2560). For our 2560 x 1707 frames:
COCO 1200 x 800, ADE 960 x 640. Per-pixel label = arg-max of post_process_semantic_segmentation (sum over queries of
class probability x mask probability), no score threshold; label map resized nearest to 1280 px for scoring.

Class map (their name -> ours):
- person: COCO person; ADE person.
- boat (our boat's railing, deck, structure): COCO fence-merged, railroad, platform, floor-wood, floor-other-merged,
  stairs, bridge; ADE railing, fence, bannister, floor, pole, step, stairs, stairway, column, base.
- vessel (a boat seen as an object): COCO boat; ADE boat, ship.
- land: COCO mountain-merged, rock-merged, sand, dirt-merged, gravel, tree-merged, grass-merged, building-other-merged,
  house, roof, road, pavement-merged, playingfield, snow; ADE mountain, hill, earth, land, rock, sand, tree, palm, plant,
  grass, field, building, house, skyscraper, tower, road, path, dirt track.
- water: COCO sea, water-other, river; ADE sea, water, river, lake, swimming pool, waterfall, fountain.
- sky: COCO sky-other-merged; ADE sky.
- other: every remaining class (kept by its own name in the whale tables).

Photo-level rule (same X for every class): thing present if its pixels are >= X of the frame, X = 0.5 % (primary).
Second, also fixed now: X = 0 (any arg-max pixel). Land as subject: land >= 25 % of the frame; land as background:
land >= 0.5 % and < 25 %. Scored against truth tags:
- person: truth person = 1; predicted person >= X.
- rail or boat parts: truth rail = 1 or deck = 1; predicted boat + vessel >= X (COCO / ADE cannot tell our own bow
  called "boat" from another vessel).
- other vessel: truth vessel = 1; predicted vessel >= X.
- land as subject: truth land_subj = 1; predicted land >= 25 %.
- land as background: truth land_bg = 1; predicted 0.5 % <= land < 25 %.
COCO-inst person: photo has >= 1 person instance at score >= 0.5.

What they call a whale: whale pixels = live mask AND S6h mask on photos whose truth has a whale above the surface
(classes whale, whale + crew). Report the share of those pixels per original class and per our group, and per photo the
group that takes most of its whale pixels; a whale photo counts as "called boat / rock / person" when that group takes
>= 50 % of its whale pixels.

Veto test: veto pixels = person + boat + vessel of one model (each model on its own, and the union of both).
- On no-whale photos (rail marker, bow cleat, crew no whale, land, water) every live-mask pixel is false: report the share
  of live-mask pixels the veto removes, per truth class.
- On whale + crew photos: false paint = live-mask pixels outside the S6h mask (round 1's sheet: S6 does not paint crew);
  true whale = live AND S6h. Report the share of each removed.
- On whale photos: share of true whale (live AND S6h) removed, and the photos losing >= 50 % of it.

## 3. Part candidate W3c

Checkpoint: /var/tmp/sat3737-parts-cache/parts_weak/runs/parts_seg_r05_w3c/checkpoints/best.pt (seed 0; seed 1 not
used). Applied as parts_seg/predict.py applies it: a full-frame map at long side 1280, and per instance an "hr" window
(1.5 x the box's long side, at least 192 px, output <= 768 px) around each instance box; class counts inside the instance
mask grown by one cse-crop pixel + 2 map px ("pixels_near"). Classes: fluke = 1, dorsal = 2 (fin + hump), other = pectoral
3 + head 4 + body 5; background 0.
- live gate: hr windows of the live (round 1) instances, counted inside the live instance masks.
- candidate gate: hr windows of the S6-src instances, counted inside the S6-src instance masks.
- no gate: the full-frame map, every pixel.
Reported per truth photo class (share of pixels fluke / dorsal / other / background) and on the pixels the ready-made
models call person / boat / land / water / sky (M2F-ADE groups; COCO as a second column). Per-instance call with the rule
of LOG 2026-11-17 (fluke if fluke share >= 0.03 and >= pectoral share, else dorsal if fin share >= 0.0005, else other).

## 4. Tape colour (first principles; NOT clean, see top)

Rail region = M2F-ADE pixels labelled railing, fence, bannister or pole, plus M2F-COCO fence-merged, on the 2560 frame
(label maps resized nearest). HSV (OpenCV: H 0-180, S and V scaled to 0-1):
- white tube pixel: S < 0.2 and V >= 0.55.
- coloured pixel: S >= 0.25 and V >= 0.15.
- a coloured pixel counts as tape when it is inside the rail region and within 15 px (2560 frame) of a white tube pixel of
  the rail region.
- hue bins: red H < 10 or H >= 160; yellow 20 <= H < 35; green 35 <= H < 85; blue 85 <= H < 130; anything else "other".
- photo colour = the bin with the most tape pixels if that count is >= 300 px and >= 2 x the next bin, else "none".
Scored once on all 427 photos: the 13 rail markers' colours and the colour calls on every other photo. No change after.

## Addendum, 2026-11-20 14:48 (before W3c ran on the day; the readymade maps existed, no part output did)
Section 3's call rule is the LOG 2026-11-17 rule of parts_seg_r05_a. W3c's own rule, chosen on training days by
evaluate_real (experiments/parts_weak/parts_seg_r05_w3c_real_metrics.json, variant hr): fluke if fluke share >= 0.005,
else dorsal if fin share >= 0.02, else other (shares of the instance's covered pixels incl. predicted background). Both
rules are reported; the W3c rule is the primary one because it is the candidate's own.
