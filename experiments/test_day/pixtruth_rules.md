# Test day 2026-04-07: pixel truth for the extra classes, and how every model is scored on it

Registered 2026-11-20 18:20, before any model output was opened for the sampled photos in this work and before any truth
was drawn. The untracked copy `moonshot_cache/test_days/2026-04-07/pixel_truth/rules_pixtruth.md` has the same text and
its sha256 is in the manifest. The day is a TEST day: nothing is fitted, tuned or chosen on it.

Known before writing (from round 1 / round 2 tables, so not blind): which photos hold markers, crew, vessels, the tape
colours, and the photo-level numbers of every model. No pixel output of any model was looked at for this work.

## 1. Sample (fixed by rule, list in the manifest)

From `truth.tsv` (round 1, unchanged):
- all 42 no-whale photos: class in {marker_rail, bow, crew_nowhale, marker_land, water};
- all 35 class whale_crew photos;
- all photos with vessel = 1 (9);
- 30 further photos of class whale, not already in the sample, with land_bg, rail or gear equal to "1", drawn with
  `random.Random(20261120).sample(sorted(seqs), 30)`.
A tag counts only when its value is exactly "1" ("?" does not count).

## 2. Truth classes (one uint8 label image per photo at half the full resolution, 2748 x 1835)

0 water / sky (also splash, blow, wake, footprint, haze); 1 whale (above water or awash and clearly whale; body visible
through clear water just under the surface is 255); 2 person (incl. clothing, life vests, hats, held items that cover
the body; blurred foreground heads are person); 3 rail (the tubes of our boat's rail and stanchions); 4 tape (the
coloured band on a rail tube); 5 deck and bow structure incl. the cleat, fittings, ropes lying on deck; 6 pontoon / hull
of our boat; 7 other vessel (its hull, structure and its passengers); 8 land (islands, shore, mountains, haze-blue land
included when its outline is visible); 9 gear (drone, pole camera, pole, crossbow, rod, held cameras);
255 unsure / do not score (motion-blurred edges, things I cannot name, out-of-focus areas where the class is not clear).

How it is drawn: my own boxes and points (positive / negative) are fed to a promptable mask tool (SAM 2.1, public
weights, no other model). Every photo's mask is checked by my eye on an overlay and corrected (more points, polygons,
paint to 255). No Mask2Former, OWLv2, CLIP, our whale masks, part maps or any other model output is used to make or
check the truth. The truth is my eye with a mask tool, not the researcher's. The tape colour of every rail marker is
written to `tape_colours.tsv` by eye.

## 3. Freeze

Before any model's label map for the sample is opened: label PNGs, `tape_colours.tsv` and the per-photo notes are
hashed (sha256) and written with a time stamp to `manifest.json`. After that no truth pixel changes. If an error in the
truth is found later it is listed in the tables, not fixed.

## 4. Scoring (all on the truth grid, 2748 x 1835; model maps resized there by nearest neighbour; 255 left out)

Per class: pooled over the sample, IoU = sum TP / sum (TP + FP + FN), recall = TP / (TP + FN), precision = TP / (TP + FP),
n photos = photos where the truth class has >= 1 pixel; also photos where the model has >= 1 pixel of it. Per photo the
same three in the per-photo tsv. Secondary (also fixed now): the same with a 2-pixel band around every truth class
boundary left out ("band" columns).

Mask2Former ADE20K and COCO panoptic (round-2 label maps, the class map of `round2_rules.md` section 2: person, boat,
vessel, land, water, sky, other). Pairs scored (truth -> model):
- person: 2 -> person;
- our boat, strict: 3 + 4 + 5 + 6 -> boat;
- our boat: 3 + 4 + 5 + 6 -> boat + vessel (they call our bow "boat" / "vessel");
- other vessel: 7 -> vessel (they cannot tell it from our boat; reported, not a fair test);
- any boat: 3 + 4 + 5 + 6 + 7 -> boat + vessel (primary "boat" number);
- land: 8 -> land;
- water / sky: 0 -> water + sky;
- rail alone: 3 + 4 -> boat (ADE railing / fence / pole live there).
No whale or gear class exists in either: those rows are reported in the confusion matrix only. Passengers on another boat
are 7 in the truth (round 1's convention), so ADE / COCO "person" there counts against person precision; this is also
reported with class 7 left out.

Whale masks (live seg_r04_ft4 mask of the day folder; candidate S6h mask of `cand_s6/`): binary, whale = 1 -> mask.
Confusion row: the share of the mask's pixels on each truth class 0-9 and 255.

Part candidate W3c (if round 2 stored a full-frame map): "any part" = classes 1-5 of the no-gate full-frame map,
scored as whale = 1 the same way.

Confusion matrix per model: rows = truth classes 0-9 (255 shown as its own row, not in the scores), columns = the model's
classes (groups above; whale masks: whale / not), in pixels and row-normalised.

Tape (nothing segments it): from the truth only, per marker photo: tape pixels (half-res grid and full-res equivalent x 4),
share of the frame, and band height in tube widths = per connected tape component, extent along the local tube axis
divided by the extent across it (axis = principal direction of the rail + tape pixels within 40 px of the component).

## 5. Sheets

24 photos spread over the classes (chosen from the truth alone, before scoring: at least 2 per class present in the
sample), photo | truth | each model's map in the same false colours. One sheet with the worst photo per class and model
(lowest per-photo IoU among photos where the truth class has >= 500 pixels).
