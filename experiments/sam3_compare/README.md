# SAM 3 against our whale masks (live seg_r04_ft4, candidate S6h), 2026-11-21

A small comparison asked for by the user: is SAM 3 (text-prompted concept segmentation, weights on this machine from the
underwater thread) better or worse than our trained DINOv3 whale mask, and can it find the marker photos (rail + tape,
bow bollard, land) and read the tape colour. Text only here. Masks, per-photo rows and sheets are in the untracked
`moonshot_cache/sam3_compare/` (sheets also in a session scratch folder). Photos are named by their 3-digit sequence
number on the test day.

Rules: `rules_sam3.md` (part A 00:01 before any SAM 3 output; part B 03:48 after the dev run, before the one test-day
run; hashes in `moonshot_cache/sam3_compare/rules_sam3.sha256`). Settings used on the test day: `settings_partB.json`.

**Caveats.** The test day 2026-04-07 has by now shaped several earlier choices (which models were tried in rounds 1-2,
the S6h candidate, the marker rules), so it is a weaker test than a fresh unculled day. The pixel truth was drawn with
a point / box prompted SAM 2.1 and corrected by eye: its outlines are SAM-family outlines, which favours SAM 3 at the
pixel level (section 1b measures how much). Photo-level and instance-level numbers do not have that problem.
Pixel scores use the truth freeze of 2026-11-21 01:06 (freeze 2b, rails in crew photos re-drawn along the tube).

## Commands

    S=/home/sat3737/NOAA_whales/moonshot_cache/sam3_compare; PY3=/home/sat3737/NOAA_whales/Underwater-WhaleID/.venv/bin/python   # used read-only
    flock /var/tmp/sat3737-parts-cache/gpu_train.lock $PY3 experiments/sam3_compare/run_sam3.py --list $S/lists/SET.tsv \
        --prompts experiments/sam3_compare/prompts_all.json --set SET --out $S/raw          # SET = val, nw, test; click uses prompts_click.json
    PYTHONPATH=. .venv/bin/python experiments/sam3_compare/dev_select.py --out $S/dev.json                 # part A choice data
    PYTHONPATH=. .venv/bin/python experiments/sam3_compare/val_whale.py --prompt whale --thr 0.5 --out $S/val_whale_whale05.json
    PYTHONPATH=. .venv/bin/python experiments/sam3_compare/val_markers.py --settings experiments/sam3_compare/settings_partB.json
    PYTHONPATH=. .venv/bin/python experiments/sam3_compare/analyze_clicks.py --settings experiments/sam3_compare/settings_partB.json --out $S/clicks.json
    PYTHONPATH=. .venv/bin/python experiments/sam3_compare/analyze_test.py --settings experiments/sam3_compare/settings_partB.json --out $S/test
    PYTHONPATH=. .venv/bin/python experiments/sam3_compare/whale_band.py --settings experiments/sam3_compare/settings_partB.json --out $S/test/whale_band.json
    PYTHONPATH=. .venv/bin/python experiments/sam3_compare/hard_kinds.py --settings experiments/sam3_compare/settings_partB.json --out $S/test_hard_kinds.json
    PYTHONPATH=. .venv/bin/python experiments/sam3_compare/sheets.py differs|extra|markers|hard|val_only --settings ... --out SHEET.jpg --seqs ...
    flock ... env PYTHONPATH=. .venv/bin/python experiments/sam3_compare/cost_ours.py --list $S/lists/val.tsv --n 100 --out $S/cost_ours.json

SAM 3 setup: transformers 5.20.1 `Sam3Model` / `Sam3Processor`, image mode, fp16, one image encoding per photo then one
decoder pass per phrasing. The processor squashes every photo to 1008 x 1008 (aspect not kept); masks come out at
288 x 288 for the whole photo and are upsampled. Score = class prob x presence prob.

## Choices made on other days (dev; rules A3 / A4)

Whale, val days (883 catalogue photos) + 184 by-eye non-whale instances (nw_labels):

| phrasing @ threshold | catalogue photos with an instance | non-whale instances painted >= 25 % | sure-whale pixel recall | SAM 3 pixels outside live OR S6h |
|---|---|---|---|---|
| whale @ 0.3 / 0.5 | 874 / 871 | 5 / 5 of 184 | 0.963 / 0.962 | 6.2 % / 5.1 % |
| **humpback whale @ 0.3 / 0.5** | 876 / **874** | 5 / **5** | 0.956 / **0.955** | 5.4 % / **4.4 %** |
| marine mammal @ 0.3 / 0.5 | 873 / 872 | 6 / 5 | 0.952 / 0.952 | 5.1 % / 4.2 % |

Chosen: "humpback whale" @ 0.5 (= the model card's example threshold). The wording barely matters for the whale on
these days (871-876 of 883). Person: "person" @ 0.3 (finds 90 of 96 person crops, paints >= 25 % of the whale in 6 of
832 val photos). Gear: union of drone / camera / pole / crossbow @ 0.3 (33 of 37 crops). Rail: "railing" @ 0.5 (7 of
9 rail / deck crops; weak). No dev labels for tape, deck, pontoon, other vessel, land, bollard: first phrasing @ 0.5,
written down before the run.

## 1. Whale mask

### 1a. Test day, photo level (all 427; SAM 3 = >= 1 "humpback whale" instance; ours: raw mask >= 1 px, and in brackets
the chain's ">= 1 kept instance" from round 2)

| truth | n | live | S6h | SAM 3 |
|---|---|---|---|---|
| whale above surface | 368 | 365 (340) | 360 (358) | 357 |
| - whale | 333 | 330 (328) | 326 (324) | 323 |
| - whale + crew | 35 | 35 (12) | 34 (34) | 34 |
| whale only under water | 17 | 17 (7) | 4 (3) | 7 |
| no whale, given a whale | 42 | 36 (3) | 0 (0) | 2 |
| - rail 13 / cleat 7 / crew 7 / land 3 | | 13 / 7 / 7 / 2 (0) | 0 | 0 |
| - water / splash 12 | | 7 (3) | 0 | 2 (175, 346) |

- The 2 water photos SAM 3 gives a whale (175, 346): a small dark blob inside a splash line; by eye unclear.
- SAM 3 leaves 11 whale photos empty: 118, 233, 251, 261, 262, 277, 278, 314, 416, 422 and 350. S6h leaves 8 empty
  (118, 246, 260, 261, 262, 277, 350, 416). Shared: 118, 261, 262, 277, 350, 416.
- By eye (sam3_whale_differs.jpg): 251, 278, 314 are LARGE close backs filling the frame, painted by S6h, missed whole by
  SAM 3. 246 and 260 (thin back in wash, head under spray) are painted by SAM 3, not by S6h.
- SAM 3 paints whole bodies seen through the water (260, 302, 323, 380, 404 ...). Our masks and the truth keep to what is
  above water or awash. So SAM 3's "whale" is the whole animal, ours is the ID-useful surface.

### 1b. Test day, pixel level (pixel truth, 116 photos, 72 with whale; freeze 2026-11-21 01:06)

| | IoU | recall | precision | its whale pixels on: whale / water / person / rail / deck / unsure |
|---|---|---|---|---|
| live | 0.093 | 0.841 | 0.094 | 0.09 / 0.05 / 0.25 / 0.16 / 0.35 / 0.03 |
| S6h | 0.720 | 0.822 | 0.853 | 0.81 / 0.13 / 0.006 / 0 / 0 / 0.05 |
| SAM 3 | **0.861** | **0.957** | **0.896** | 0.65 / 0.08 / 0 / 0 / 0 / 0.28 |

Outline bias (truth drawn with SAM 2.1). Same scores with a band of r px around every truth boundary left out
(half-resolution grid): r = 0 / 2 / 5 / 10 / 20 px -> S6h 0.720 / 0.746 / 0.771 / 0.794 / 0.815; SAM 3 0.861 / 0.883 /
0.891 / 0.891 / 0.880. So about half of the gap (0.14 -> 0.065) is outline placement within 20 px; the rest is whole
regions (S6h misses 251-type close backs and parts of awash backs; SAM 3's 28 % on "unsure" is the submerged body).
Phrasing spread on the same pixels (sensitivity, not a choice): "whale" 0.847, "humpback whale" 0.861, "marine mammal" 0.872.

### 1c. Val days, photo level (883 catalogue photos)
- Catalogue photos with no SAM 3 whale instance ("whale" @ 0.5): 12. Ours: raw mask empty live 1, S6h 7; no kept
  instance after the chain live 28, S6h 9. By eye the 12: a far tiny whale, a breach inside spray, 5 whales under the
  surface beside crew at the bow, 5 awash / under-surface backs.
- SAM 3 instances off our masks (< 30 % on live OR S6h): 94 of 1 130. By eye on the 32 largest: mostly real whale parts
  our masks miss (second far whales, a dorsal fin behind a fluke, white pectoral fins); a few are wave shapes. Not
  counted one by one.
- On the 34 by-eye non-whale instances of 2025-03-14 (crew, pole, crossbow) SAM 3 paints >= 25 % of 3 (all 184: 5).

### 1d. Instances
Click days (2025-01-02, 2025-03-18; training days of ours, scored as switch_eval.py scores; SAM 3 instances de-overlapped):

| | flukes, both tips in one instance (n 83) | main whales, all clicks in one instance (n 365) | other whales own / fused / none (n 17) | clicked photos without an instance |
|---|---|---|---|---|
| live, parts_hr after filter | 47 | 310 | 4 / 5 / 8 | 2 |
| S6h, parts_hr after filter | 51 | 316 | 4 / 5 / 8 | 1 |
| live / S6h, instances_v2 (LOG 11-17) | 65 / 66 | 334 / 337 | 13 / 4 / 0, 12 / 4 / 1 | - |
| **SAM 3** | **70** | **340** | **16 / 1 / 0** | 1 |

Test day, whale + crew (35 photos, all with pixel truth): SAM 3 whale pixels on person 0; instances fused whale + person
(>= 10 % and >= 200 px each) 0 photos. Connected components of S6h: 1 photo, live: 8 (round 1's by-eye count on live
instances: 25 fused, 20 filtered away).

### 1e. Hard kinds (selection from round 1's truth; measured; by eye on 6 per kind, sam3_hard_cases.jpg)

| kind | n | SAM 3 any | S6h any | median IoU SAM 3 vs S6h | by eye, 6 tiles: SAM 3 better / same / worse |
|---|---|---|---|---|---|
| thin back awash | 30 | 23 | 25 | 0.62 | 1 / 3 / 2 |
| splash or blow | 87 | 75 | 78 | 0.76 | 2 / 2 / 1 (+1 shared) |
| just under the surface | 17 | 7 | 4 | 0.00 | 2 / 3 / 1 |
| far small | 43 | 40 | 42 | 0.82 | 0 / 5 / 1 |
| white fluke underside | 9 | 9 | 9 | 0.94 | 0 / 6 / 0 |
| two whales | 16 | 16 | 16 | 0.83 | 1 / 4 / 1 |

## 2. Extra classes and markers

### 2a. Pixel level (pixel truth; SAM 3 primary phrasing; Mask2Former from pixel_truth/scores, scored 01:04)

| class (truth photos) | SAM 3 IoU / recall / precision | M2F-ADE | M2F-COCO | best other SAM 3 phrasing (seen on the test day: not a test) |
|---|---|---|---|---|
| person (49) | 0.91 / 0.97 / 0.93 | 0.88 / 0.96 / 0.92 | 0.85 / 0.95 / 0.89 | "people" 0.91 |
| rail + tape (36) | 0.74 / 0.77 / 0.96 ("railing") | 0.28 (as "boat") | 0.00 | "metal pole" 0.88 |
| tape (36) | 0.43 / 0.43 / 0.97 ("colored tape") | no class | no class | "colored band" 0.85 |
| deck incl. cleat (23) | 0.07 / 0.07 / 0.65 | in "boat" | in "boat" | bollard phrasings 0.17 / 0.17 / 0.96 |
| pontoon (12) | 0 (no paint) | in "boat" | in "boat" | none works |
| any boat (52) | 0.65 / 0.66 / 0.98 ("boat") | 0.86 | 0.30 | |
| other vessel (9) | 0.03 (paints our bow too) | 0.02 | 0.05 | |
| land (29) | 0.005 ("island") | 0.62 | 0.69 | "land" 0.57, "mountain" 0.57 |
| gear (56) | 0.06 / 0.54 / 0.06 (the "pole" phrasing paints rails) | no class | no class | "drone" 0.15 |

### 2b. Photo level (427; class pixels >= 0.5 % of the frame; land subject >= 25 %)

| tag (truth n) | SAM 3 recall / precision | M2F-ADE | M2F-COCO |
|---|---|---|---|
| person (58) | 0.95 / 0.81 | 0.95 / 0.93 | 0.93 / 0.84 |
| rail (44) | 0.86 / 1.00 | rail or boat 0.96 / 0.68 | 0.63 / 0.78 |
| deck (28) | 0.50 / 0.93 | | |
| gear (74) | 0.51 / 0.75 | | |
| other vessel (9) | 1.00 / 0.23 | 1.00 / 0.16 | 1.00 / 0.23 |
| land subject (3) | 0 / - ("island"); post-hoc "land" + "mountain" 1.00 / 0.60 | 1.00 / 0.21 | 1.00 / 0.43 |
| land background (44) | 0.09 / 1.00; post-hoc 0.93 / 0.84 | 0.77 / 0.11 | 0.73 / 0.65 |

Person precision: the 9 photos of passengers on another boat are "vessel" in the truth (as for Mask2Former).

### 2c. Markers (rule B3, fixed before the run; deliberate subject, no whale)

| rule | rail + tape (13) | bollard (7) | land (3) | false markers on the other 404 | val days (884) |
|---|---|---|---|---|---|
| primary phrasings, gate SAM 3 | 3 | 5 | 0 | 1 (390) | 0 |
| primary, gate S6h | 3 | 5 | 0 | 1 | 0 |
| variant rail + tape >= 5 % without tape | 9 | 5 | 0 | 1 | |
| post-hoc phrasings ("metal pole", "colored band", "land" + "mountain"), not a test | 12 (13 variant) | 5 | 3 | 3 (382, 389, 390: crew at the bow over a whale under water) | |
| Mask2Former ADE rule (LOG 11-20 23:53) | 8 flagged as role marker (+1 as group change) | 7 | 3 | 0 | 0 |

- Every marker that was found was the right kind: SAM 3 tells a bollard close-up from a rail close-up (0 confusions),
  which the ADE rule could not. The bollard phrasing misses 135 and 136 (bollard partly in shadow, a rod across it).
- Why the primary rule misses rails: "railing" misses the single white tube on blue sea (007, 028, 070, 078) and
  "colored tape" misses every pink / red band (011, 027, 068, 072, 179, 186). With no tape the rail falls to "crew"
  (a hand or leg) or "water". The post-hoc phrasings cover both, but were picked after seeing this day.
- Tape colour (rule B4): primary 6 of 13 right (every blue, the yellow), 6 "none" (all the red ones), green read as blue;
  6 false colour calls on other photos. Post-hoc "colored band": 11 of 13 right (the 6 reds right; 027 none; green ->
  blue), 10 colour calls on non-markers (bow stanchions in crew photos) which the marker gate would not reach.
  Round 2's first-principles rule on Mask2Former: 3 of 13.

## 3. Parts (click days, "dorsal fin" @ 0.5, flukes @ 0.5)
- Dorsal fin (197 clicked triangles): no fin instance on 42; triangle covered median 0.61 (>= 0.5 on 118, >= 0.8 on 55);
  fin tip inside a fin instance 155; the best fin instance paints a median 15 % outside the triangle. For reference
  W2 (our part model, LOG 11-17): fin tip inside the largest fin region 91.6 %, tip painted mostly fin 100 %.
- Flukes (83): "whale fluke" all of tips + notch in one instance 44 (no instance 1); "whale tail" 41. "whale fluke" is
  not a part: on the val days it covers 85 % of all sure-whale pixels (backs too). Not a usable part model as is.

## 4. Cost and practicalities (measured on this machine, RTX 4000 Ada 20 GB, shared)

| | input the model sees | GPU s / photo | peak GPU memory |
|---|---|---|---|
| SAM 3, 1 phrasing (~0.84 B params, 3.3 GB fp32 file) | 1008 x 1008 squashed; masks 288 x 288 | ~0.21 (0.18 vision + 0.03) | 2.0 GB (fp16) |
| SAM 3, 31 phrasings | same | 1.1-1.3 (wall 1.3-1.9 with decode + saving) | 2.0 GB |
| live / S6h (88 M params) | long side 1280 (1280 x 854), prob map at 1280 upsampled | 0.059 / 0.074 (incl. upsampling, hysteresis) | 0.72 GB (bf16) |

- Far small whales: SAM 3's 288-px mask grid is ~9 x 6 photo pixels per mask pixel at 2560 px; thin awash backs and fin
  edges are blocky. It still found 40 of 43 "far" photos (S6h 42). Tiling (e.g. 2 x 2 crops) would cost 4x.
- Licence (model card, "SAM License", 2025-11-19): non-exclusive, royalty-free; use, copy, modify, make derivative works
  and redistribute under the same licence; publications must acknowledge SAM; no ITAR / military / weapons uses; gated
  download. Fine-tuned derivatives are ours to own, distributed under the SAM licence.
- New class: SAM 3 = a phrase (minutes), but wording decides it (tape 0.43 vs 0.85, land 0.005 vs 0.57, pontoon 0).
  Ours = renders + labels + training (days), but then it is exactly the class we mean.
- Wording stability: whale stable (871-876 of 883 val photos; test IoU 0.85-0.87); small or odd classes unstable.
- Fine-tuning here: transformers has Sam3Model with training-capable code and the weights fit in 20 GB for inference; a
  full fine-tune of the ~0.85 B model would need most of the GPU (not tried). Using it as an offline labeller needs none.

## 5. Reading (judged)
- SAM 3 is better at: whole-whale outlines (pixel IoU 0.86 vs 0.72, about half of that is outline style), keeping whales
  and crew apart (0 fused), separate instances (flukes 70 vs 51-66, main whales 340 vs 316-337, other whales own 16 of 17),
  whales under the surface, naming person / rail / tape / bollard without training.
- Ours is better at: large close backs that fill the frame (3 photos SAM 3 misses whole), photo-level "keep a whale"
  (S6h 360 vs 357 of 368), not painting submerged bodies, fine outlines at full resolution, a fixed meaning with no
  wording risk, speed (about 3x faster per photo than SAM 3 with one phrasing, a third of the memory), and dorsal fin parts.
- Tie: flukes, far whales, two-whale photos, no-whale photos (S6h 0, SAM 3 2 of 42 water photos).

## 6. Mixed options (judged; costs from section 4)
1. SAM 3 beside our whale mask, only to name what fills the photos our mask leaves empty (markers, crew, land, water):
   runs on ~10 % of an unculled day, ~0.2 s + 0.03 s per phrasing per photo, 2 GB. Needs its phrasings chosen on another
   unculled day first (the primary ones found 8 of 23 markers here; the best ones, picked after the fact, 20 of 23 with
   3 false). Lowest cost, nothing downstream changes.
2. SAM 3 as an offline labeller for our model on real photos (speed stays ours): one pass ~0.2 s / photo. Needs a rule
   for what to keep: SAM 3 paints submerged bodies and misses some large close backs, so labels would need its masks
   gated by ours (agree -> label, disagree -> by-eye or ignore). Then a retrain and the usual val-day check (days).
   Most promising for instance separation (flukes, neighbouring whales, crew) where SAM 3 is clearly ahead.
3. SAM 3 as the whale mask: ~3x slower per photo, misses 3 large close backs here, paints submerged bodies, and every
   later stage (CSE use map, instances, clusters, view class, ranking) was fitted on our masks and would need re-checking.

Recommendation from the numbers: keep S6h as the whale mask; add SAM 3 as in option 1 for the empty photos and markers
after a phrasing check on another day; test option 2 for instances. What would change it: on a fresh unculled day,
phrasings fixed in advance finding >= 20 of 23-type markers with <= 1 false call (option 1 confirmed); SAM 3 instances
lowering clusters per animal on the val days with 0 wrong links (option 2 / 3); SAM 3 missing no large close backs on
more data and the pixel gap holding with a truth not drawn by a SAM-family tool (option 3).
