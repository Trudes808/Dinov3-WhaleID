# Experiment log

Data: Whale-Image-Data-Factory `run01` (2000 renders, 1280x720), split
`model_fine_tuning/splits/run01.json`: 1400 train / 300 val / 300 test, stratified by pose
category, seed 0. Backbone: DINOv3 ViT-B/16 (LVD-1689M weights, `~/dinov3/weights`).
GPU: RTX 4000 Ada 20 GB; bf16 autocast; AdamW, cosine schedule, 200 warm-up steps.
All numbers below are on the **test** split (the val split selected the checkpoint).

## 2026-10-11 — segmentation

Training: random 512-px crops at 0.5-1.5x native scale, 70 % biased to contain whale
pixels, horizontal flips, photometric augmentation (exposure/white balance/saturation/blur/
noise/JPEG/grey). Loss BCE + Dice. Validation on full 1280x720 frames. 30 epochs, batch 8.
Head LR 3e-4, backbone LR 3e-5 with layer decay 0.8, weight decay 0.05.

| run | backbone trained | best val IoU (epoch) | test dataset IoU | mean image IoU | boundary F1 | negatives clean | time |
|---|---|---|---|---|---|---|---|
| seg_vitb16_frozen | none (head only, 2.5 M params) | 0.9145 (19) | 0.9043 | 0.848 | 0.858 | 0.922 | 52 min* |
| seg_vitb16_ft4 | last 4 blocks + norm (30.9 M) | 0.9130 (13) | 0.9035 | 0.844 | 0.849 | 0.941 | 61 min* |
| seg_vitb16_ftall | all 12 blocks + patch embed (88 M), backbone LR 2e-5 | 0.9160 (22) | **0.9068** | 0.852 | 0.862 | 0.961 | 49 min* |

\* three runs shared the GPU; alone, each is about half that.

Per category (dataset IoU, frozen / ft4): arch 0.92 / 0.92, blow 0.96 / 0.96, breach 0.92 / 0.90,
fluke_up 0.94 / 0.93, logging 0.91 / 0.90, pec_up 0.85 / 0.85, spyhop 0.92 / 0.93,
submerged 0.77 / 0.75, **awash 0.57 / 0.61**. By visible size: tiny (< 500 px) 0.12 / 0.12,
small (< 3 k) 0.80 / 0.80, medium (< 20 k) 0.83 / 0.83, large 0.93 / 0.93. The full fine-tune
(ftall) lands within 0.01 of these everywhere (awash 0.61, submerged 0.75, tiny 0.10, large 0.93).

Reading: on the synthetic test split all three are tied on IoU (0.904 / 0.904 / 0.907; the
full fine-tune's +0.003 is within the spread between checkpoints). What training the backbone
does buy is cleaner negatives: empty frames with no false positives go 92 % -> 94 % -> 96 %, and
the frozen head's false-positive pixels on empty rough-sea frames (mean 1400 px) vanish once
any blocks are trained (~50 px), i.e. the frozen features confuse whitecaps with whale and the
fine-tune fixes that. All three saturate on the same cases: whales barely breaking the surface (a few labelled pixels),
and bodies clearly visible *through* the water that the strict "above the water" label calls
background (these count as false positives; see `eval_test/overlays/worst_*.jpg`). That is a
label-definition question for the factory (an "ignore" or "submerged-visible" class), not a
model-capacity one. The fine-tuned backbone may still matter on real photos, which is what
`test_data_do_not_use` is for; keep both checkpoints and compare there.

## 2026-10-11 — body coordinates + pose (body_vitb16_ft4)

Training: whale-centred square crops (GT mask box, margin 1.25-2.2x, 12 % centre jitter,
min 160 px source side) resized to 512; 1112 train frames with >= 300 visible pixels; flips
mirror the angle and the rotation; same photometric augmentation. Last 4 blocks trained.
Losses: visibility BCE (1), L1 on u (10), 1 - cos on the angle (5), Frobenius on R (2),
L1 on log pixels-per-metre (1). 40 epochs, batch 8, 61 min shared. Best epoch 33.

Test (232 frames with >= 300 visible px), oracle crops (GT box) vs end-to-end crops from
seg_vitb16_ft4's mask:

| metric | oracle crops | seg crops |
|---|---|---|
| u MAE (fraction of body length) | 0.0207 (0.33 m; median 0.013) | 0.0211 (0.34 m) |
| angle around body MAE | 6.5° (median 3.5°) | 6.7° |
| visibility IoU (in crop) | 0.863 | 0.863 |
| rotation error, direct head: median / < 15° / < 30° | 11.1° / 68 % / 91 % | 11.4° / 68 % / 90 % |
| rotation error, PnP on dense coords: median / < 15° / < 30° | 11.4° / 61 % / 82 % | 10.8° / 63 % / 80 % |
| log-scale error (pixels per metre), direct / PnP, median | 0.063 / 0.078 | 0.065 / 0.076 |
| landmark PCK@0.1 body length (PnP pose, visible landmarks) | 0.823 | 0.822 |
| landmark reprojection error, median (fraction of body length) | 0.034 | 0.031 |

Floor for PnP on *ground-truth* coordinates (table + RANSAC): median 1.2°, p90 4°.

Frozen-backbone comparison (body_vitb16_frozen, head only, 40 epochs, 34 min, best epoch
val score 0.880 vs 0.915), test with seg crops: u MAE 0.029 (0.47 m), angle MAE 10.2°,
rotation median 14.8° direct / 18.0° PnP, < 15°: 52 % / 45 %, PCK@0.1 0.75. Unlike
segmentation, fine-tuning the last blocks clearly helps the dense-correspondence task
(angle error -35 %, rotation < 15° +16 points).

By visible size (seg crops): 2-10 k px: u 0.029, angle 7.3°, rotation median 13.6° (direct)
/ 16.7° (PnP); 10-50 k px: u 0.018, angle 6.3°, 10.8° / 8.4°. PnP beats the direct head
when the whale is large and well segmented; the direct head is more robust when it is small.
Failures are rolled / inverted whales in spray (pec slaps, breaches) where the dorsal /
ventral side is confused and the whole rotation flips (~8 % of frames > 60° off); the two
estimates disagree strongly there, which is a usable confidence signal
(`rot_direct_vs_pnp_deg` in `predict.py` output).

## Follow-ups

- Factory: an ignore band / "visible through water" class at the waterline; the awash and
  submerged categories are where labels and appearance disagree.
- Real photos: run `test_scripts/bash/predict_test_data.sh` on `test_data_do_not_use`; the
  agreement between the direct rotation and the PnP rotation is the first thing to look at.
- Which segmentation checkpoint to carry forward: on run01 they are interchangeable, so the
  choice should be made on real photos (frozen is the most conservative, ftall the cleanest on
  empty rough water); the body model was evaluated end-to-end with `seg_vitb16_ft4` crops.

## 2026-10-12 — run02: fixed waterline, size floor, through-water ignore class

Data: `run02` (2000 renders, seed 2, config `poc_boat.json` of 2026-10-11): the Water-plugin
amplitude mismatch fixed (labels now sit on the rendered waterline), left/right no longer
mirrored, body deformation on 85 % of shots and six skins, camera zoomed so the above-water
whale spans >= 25 % of the frame, fewer negatives (8 %), and a new mask class 2 (whale
surface within 1.5 m below the mean water plane, not in the rendered label) that the
loaders turn into IGNORE: not trained on, not scored. Split `splits/run02.json`: whale shots
with < 2000 visible pixels (98) and defect-flagged shots (empty mask, label outside the
projection; 27 more) dropped -> 1313 train / 281 val / 281 test, stratified by category.
Same recipes as run01, trained three at a time (wave 1: ftall, frozen, body ft4; wave 2).

### Segmentation, run02 test split

| run | best val IoU (epoch) | test dataset IoU | mean image IoU | boundary F1 | negatives clean |
|---|---|---|---|---|---|
| seg_r02_frozen | 0.9512 (23) | 0.9583 | 0.919 | 0.882 | 1.000 |
| seg_r02_ft4 | 0.9534 (22) | 0.9606 | 0.924 | 0.893 | 0.960 |
| seg_r02_ftall | 0.9541 (7) | 0.9585 | 0.919 | 0.879 | 1.000 |

Cross-evaluation (dataset IoU / negatives clean), test splits of both runs:

| model | run01 test | run02 test |
|---|---|---|
| seg_vitb16_frozen (run01) | 0.9043 / 0.92 | 0.8588 / 1.00 |
| seg_vitb16_ft4 (run01) | 0.9035 / 0.94 | 0.8689 / 0.96 |
| seg_vitb16_ftall (run01) | 0.9068 / 0.96 | 0.8391 / 1.00 |
| seg_r02_frozen | 0.8685 / 0.63 | 0.9583 / 1.00 |
| seg_r02_ft4 | 0.8703 / 0.73 | 0.9606 / 0.96 |
| seg_r02_ftall | 0.8636 / 0.80 | 0.9585 / 1.00 |

Per category on run02 test (dataset IoU, r02_ft4 / run01 ft4): arch 0.95 / 0.95, awash 0.98 / 0.70, blow 0.94 / 0.93, breach 0.94 / 0.93, fluke_up 0.95 / 0.95, logging 0.96 / 0.95, pec_up 0.93 / 0.91, spyhop 0.97 / 0.97,

Reading:
- On run02 the three variants are again within 0.003 of each other, now at 0.96 instead of
  0.90. Most of the gain is the data, not the model: the run01 models score 0.84-0.87 on
  run02, and the gap is almost entirely the awash category (0.58 -> 0.98), i.e. the waterline
  defect that run01 trained them on. The through-water ignore band removes the other large
  error source from the score (through-water pixels are no longer counted either way).
- The run02 models drop to 0.86-0.87 on run01's test split with poor negatives-clean
  (0.63-0.80): they were never told that a body seen through the water is background, so
  they now segment run01's submerged "negatives" and the whale under run01's too-high
  labels. That is the intended behaviour change; on real photos it means a faintly visible
  submerged body will tend to be included. If that is unwanted, lower
  `labels.through_water_depth_m` and re-run `factory.postprocess` (no re-render needed).
- Checkpoint to carry forward: `seg_r02_ft4` (best on both test splits). `seg_r02_frozen`
  is the conservative alternative (no empty-frame false positives on run02).

### Body coordinates + pose, run02 test split (seg crops from seg_r02_ftall)

| metric | body_r02_ft4 (oracle crops) | body_r02_ft4 (seg crops) | body_r02_frozen (seg crops) |
|---|---|---|---|
| u MAE (fraction of body length) | 0.0234 | 0.0223 | 0.0316 |
| angle around body MAE | 10.2° | 10.3° | 12.0° |
| visibility IoU (in crop) | 0.886 | 0.887 | 0.875 |
| rotation error, direct head, median / < 15° | 12.3° / 62 % | 12.6° / 61 % | 19.0° / 39 % |
| rotation error, PnP, median / < 15° | 17.6° / 45 % | 17.1° / 44 % | 31.3° / 22 % |
| landmark PCK@0.1 | 0.857 | 0.871 | 0.807 |

Reading: run02 is a harder body task than run01 because 85 % of the whales are bent (arch,
tail curl, lateral stroke, fin raise), while the targets are still the rigid pose of the
undeformed mesh and the canonical body table. The dense predictions hold up (u MAE 2.2 %,
PCK 0.87, up from 0.82), the direct rotation head is about where it was (median 12-13°),
and PnP suffers most (median 17-18°, it fits a rigid shape to a bent one). Fine-tuning the
last four blocks still beats the frozen backbone on every metric. A bent-body-aware pose
(fit the deformation parameters too) is the next step if rotation matters more than
correspondence.

### Real photos, first pass (2024-12-18, 336 Canon CR2/CR3 frames, no labels)

`test_scripts/python/convert_raw.py` decodes the RAW files (rawpy, half-size demosaic,
2560 px long side) into `experiments/real_2024-12-18/images`; `pred_r02/` is
seg_r02_ft4 + body_r02_ft4 (`bodycoords.predict`, 1280 px inference), `seg_seg_r02_frozen/`
and `seg_seg_vitb16_ft4/` the other two segmentation models on the same frames, and
`sheets/` contact sheets. No calibration (CR3 EXIF not readable here), so only rotations
and masks are meaningful.

What the masks look like (eyeballed on ~30 frames, three models side by side):
- Above-water backs, dorsal fins and raised flukes are segmented tightly by all three,
  including small distant ones (a few thousand pixels at 1280 px). Median mask agreement
  between the models is 0.90 IoU; 334 / 336 frames get a mask.
- Where run02 helps: whales awash in glare / flat light that run01 misses entirely
  (184A2556 and neighbours): the waterline fix carried over to real water.
- Where everything fails: foreground distractors that the synthetic data never contained.
  A researcher in a vest on the bow, a tagging pole, out-of-focus equipment, and a
  headland at the frame edge are segmented as whale; the run02 models (trained on large,
  close whales) do it more than run01, the frozen backbone slightly less than ft4.
- Submerged bodies seen through the water at close range are left unlabelled (184A2588),
  consistent with the label definition.
- Pose: the direct head and PnP disagree by 50° median on real frames (12 % under 15°), vs
  ~10° on synthetic. The body-coordinate maps look plausible on large clean backs, but the
  rotation is not trustworthy on real photos yet; treat it as untested.

Next steps, in order of expected payoff: (1) distractors in the factory (people, boat
parts, poles, land / islands, other boats) so the segmenter learns what is *not* whale;
(2) a small hand-labelled real set (even 30 frames) to replace eyeballing with numbers;
(3) only then revisit the pose head on real data.

### Body-coordinate diagnosis (2026-10-12): why left/right is patchy and the keypoints scatter

Tools: `side_diag.py` (scratch; outputs in `experiments/real_2024-12-18/side_diag/` and
`experiments/body_r02_ft4/side_diag/`): run the body head on a crop and on its mirror,
un-mirror, and compare the side (sign of sin theta) per pixel; on synthetic crops also
against the ground truth.

| | real crops (120) | synthetic test crops (256) |
|---|---|---|
| crops where > 20 % of pixels change side when the crop is mirrored | 24 (20 %) | 22 (9 %) |
| mean fraction of pixels changing side under mirroring | 0.16 | 0.07 |
| rotation head: change under mirroring, median / mean | 14.5° / 24° | 9° / 16° |
| side wrong vs GT (pixels > 10° from the seam) | – | 5.6 % overall; **pec_up 56 %**, profile views 9 %, back views 0 % |
| theta error vs GT, median: all / near seam / on the flank | – | 2.4° / 1.8° / 5.5° |

Findings:
1. Which flank a pixel is on is a *global* property (head direction x dorsal direction x
   viewing side), but the head regresses (sin, cos) of theta independently per pixel, so
   the left/right bit is decided pixel by pixel from weak local cues. It is unstable under a
   mirror on a fifth of real crops, and on synthetic rolled whales (pec slaps, belly-up
   breaches) the whole visible flank is assigned to the wrong side more often than not.
2. On real backs the angle collapses to the dorsal seam (theta ~ 0 over the whole back):
   under domain shift the regression falls back to the training prior, which is the ridge
   (whales float ridge-up, so most training pixels are near theta = 0). The few places that
   do commit to a side are then arbitrary; that is the "mix" seen in the overlays.
3. The (u, theta) parameterisation is degenerate on the appendages: the right fluke blade
   (2.4 m wide, 518 vertices) occupies theta 103-122° at constant u, so tip and mid-blade
   share the same coordinates; pectoral fins 32 vertices per (1 % u x 5°) cell vs 7 on the
   torso. No network can map fluke points in these coordinates and the PnP lookup is
   ill-posed there.
4. The scattered yellow points are the canonical landmarks reprojected with the PnP pose,
   which fails on real photos (direct vs PnP rotation 50° median) for the reasons above plus
   bent bodies and unknown focal length. They are not the dense map.
5. Smaller: the 1 - cos loss is nearly flat within ~20° of the seam, so the seam can wander
   cheaply; the texture atlas has distinct UVs for both sides (no mirrored texture), so
   appearance cues for the side exist in principle but are random mottling.

Proposed redesign (see the chat of 2026-10-12): canonical 3D surface point per pixel with
the side folded out of the dense target, (x, |y|, z); the side decided once per whale from
the pooled rotation (or deferred to the matching stage by testing both sign hypotheses);
landmarks read off the dense map; flip-consistency as a reported confidence and, later, as
a label-free loss on real photos.

## 2026-10-12 — canonical surface points (canon_r02_ft4): the body-coordinate redesign

Changes (all in this repo and the factory, no re-render):
- Labels: `canon/<id>.npz` per shot, the point on the undeformed mesh seen at each pixel
  (factory `postprocess --canon-only --workers 32`, 2000 shots in ~3 min).
- Target: folded, normalised canonical point ((x - x_min)/L, |y|/L, z/L); the side (sign of
  y) is not regressed. Mirrored crops have identical targets, so the flip augmentation needs
  no label surgery. L1 loss (w 10) + visibility BCE + rotation + scale, same recipe as the
  body model (last 4 blocks, 40 epochs, 23 min alone on the GPU).
- Side: once per whale, `geometry.resolve_side`: two global hypotheses (prior from the map's
  own head x dorsal directions when confident, else the rotation head), each refined by
  alternating RANSAC PnP and a per-pixel vote (reprojection residual + penalty when the
  surface would face away from the camera, from mesh normals in the folded table); the
  cheaper hypothesis wins. On GT maps: side accuracy 0.98-1.00, PnP rotation < 0.3°, and a
  deliberately wrong initial side is recovered (test_canon.py).
- Points of interest are read off the map (nearest predicted canonical point to each
  landmark, within 1 m); every crop is also run mirrored, the two folded maps are averaged
  and their disagreement reported (`flip_disagreement_median_m`), together with the side
  agreement of the two passes and the hypothesis margin.
- `canon/evaluate.py` (surface error in metres by region, mirror consistency, side accuracy,
  PnP with the resolved side, landmark PCK from the map and via PnP) and `canon/predict.py`
  (real photos; writes `<stem>_canon.npz` with signed points, side, use mask, disagreement).

### Synthetic, run02 test split (256 crops, TTA on), oracle crops

| metric | (u, angle) model body_r02_ft4 | canonical surface canon_r02_ft4 |
|---|---|---|
| dense error | u MAE 0.38 m along the body, angle MAE 10° | **surface error median 0.21 m** (mean 0.48 m; torso 0.42, flukes 0.81, fins 1.09 m) |
| mirror consistency of the dense map | side of 7 % of pixels changes (9 % of crops > 20 %) | folded map differs by 0.24 m mean; side (after resolution) agrees on 91.5 % of pixels, 6.6 % of crops unstable |
| side accuracy vs GT (pixels > 10 cm off the midline) | 94.4 % | 93.1 % (facing flank per crop: rotation head 94.9 %, map cue 94.1 %; rolled whales are the misses) |
| rotation, direct / PnP median | 12.3° / 17.6° | 12.6° / 14.5° |
| points of interest, PCK@0.1 body length | 0.86 (PnP-reprojected) | 0.66 from the map (found 68 %, found ones median error 0.8 % of length); 0.88 PnP-reprojected |

Seg-crop numbers are within 0.01-0.02 of these.

### Real photos (2024-12-18, 334 crops)

| | before (u, angle) | after (canonical surface) |
|---|---|---|
| crops where > 20 % of pixels change side when the crop is mirrored | 21 % | 20 % |
| folded coordinates, disagreement under mirroring | – | 0.32 m median (0.19-0.50 p25-p75) |
| rotation head change under mirroring, median | 14.5° | 15.6° (15 % of crops > 45°) |
| direct vs PnP rotation, median | 50° | 47° |

Reading:
1. The representation does what it was meant to do on synthetic data: the dense map is
   mirror-consistent to a quarter of a metre, accurate to 21 cm median on the torso, and the
   flukes and fins now have coordinates at all (0.8-1.1 m there: thin, deformed by fin raise
   and fluke twist, and the rigid canonical mesh does not model that bend). The patchy
   left/right of the old head is gone by construction; what remains is a single per-whale
   decision that is right 93-95 % of the time, with rolled whales (pec slaps, belly-up
   breaches) as the hard cases where neither the rotation head nor the map cue can tell.
2. On real photos the per-whale side decision is **not** more stable than before (20 % of
   crops still flip under mirroring). The cause moved: it is now the dense map itself drifting
   under domain shift (0.32 m mirror disagreement on real vs 0.24 synthetic, worst on large
   backs seen from behind), which makes the two global hypotheses nearly tied and leaves the
   choice to the rotation head, whose own facing-flank bit flips on ~15 % of real crops.
   The margin and the mirror statistics now flag those crops, which the old pipeline could not.
3. Mirror-consistency as a label-free loss on the real photos (canon_r02_ft4_real_COLLAPSED,
   joint with synthetic supervision, w 5, 10 epochs) collapsed: on real crops the model
   predicts an almost constant point (coordinate spread 2 cm vs 78 cm) while all synthetic
   metrics stayed normal. Pure consistency is satisfiable trivially; it would need a frozen
   teacher / stop-gradient scheme or, better, labels. Kept only as a record.
4. Landmarks from the map are precise when found (0.8 % of body length) but a third are not
   found within 1 m, mostly where the side is wrong or on fins; PnP-reprojected landmarks
   remain better as a *probe* because they integrate the whole map. For matching, the map is
   the product, the landmarks only a check.

What would move the real-photo numbers now: a small labelled real set (even 30 frames with
a handful of clicked points each) to measure and to fine-tune on; bent-body targets (the
factory knows the deformation per shot, so the canonical point could be the point on the
*deformed* mesh for fins and flukes) to bring fins and flukes to torso accuracy; and for the
clustering goal, matching under both side hypotheses so the per-whale decision never has to
be final. Checkpoint to carry forward: `canon_r02_ft4` with the v2 inference
(`eval_test_v2`, real predictions in `experiments/real_2024-12-18/canon_r02_ft4_v2/`).

## 2026-10-12 — clustering a day's photos by individual (2024-12-18, 336 photos, 17 whales)

Pipeline (`model_fine_tuning/cluster/`): canonical surface cells (512, k-means on the folded
mesh) -> per-photo atlas of DINOv3 patch features and LAB colour per visible (cell, side)
(`atlas.py`; frozen DINOv3 blocks 8 and 11, and the canon-finetuned backbone) -> pairwise
scores: per-cell z-scored similarity over the cells both photos show, under both side
hypotheses (sides as resolved, or photo B mirrored; the matcher may take the better), plus a
step time term (+1.5 within 2 s, +1 within 30 s, +0.5 within 5 min, 0 within 1 h, -0.5
beyond) (`match.py`) -> average-linkage clustering, threshold self-calibrated from the day's
own bursts (<= 2 s) and far pairs (>= 90 min), never from labels (`cluster.py`) -> link
heatmaps per cluster: per-cell (or per-patch) contribution painted on both crops, with the
time term in the header (`explain.py`). Evaluation only: individual labels read from the
Lightroom catalog collections (`labels_from_lightroom.json`; G07A01/02 = G04A01/02 per the
collection names), capture times from EXIF (CR3 parsed from its CMT boxes).

Pair-level AUC (same individual vs different), labels used for scoring only:

| appearance score | all pairs | same group | competition group G02 | pairs > 10 min apart |
|---|---|---|---|---|
| per-cell DINOv3 block 11 (frozen) | 0.60 | 0.55 | 0.65 | 0.57 |
| same, canon-finetuned backbone | 0.60 | 0.58 | 0.63 | 0.57 |
| per-cell LAB colour / lighting-normalised | 0.60 / 0.56 | 0.54 / 0.55 | 0.65 / 0.53 | 0.55 / 0.53 |
| 2x-resolution features, 128-cell grid, both-hypothesis max | all within 0.55-0.61 | | | |
| patch-level aligned-minus-misaligned (`local.py`), radius 0.3 / 0.6 m / unaligned | 0.60 / 0.62 | 0.59 / 0.62 | 0.64 / 0.62 | 0.54 / 0.57 |
| SIFT keypoints within the canonical radius | 0.50 | 0.47 | 0.57 | 0.49 |
| dorsal-fin crop (`fin.py`): CLS / pooled / silhouette | 0.63 / 0.59 / 0.54 | 0.58 / 0.55 / 0.51 | 0.71 / 0.66 / 0.59 | 0.57 / 0.57 / 0.47 |
| **time term alone** | **0.91** | 0.63 | 0.75 | – |
| time-contrastive head on cell features (`identity.py`, no labels) | 0.94* | 0.70-0.76 | 0.80-0.82 | 0.84* |

\* inflated by construction (the head is trained with far pairs as negatives). The honest
subset is pairs in the same group 10 s to 1 h apart, never positives nor negatives:

| head trained on | 7D-7D pairs | R5-R5 pairs | cross-camera |
|---|---|---|---|
| none (raw cosine) | 0.52 | 0.52 | – |
| all photos (transductive, the deployment protocol) | 0.75 | 0.996 | 0.73 |
| 7D photos only (R5 held out) | 0.73 | 0.45 | 0.59 |

Clustering (ARI / pair precision / pair recall / clusters, 17 individuals), self-calibrated threshold:

| score | ARI | P | R | clusters |
|---|---|---|---|---|
| time only (threshold 0.25) | 0.59 | 0.86 | 0.51 | 23 |
| DINOv3 cells + time | 0.29-0.31 | 0.76-0.86 | 0.23 | 35-40 |
| colour cells + time | 0.40 | 0.85 | 0.31 | 29 |
| local aligned + SIFT + time | 0.43-0.56 | 0.72-0.82 | 0.35-0.51 | 20-34 |
| identity head + time | 0.58 | 0.83 | 0.52 | 28 |
| identity head alone | 0.43 | 0.78 | 0.35 | 49 |

Reading:
1. Generic features do not carry identity. Every variant of DINOv3 (frozen, fine-tuned, 2x
   resolution, cell means, patch-level aligned matching, fin crops) and SIFT sits at 0.55-0.65
   pair AUC. The heatmaps show why: the strongest off-burst links pair photos of the *same
   view* (a dorsal fin from the side) of different whales taken hours apart. Alignment is not
   the bottleneck (radius 0.3 m, 0.6 m and no alignment give the same AUC).
2. Time does most of the work (ARI 0.59) and fails exactly where it must: inside groups.
   Mother/calf/escort trios and the competition group are interleaved within seconds; the
   time-only clusters mix G03A01/A02/A03, G05 with G06, G02A03 with G02A04.
3. A head trained from the day's own timestamps (same cell within 10 s = same whale, 1 h apart
   = different) is the first thing that separates animals inside a group: honest AUC 0.52 ->
   0.75 on the main camera. It is transductive: it must be fitted on the photos it will
   cluster (held-out camera: chance), and on the R5 camera part of its power is photographer
   behaviour (focal length and whale size differ *more* between same-animal pairs there,
   AUC 0.10 / 0.17 for their differences, i.e. strongly structured), not skin. It does not yet
   improve the clustering (ARI 0.58 vs 0.59): a 0.75 pair AUC is not enough for transitive
   merging and the time term still glues interleaved groups together.
4. What the heatmaps deliver: for every cluster, the maximum-spanning-tree links with both
   crops painted by per-cell (or per-patch) contribution, the number of body cells compared,
   which side hypothesis was used, and the time difference with its term
   (`experiments/real_2024-12-18/cluster_id/explain/`). Within bursts the whole shared body
   lights up (trivial); the informative links are the sparse red cells on dorsal fins and
   flank patches between bursts.

What would change the picture: identity-supervised training on *other days*. The Lightroom
catalog holds about 11,700 labelled photos across dozens of days (G..A.. collections for
2024-12-23, 2024-12-26, 2025-01-15 ...). Training a metric embedding on canonical-aligned
cells (or fin and fluke crops) with those days' labels and testing on 2024-12-18 is the
standard, legitimate protocol (train/test split by day) and the only route to a real
appearance signal; the day's timestamps then serve as a prior, not the backbone. Second,
treat the time term as a chain prior (consecutive-photo continuity) rather than a global
pairwise bonus, so that an appearance break inside a group can cut a chain. Third, fluke
undersides (30 photos here) deserve a dedicated pattern matcher for the HappyWhale step.

### Part charts at full resolution (2026-10-12): the researcher's method, tested

`cluster/chart.py` cuts the dorsal fin and its surround from the RAW at full sensor
resolution (R5 8192 px, 7D 5472 px), normalised for pose with the canonical map: mirrored to
a common flank, rotated so the head points right and the fin up (body axis = principal axis
of the visible mask, head end = sign of the body-fraction gradient), scaled to the fin box.
`cluster/finshape.py` then isolates the fin silhouette from the mask (protrusion above a
wide horizontal opening) and describes it by its outline, trailing edge, aspect, and the
DINOv3 CLS of a tight crop. Scored on 233 fin charts (203 silhouettes); pair AUC, labels for
scoring only:

| descriptor | all pairs | near-profile pairs (both |nose_right| > 0.6) | same group, near-profile |
|---|---|---|---|
| chart DINOv3 patches, position-aligned minus misaligned | 0.59 | **0.72** | 0.67 |
| chart SIFT + RANSAC affine inliers | 0.53 | 0.62 | 0.67 |
| fin silhouette: outline / trailing edge / aspect | 0.51 / 0.56 / 0.53 | 0.58 / 0.62 / 0.61 | 0.58 / 0.62 / 0.61 |
| tight fin crop DINOv3 CLS | 0.59 | 0.70 | 0.62 |
| time term | 0.93 | | |

Reading: resolution and pose normalisation are now right (sheets in the scratch dir and
`experiments/real_2024-12-18/cluster_finshape/fins/`), and to the eye the isolated fins of
different individuals are distinct (hooked with a notch, tall triangle, low rounded). Gating
on viewing angle is what moves the numbers: fins seen from the front or back are not
comparable, and once both photos are near-profile the chart features reach ~0.7. The
hand-made silhouette distances stay weak because the base line of the fin is ambiguous and
foreshortening changes the outline more than identity does; SIFT finds few marks on smooth
grey fins. The remaining gap to a usable matcher is a *learned* comparison of these
normalised fin charts (and fluke undersides), trained on the catalog's other days.

## 2026-10-12 — plan for the identity model on the full catalog (prepared before the data arrived)

Data: `/datasets/NOAA_whales_data/<day>/...` (same layout as `test_data_do_not_use`), 40
labelled days, 11,400 labelled photos, 517 identities after merging the catalog's "same as"
notes (483 day-identities with >= 2 photos). Labels: `model_fine_tuning/splits/labels_all_days.json`
(`cluster/labels.py`, read from the catalog; the "O" = other animals are excluded as classes).

Split by whole days (`splits/days.json`), frozen now: eval = 2024-12-18, 2024-12-26,
2025-02-19, 2025-03-21 (2,167 photos, 92 identities; never used for training, selection or
tuning); val = 2025-01-15, 2025-02-27, 2025-03-14 (889 photos; epoch and hyper-parameter
selection); train = the other 21 days with >= 40 labelled photos (8,065 photos, 308
identities). Identities merged across days never straddle the splits (checked).

Per day (`test_scripts/bash/process_day.sh`, driven by `process_all_days.sh` as folders
finish transferring): RAW -> 2560-px jpg + EXIF time -> segmentation + canonical surface
(seg_r02_ft4 + canon_r02_ft4) -> part charts from the full-resolution RAW (`cluster/parts.py`):
fin (right-flank view, head right, fin up, silhouette), fluke (span horizontal, peduncle up,
right tip right; underside or top), body (whole visible whale, right-flank view, head right).
On 2024-12-18: 328 body, 233 fin, 75 fluke charts.

Objective (`cluster/train_id.py`): one DINOv3 ViT-B/16 (last 4 blocks fine-tuned) + a 512-d
embedding with a part token; ArcFace (scale 30, margin 0.4) with one class per identity and
one class-centre set per part type, plus supervised contrastive within part type. Batches
are P=8 identities x K=4 charts drawn mostly from one day, so negatives are the animals
photographed together (the clustering case), with a second day mixed in 30 % of the time.
Augmentation: random resized crops, +-8 deg rotation, exposure / white-balance jitter, blur;
no horizontal flips (charts are mirrored to one flank and marks are asymmetric). Selection:
mean same-day pair AUC over parts on the val days. Smoke-tested end to end on 2024-12-18.

Inference (`cluster/eval_id.py`): photo-pair appearance = best over shared part types of the
embedding cosine, fins only when both photos are near-profile (|nose_right| >= 0.6), fin and
body also against the other photo's mirrored chart (the flip hypothesis), z-scored per day;
then the unchanged clustering (time term, self-calibrated threshold) and explanations.
Baselines reported per eval day: time only; identity + time; identity alone.
