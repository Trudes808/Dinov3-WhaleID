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

## 2026-10-14 — run04: contour-weighted losses (seg_r04_cw / canon_r04_cw) vs the plain recipe (seg_r04_ft4 / canon_r04_ft4)

Run04 (Whale-Image-Data-Factory, rigged 4k humpback, individuals, mom + calf, frozen waves, 10 000 shots) labels every
boundary pixel of the visible mask: 1 = the whale's own silhouette (open air against sky or water), 2 = a seam where
the body continues under the surface or behind foam, 3 = the edge against the calf. In a real photo the top contour is
the trustworthy one and the waterline / foam edges are not, so the losses now weight the pixels within 3 px of the
contours: silhouette x4 (segmentation BCE, visibility BCE) and x3 (canonical-coordinate L1), seams x0.3 (BCE only;
the seam's coordinates are exact, its position in a photo is not). `common/contours.py`; flags on both trainers.

Evaluation adds the boundary F-score split away from / at the seams (`boundary_f1_open` / `boundary_f1_seam`) and
the recall of the silhouette pixels, for the segmentation model and the visibility head. `scripts/run04_train.sh`
trains the contour-weighted variant, evaluates it, then the plain run03 recipe on the same split (the ablation);
`run04_evaluate.sh` also evaluates the run03 models on the run04 test split, so the synthetic comparison is like
for like; `run04_report.py` writes `experiments/run04_report.md`. Smoke-tested end to end (3 steps, h4k_test12).
The Whale-Image-Data-Factory pipeline (`tools/run04_pipeline.sh`) runs both scripts after the render.

Dev rehearsal (run04_dev: 752 shots from 21 finished run04 batches, 6 epochs per model, both variants): the
training chain, evaluation and report run end to end; at 6 epochs the weighting changes nothing measurable
(segmentation dataset IoU 0.960 vs 0.959, pooled open-air boundary F1 0.805 vs 0.803, run03 model 0.800 on the
same images; run04_dev_report.md). The open-air score is scored along the gt silhouette only (predicted boundary
within 8 px of it): scored over the whole "far from seams" zone it was dominated by predictions straying a few
pixels past the waterline or foam edge, precision 0.2, which is seam error, not silhouette error. The run03 canon model on the
run04 dev test split: surface error 2.8 m, rotation median 100 deg, facing accuracy 74 % (own split: 0.25 m,
9 deg, 91 %): it mirrors the side on run04's close-ups and head-on views; the labels themselves were checked
(same canonical frame as run03, GT rotation within ~10 deg of a PnP on the factory keypoints for straight poses).
Boundary scores are pooled over the split (`common.contours.pooled_boundary_scores`) because a surfacing whale
seen from an elevated camera has almost no open-air contour (its back ends at the far waterline: all seam).

Results, contour-weighted pair (2026-10-16 03:46 UTC; seg 1.9 h, canon 1.9 h on 6 676 / 6 296 train images;
run04_report.md). Synthetic, run04 test split, like for like against the run03 models on the same images:
segmentation IoU 0.961 vs 0.935, open-air boundary F1 (pooled) 0.857 vs 0.790, silhouette recall 0.72 vs 0.54;
canon surface error 0.43 m vs 2.37 m (median 0.27 vs 1.63), rotation within 15 deg 68 % vs 2 % (direct head;
run03 on its own easier split: 74 %), side accuracy 0.87 vs 0.66, landmark PCK@0.1 0.90 vs 0.51. The PnP
rotation is poor for the run04 model even on synthetic data (median 31 deg, 10 % within 15 deg; run03 on its own
split 12.7 deg): a rigid K9239 fit against a dense map of bent bodies (every run04 shot is a keyframe pose).
Real photos (2024-12-18, label-free outline metric): worse than run03, with the PnP pose (above-water IoU median
0.20 vs 0.31, 71 % of photos worse) and with the rotation from the direct head and a fitted translation (0.23 vs
0.31; reprojection error 30 px vs 21 px). Mirror test on real photos: equal (side flips 18 % vs 17 %, folded
disagreement 0.30 m vs 0.33 m). On the photo that worsened most (a raised fluke) both maps are the same up to
the left/right assignment, and the rigid PnP then lands on different poses: the fluke-only side ambiguity.
Plain ablation (2026-10-16 08:02 UTC, the pipeline's last stage): on the synthetic test split the two variants
are within noise of each other (seg IoU 0.961 vs 0.960, open-air boundary F1 0.857 vs 0.851, surface error 0.43
vs 0.44 m, rotation 68 % both); on the real photos the plain recipe is slightly better (outline IoU 0.232 vs 0.197
with PnP, 0.235 vs 0.228 with the direct rotation; folded mirror disagreement 0.25 vs 0.30 m). The contour
weighting as configured (x4 / x0.3, 3 px) is a null result: keep the plain recipe as the run04 reference
(seg_r04_ft4 / canon_r04_ft4) unless a stronger setting is tried.
Domain check (experiments/run04_domain_check/): the real-photo gap is explained by the data, not the losses.
Open questions for the plain ablation (training) and after: whether the weighting costs anything on real
photos, and why the run04 map fits real photos worse (render look / pose distribution / the 4k whale), since the
synthetic gains are large. The dense map + rotation head on real photos is the number to improve; the rigid PnP
is the weak link in both evaluations and the bendable fit is the planned replacement.

## 2026-10-16 — mesh fit: bendable fit instead of the rigid PnP (canon/fit_mesh.py), and the run05 pose fix

The rigid PnP of the dense map is the weak link of every real-photo number (a bent whale cannot be matched by a rigid
mesh; the rotation goes with it). `canon/fit_mesh.py` fits the pose and the factory's body bend (arch, tail curl, head
pitch, lateral sweep; `factory/deform.py` imported from the sibling repo) to the prediction: (1) rotation from the
direct head, translation by a linear least squares on the correspondences (PnP-independent: the PnP translation is
wrong whenever its rotation is); (2) robust least squares on the dense correspondences with a soft prior on the bend;
(3) a bounded Powell refinement on the rendered above-water silhouette: per-column top edge against the mask's top
edge (the whale's own contour; the waterline is never matched), mesh above the whale, columns beyond the mask, excess
above water. The silhouette renderer used every third face (a wireframe decimation that leaked from the review app)
and left holes; fixed in outline_metric too (its numbers for all models were refreshed with the full mesh).
Synthetic test crops of run04 (48, oracle crops, scored against the true mask and rotation, canon_r04_ft4, K9239
mesh): above-water IoU vs truth 0.40 (rigid PnP) -> 0.51 (bend), top-edge error 4.5 -> 2.0 px, rotation error
33 -> 25 deg median. Real photos and the real-vs-synthetic grid: experiments/real_2024-12-18/fit_*/ and
fit_grid_canon_r04_ft4.jpg (orange rigid PnP, green bend, white refined). `scripts/fit_real_photos.sh`,
`scripts/fit_synthetic.sh`, `canon/fit_grid.py`.

Experiment 3a: canon_r04_ft4_real = canon_r04_ft4 fine-tuned 6 epochs with the mirror-consistency loss on the real
photos, EMA teacher with stop-gradient (`--consistency-teacher ema`; the two-branch loss of 2026-10-12 collapsed),
w 1.0: no collapse (coordinate spread on real crops 0.05-0.09 body lengths throughout), val score 0.713 vs 0.708.
Experiment 3b: canon_r0304_ft4 = the plain recipe on run03 + run04 together (`common.data.CombinedRun`,
`--extra-run-dir/--extra-split`, `scripts/run0304_combined.sh`). Results: see the follow-up entry.

Run05 pose finding (factory): the straight dive-start keyframes carry the head high and the dorsal fin low in the
levelled frame (zero bend: snout 70 cm above the dorsal tip), so a mild hump centred mid-body exposes the blowhole-to-
rostrum region, not the back (first test render: 78 % of the surfacing pixels on the rostrum). The `surface` pose is
therefore a milder back_arch with the apex at the dorsal fin (arch -55..-30 at body fraction 0.36-0.42, head pitch
-30..-12, tail curl -25..-8), with per-category pitch/roll jitter (4/8 deg) and the back-height clearance measured
without the head (x < 450 cm). Framing: extent 0.08-0.7, 70-600 mm, 50-400 m. Config factory/config/run05_realism.json.

## 2026-10-16 — redesign started: CSE body-part map, WildFusion baseline

Plan: docs/correspondence_redesign.md. Built: the 4k template (assets/humpback4k_template.npz: 4 000 farthest-point
vertices, mesh geodesics, part / side / dorsal-ventral tables, k=16 geodesic soft-target neighbours, mirror table),
per-pixel vertex labels for run04 (common/vertex_labels.py, 10 000 shots), the CSE head and loss (canon/cse.py: 32-D
pixel embedding vs a learned vertex table, soft cross-entropy, visibility BCE, the pose head kept; inference with a
whale-level side vote that snaps pixels to their mirror vertex) and its trainer (canon/train_cse.py; selection = share
of labelled val pixels within 30 cm geodesic). Training launched: cse_r04_frozen (backbone frozen) then cse_r04_ft2
(last two blocks). WildFusion baseline (scripts/wildfusion_baseline.py; MegaDescriptor-L-384 cosine, ALIKED +
LightGlue, isotonic calibration on half the identities, fused pair AUC on the other half; uncalibrated AUCs over all
pairs for comparison with the current pipeline) launched on 2024-12-18. Results: next entry.
Experiment 3a (canon_r04_ft4_real, EMA-teacher mirror consistency): null on real photos (outline IoU 0.21 vs 0.22).

WildFusion baseline on 2024-12-18 (334 labelled crops, 17 identities): pair AUC over all same-day pairs,
MegaDescriptor-L-384 cosine 0.662, ALIKED + LightGlue 0.592; on the held-out half of the identities (9 ids, 180
photos) MegaDescriptor 0.654, LightGlue 0.570, calibrated WildFusion 0.651; the current identity pipeline's
appearance score (id_smoke, `id__same`) on the same photos: 0.519 (all) / 0.518 (half), i.e. near chance.
Whole-whale appearance on a single day of mostly surfacing backs is a hard problem for every method; the part-aligned
matching of the redesign is the hypothesis for beating 0.65. experiments/days/2024-12-18/wildfusion/metrics.json.

Bendable fit on the real photos (335, above-water IoU median, rigid PnP -> silhouette-refined bend; K9239 mesh):
run03 0.306 -> 0.319, run04 plain 0.220 -> 0.284, run04 real-fine-tuned 0.221 -> 0.279; top-edge error 3-8 px ->
2 px for all. With the 4k canonical mesh the run04 fit is worse (0.176 -> 0.188), but the stored predictions were
unfolded onto K9239 vertices, so that is not a clean comparison. The bend fit helps the bent-pose models most and
still does not reach the run03 rigid fit; superseded by the correspondence redesign (docs/correspondence_redesign.md).

CSE frozen backbone, epoch 15 checkpoint on the 2024-12-18 photos (canon/predict_cse.py, 335 photos): predicted
whale pixels by part fluke 0.29 / peduncle 0.18 / dorsal back 0.42 / mid body 0.06 / head 0.04 (the real profile);
mirror consistency: part agreement 0.79 median, side agreement 0.61, geodesic disagreement 264 cm median. The
overlays (experiments/real_2024-12-18/cse_r04_frozen_ep5/sheet.jpg) read correctly part by part with some
patchiness; the side remains the unstable quantity.
Charts from the CSE map (cluster/parts.py on cse_r04_frozen_canon, epoch 17): body 331, fin 229, fluke 184 (old map:
328 / 233 / 75). Of the 111 extra fluke charts about half are genuine raised flukes the old map missed and half are
backs with a fluke-labelled patch (parts_cse/extra_flukes_sheet.jpg): chart gating should require a majority of the
chart's pixels in the part, not just enough of them. Per-part WildFusion scoring of both chart sets: next entry.

cse_r04_frozen (backbone frozen, 30 epochs, 3.3 h shared GPU), run04 val (600 crops): part accuracy 0.910, raw per-pixel
side accuracy 0.881, geodesic error median 93 cm, within 30 cm 0.196 (vertex spacing ~40 cm), per-part coverage error
0.054, visibility IoU 0.894. The all-pixel side snap of the first vote lowered side accuracy to 0.750 (both flanks are
often legitimately visible); the vote now moves only ambiguous pixels and selection is on the raw map. cse_r04_ft2
(last two blocks tuned) started 2026-10-16 18:41 UTC with the corrected code.

Per-part matchers on the charts (scripts/wildfusion_parts.py, 2024-12-18, pair AUC; MD = MegaDescriptor cosine, LG =
ALIKED + LightGlue). Old map (parts: 328 body / 233 fin / 75 fluke): body MD 0.640, fin 0.597, fluke 0.664, best over
shared parts MD 0.631, LG 0.536. CSE map, epoch-17 frozen checkpoint, ungated (331 / 229 / 184): body 0.622, fin 0.650,
fluke 0.634, best over shared parts MD 0.644, LG 0.573. Gated charts (fluke >= 30 % of the whale's pixels, fin >= 3 %;
331 / 229 / 108): next entry. Whole-whale WildFusion for reference: MD 0.662 all pairs, fused 0.651 on the held-out half.
Gated CSE charts (331 / 229 / 108): body 0.622, fin 0.650, fluke 0.651, best over shared parts MD 0.627, LG 0.543,
MD+LG 0.603; ungated 0.644 / 0.573 / 0.627; old map 0.631 / 0.536 / 0.600. Gating raises the fluke chart AUC but
loses pair coverage; the CSE map beats the old map by ~0.02-0.03 on every fused number but no part-based variant yet
beats whole-whale MegaDescriptor (0.662) on this day of 17 whales (differences of 0.02 are near the noise of the set).
Fusion of the whole-whale MegaDescriptor score with the best-shared-part chart score (z-scores, mean; common photos,
n=330): whole 0.665 -> 0.673 with the CSE charts (ungated), 0.664 with the old-map charts, 0.662 with the gated CSE
charts. Local matching in chart space (LightGlue) stays at 0.54-0.60 on every chart set: the 512-px charts cut from
2560-px photos may not carry the marks, and this day's backs show few. Everything on 2024-12-18 sits within ~0.02 of
each other for every method; a second labelled day is needed before any identity conclusion.

## 2026-10-18 — CSE 2-block result, combined canon model

cse_r04_ft2 (last two DINOv3 blocks tuned, 30 epochs): run04 val part accuracy 0.928, raw side accuracy 0.902, geodesic
error median 61 cm (frozen: 0.910 / 0.881 / 93 cm), within 60 cm 0.474, per-part coverage error 0.030, visibility IoU
0.90. The ambiguity-aware side vote no longer hurts (0.902 either way). Tuning the last blocks helps the correspondence
clearly; the real-photo mirror consistency decides whether it also transfers (next entry).
cse_r04_ft2 on the 2024-12-18 photos (predict_cse): mirror part agreement 0.87 (frozen 0.78), mirror side agreement
0.80 (frozen 0.57-0.61), mirror geodesic disagreement 177 cm (264), ambiguous pixels 10 %: tuning two blocks transfers.
canon_r0304_ft4 (run03 + run04 combined, plain recipe, experiment 3b): synthetic run04 test slightly worse than run04-only
(surface error 0.473 vs 0.441 m, rotation < 15 deg 0.661 vs 0.677); real photos better than either single-run model:
outline IoU 0.318 rigid PnP (run03 0.306, run04 0.223), bend fit 0.323 -> 0.360 (the best mesh-fit numbers so far).
The flat run03 poses matter for the real photos; run05's surfacing poses are the same lesson applied to the renders.

## 2026-10-18 — all days available: burst-level clustering into animals and groups

Data: /datasets/NOAA_whales_data (Lightroom images/Topside research/2025 Research/<"YYYY-M-D description"> folders,
11 739 RAW; Test.lrcat identical in content to splits/labels_all_days.json: 40 days, 11 400 labelled photos, 517
identities, group types). Processing with the v2 day pipeline (test_scripts/bash/process_day_v2.sh: RAW -> 2560-px
jpg (parallel decode) -> seg_r04_ft4 + canon_r0304_ft4 crops / rotation -> cse_r04_ft2 map in the canon layout ->
gated part charts -> EXIF timestamps) into experiments/days_all (root disk), eval days first (process_all_days_v2.sh).
Bursts (cluster/bursts.py: one camera folder, gaps <= gap s) on 2024-12-18: gap 1 s -> 149 bursts, 98 % of the
multi-photo bursts hold one animal, 241 of 336 photos covered; 2 s -> 92 %, 304 covered; 4 s -> 90 %.
cluster/cluster_bursts.py: burst nodes, per-part z-scored MegaDescriptor cosine on the CSE charts, best over shared
parts, top-3 mean between bursts, step time term, self-calibrated threshold, interleaving rule for groups.
2024-12-18 (development day; it is an eval day, so from here on parameters are tuned on the val days only):
burst level ARI 0.415, pair P/R/F1 0.57/0.45/0.50, 28 clusters for 17 animals (oracle-threshold ARI 0.46); photo
level ARI 0.305, F1 0.36; the previous pipeline (id_smoke, photo level) ARI 0.39, F1 0.46. Groups (interleaving
within 120 s, >= 2 pairs): ARI 0.23, 9 groups for 8. Next: the identity embedding trained on the train days'
CSE charts (cluster/train_id.py), tuning on val, final numbers on the four eval days, WildFusion-on-charts and the
old pipeline as baselines.

Seven days clustered (v2 pipeline charts, MegaDescriptor, burst level, tw 1.0, self-calibrated threshold); animals
ARI / pair F1 / purity, clusters for animals: eval 2024-12-18 0.384 / 0.48 / 0.68 (27 for 17), 2024-12-26 0.144 / 0.23 /
0.27 (16 for 30), 2025-02-19 0.155 / 0.26 / 0.41 (23 for 21), 2025-03-21 0.168 / 0.26 / 0.37 (14 for 24); val
2025-01-15 0.133 / 0.28 / 0.46, 2025-02-27 0.508 / 0.56 / 0.73, 2025-03-14 0.341 / 0.43 / 0.58. Groups by interleaving:
ARI ~0 everywhere. The big days over-merge (purity 0.27-0.41) and even their oracle thresholds reach only 0.27-0.41,
so the appearance signal is the limit there; the identity model trained on the train days is the next lever.
Tuning on the val days from the saved matrices (cluster/recluster.py): time weight 1.5 (val ARI 0.327 -> 0.393; eval
0.213 -> 0.287, for the record); groups by interleaving within 30 s with >= 10 photo pairs (val group ARI ~0 -> 0.584,
eval 0.455). These are the new defaults of cluster_bursts.py.
Whole-crop MegaDescriptor through the same burst clustering (tuned defaults), animal ARI: eval 2024-12-18 0.489,
2024-12-26 0.165, 2025-02-19 0.235, 2025-03-21 0.302 (mean 0.298); val 0.445 / 0.587 / 0.362 (mean 0.465). The part
charts with MegaDescriptor at the same settings: eval 0.459 / 0.148 / 0.283 / 0.257 (0.287), val 0.342 / 0.494 /
0.344 (0.393). With an untrained descriptor the whole crop wins; the identity model trained on the charts decides.
Offline fusion of the two saved photo-score matrices (z-scores, mean where both exist): animal ARI eval mean 0.320
(whole 0.298, parts 0.287), val 0.457 (whole 0.465); the max rule 0.339 / 0.364. Parts add a little to whole crops on
the eval days and nothing on the val days with MegaDescriptor; the trained identity model is the real test.

id_v2 (cluster/train_id.py on the train days' CSE charts: 15 166 charts, 263 animals, 21 days; 15 epochs, 35 min):
val same-day pair AUC body 0.841, fin 0.841, fluke 0.872 (MegaDescriptor on the same charts: 0.62-0.66 on 2024-12-18).
Burst clustering with it (tuned defaults), animal ARI / F1 / purity: eval 2024-12-18 0.500 / 0.55 / 0.90, 2024-12-26
0.271 / 0.31 / 0.51, 2025-02-19 0.318 / 0.37 / 0.60, 2025-03-21 0.278 / 0.33 / 0.57 (mean ARI 0.342; parts MD 0.287,
whole MD 0.298, fusion 0.320); val 0.443 / 0.644 / 0.407 (mean 0.498; whole MD 0.465). Groups ARI 0.31-0.68.
The big days (30, 21, 24 animals) stay at ARI ~0.3 with oracle thresholds 0.29-0.36: appearance, not the threshold.
Fusion of id_v2 part scores with whole-crop MegaDescriptor (z-scores, mean): eval mean ARI 0.359, val 0.522 (id_v2
parts alone 0.342 / 0.498); the 0.75/0.25 weighting 0.349 / 0.491. Per day (eval): 0.503, 0.241, 0.323, 0.368.
Big-day diagnosis (id_v2 clusters, 2024-12-26 / 2025-02-19 / 2025-03-21): impure predicted clusters are almost
all merges of animals from the SAME true group (21/25, 10/14, 14/17 impure clusters); different-group pairs are merged
only ~1 % of the time, while different-animal same-group pairs are merged 15-20 % and same-animal pairs only 29-35 %
(22, 9, 13 animals split over several clusters). So the time/interleaving term already separates groups; the remaining
error is within-group confusion plus fragmentation of the same animal, i.e. appearance-limited.

## 2026-10-19 — follow-ups: hi-res charts, instances/tracklets, hi-res DINO head, fusion (in progress)

Plan (user request: train the follow-ups, add a DINO-based clustering head, use maximum resolution inside the segmented
whale for scars/contours/barnacles, handle several whales per photo/burst):
- cluster/parts.py --instances: whale instances per photo (mask components; a component joins an instance whose canonical
  body coverage it does not duplicate, so a separate back + fluke blob stay one whale, two backs are two instances), charts
  at 1536 px from the RAW -> experiments/days_all/<day>/parts_hr (scripts/parts_hr_all.sh, parts_hr_update_multi.sh).
  First version split every component: on 2025-02-27 the oracle ARI fell 0.73 -> 0.52 (a whale's own fluke blob became a
  second, unlabelled instance), hence the coverage rule.
- cluster/tracklets.py: instances linked through a burst (position, size, appearance; Hungarian), cannot-link for two
  instances of one photo with overlapping coverage; cluster/linkage.py: constrained average linkage (reproduces sklearn
  exactly without constraints). 2025-02-27: 106 bursts -> 151 tracklets, 51 cannot-link pairs, tracklet purity 0.93 = burst purity.
- cluster/train_id.py --group-hard (batch identities drawn group by group = the animals swimming together as negatives),
  --init; id_v3 = 36 epochs, 512 px, group-hard (training).
- cluster/dino_hr.py: DINOv3 ViT-B/16 at 1024 px on the instance charts, masked attention pooling (per-part query) ->
  512-d global embedding (ArcFace + supcon, group-hard), last-block tokens kept as PCA-64 local descriptors (extract).
  Queued behind id_v3 (scripts/dino_hr_train.sh).
- cluster/local_hr.py: ALIKED + LightGlue on the 1024-px charts inside the eroded mask, chart-space alignment check;
  DINO-local mutual-NN matching. LightGlue pair AUC on 2025-02-27: fin 0.644, fluke 0.537, body 0.536 (weak, 40 ms/pair).
- cluster/cluster_day.py (channels -> fusion -> tracklets -> constrained linkage), cluster/fuse.py (per-channel
  calibration + logistic fusion with missing indicators, fitted on the val days), cluster/tune_v3.py (time weight,
  threshold offset, group rule on val). Self-calibrated threshold: the +0.75 offset over-splits 2025-02-27 (0.55 vs
  oracle 0.73 at offset ~-0.3); offset becomes a tuned parameter (val days).
Cannot-link constraints (2025-02-27, id_v2 channels, threshold offset 0): coverage rule (16 pairs) ARI 0.568 / oracle 0.659,
dorsal-fin rule (3 pairs) 0.610 / 0.743, none 0.663 / 0.778. Visual check of the 16 coverage pairs: 11 are one whale's two
pectoral fins during a slap or a fluke + its peduncle (the map's part/side labels on small blobs are unreliable); the 3
fin pairs are right but sit in an impure tracklet (a CG burst where consecutive photos show three animals, linked by the
tracklet rule), so the constraint splits the wrong things. Conclusion: hard constraints need pure tracklets first; the
tracklet link cost is tuned on the val days next, constraints default to "fin" only between graph nodes.
Tracklet link cost (2025-02-27, id_v2 body embedding as appearance): cost <= 1.0 gives 265 tracklets with every multi-photo
tracklet pure (bursts: 135 nodes, purity 0.93); cost <= 1.5: 0.97. Default max_link_cost 1.0.
fusion_A (cluster/fuse.py on the val days; channels id_v2 body/fin/fluke + MegaDescriptor body/fin/fluke on the hi-res
instance charts): fused same-day pair AUC val 0.818 / 0.878 / 0.845 (best single channel 0.823 / 0.877 / 0.867: the fusion
adds ~0.01 over id_v2 body; MegaDescriptor weights ~0 or negative); eval days for the record 0.744 / 0.796 / 0.753 / 0.748.
id_v3 (train_id.py --group-hard, 512 px, planned 36 epochs): val same-day pair AUC peaked at epoch 3 (mean 0.864: body 0.835,
fin 0.848, fluke 0.910; id_v2 best 0.852) and then declined steadily (0.80 by epoch 19), so the run was stopped by hand after
epoch 19 and best.pt (epoch 3) kept. Group-hard negatives help early and over-fit with the long schedule at this learning
rate; a shorter schedule (8-10 epochs) is the recipe to keep.
id_v3 like for like (old cluster_bursts pipeline, 512 charts, tuned defaults), animal ARI: eval 2024-12-18 0.569, 2024-12-26
0.288, 2025-02-19 0.320, 2025-03-21 0.334 (mean 0.378; id_v2 0.342); val 0.481 / 0.636 / 0.427 (mean 0.515; id_v2 0.498).
Better on every day: the group-hard negatives (best checkpoint, epoch 3) are the first clear gain of the follow-ups.
v3 pipeline rule variants (id_v2 channels only, burst-pair self-calibration, offset 0; val mean ARI / eval mean ARI):
bursts, no constraints 0.503 / 0.349; tracklets (cost 1.0) 0.474 / 0.342; + fin cannot-link 0.453 / 0.336; cost 1.5 + fin
0.474 / 0.339; secondaries ungated 0.452 / 0.336; main instances only 0.438 / 0.346. Oracles all 0.51-0.56 val, 0.37-0.39
eval. The instance / tracklet machinery costs ~0.03 val ARI at this threshold rule and nothing on eval; the catalog labels
one animal per photo, so correct secondary instances cannot score. Kept as an option (defaults decided with the fusion).
fusion_A clustering (fin CL, cost 1.0): calibrated eval 0.328 / val 0.464; tune_v3 best by val (tw 2, offset -0.5) val 0.533
/ eval 0.318; fixed log-odds threshold -2 with tw 1.5: val 0.497 / eval 0.344. Groups by interleaving, window 120 s and
>= 20 photo pairs: val ARI 0.756 / eval 0.525 (old rule 30 s / 10 pairs: 0.669 / 0.496): new group defaults.
id_v3 channels added (cluster_v3). fusion_B (id_v2 + id_v3 + md, val fit): fused pair AUC val 0.827 / 0.895 / 0.847 (id_v3
alone 0.798 / 0.878 / 0.826: the two identity runs ensemble). Clustering (v3 pipeline, burst nodes, offset 0), mean ARI:
id_v3 channel alone eval 0.379 / val 0.549 (per eval day 0.609 / 0.314 / 0.289 / 0.303); fusion_B eval 0.333 / val 0.518;
fusion_B with tracklets (cost 1.5, fin cannot-link) eval 0.340 / val 0.476 (tuned tw 1.5 off -0.5: val 0.544 / eval 0.356).
The fusion's missing-channel indicators (+0.39 per missing fin channel, -0.18 per missing fluke) shift body-only pairs as
a block, which average linkage cannot absorb: fusion_C is refit without indicators. Groups (bursts, fusion_B): window 120 s,
>= 2 pairs: val 0.689 / eval 0.614.
fusion_C (no missing indicators) clustering, bursts, offset 0: eval 0.319 / val 0.541; tuned on val (tw 0.5, fixed 0) val 0.637
/ eval 0.303. The learned fusions (B, C) lose to the single id_v3 channel on the eval days (0.379) although their pair AUC
is higher: the fusion is fitted on the val days (its val clustering is in-sample), a weighted SUM of calibrated channels
dilutes a strong fluke or fin match that the best-over-parts rule keeps, and the per-source weights (md|fin negative) are
tuned to three days. The eval-day oracle also drops (2024-12-18: 0.625 -> 0.495), so this is the scorer, not the threshold.
Next: the plain ensemble (mean z of id_v2 + id_v3 per part, best over parts) without learned weights.
Plain ensemble (mean z of id_v2 + id_v3 per part, best over parts; bursts, offset 0): eval 0.353 / val 0.543, below id_v3
alone (0.379 / 0.549). Current best day clustering = v3 pipeline + id_v3 channel + burst nodes + burst-pair calibration:
eval mean ARI 0.379 (2024-12-18 0.609, 2024-12-26 0.314, 2025-02-19 0.289, 2025-03-21 0.303), val 0.549; groups by
interleaving 120 s / >= 20 pairs: val 0.79, eval 0.62.
LightGlue fin channel (ALIKED 1024 kp on the 1024-px fin charts inside the eroded mask, aligned matches, top-20 candidate
pairs by the id_v3 + id_v2 global score): pair AUC within the candidate pairs val 0.55 / 0.63 / 0.58, eval 0.57 / 0.61 / 0.54
/ 0.63 (id_v3 fin on the same days 0.73-0.90). Keypoint matching adds little on this imagery (wet, low-contrast skin, water
highlights dominate the detections); kept as a channel but not used in the clustering.
dino_hr (1024 px, 2 blocks tuned, group-hard P6 x K3): val pair AUC mean epoch 0 0.699, epoch 1 0.781, epoch 2 0.826
(id_v2 at 448 px reached 0.852 after 15 epochs).
id_v4 (id_v3 recipe on a 10-epoch cosine schedule): val pair AUC best 0.875 at epoch 3 (body 0.850, fin 0.860, fluke 0.915),
0.864 at the end: the short schedule holds the group-hard gain (id_v3 peaked at 0.864 and decayed). Clustering (v3, bursts,
offset 0): eval 0.373 (0.609 / 0.297 / 0.295 / 0.293), val 0.536 (0.423 / 0.807 / 0.378): the same as id_v3 (0.379 / 0.549);
eval oracles 0.30-0.36 on the three big days. Pair AUC above ~0.86 on val no longer moves the day ARI: the big days are
limited by within-group look-alikes, which this family of global part embeddings does not separate.
dino_hr (ViT-B/16 at 1024 px, last 2 blocks tuned, masked attention pooling, P6 x K3 group-hard, ~6.5 min/epoch alone):
val pair AUC mean by epoch 0.699, 0.781, 0.826, 0.844, 0.835, 0.856, 0.853, 0.849, 0.851, 0.856 (epoch 9: body 0.841, fin
0.844, fluke 0.882), then 0.843, 0.839: stopped by hand after epoch 11, best.pt = epoch 9. On par with id_v2 (0.852) and
below id_v4 (0.875) at 2.3x the resolution: the global embedding does not gain from the extra pixels; the local tokens
(hrlocal) are the remaining test of the hi-res idea.
Hi-res DINO channels (dino_hr best.pt, epoch 9): global pair AUC val 0.77-0.93 per part (fluke 0.877 / 0.868 / 0.933, above
id_v4's fluke on two of three val days), eval 0.67-0.86; hrlocal (mutual-NN tokens on the top-30 candidate pairs) 0.52-0.80
within the candidates (fluke 0.80 on 2025-03-14, otherwise ~0.6). Clustering (bursts, offset 0): hi-res global alone eval
0.373 / val 0.538 (eval oracle 0.438, the highest), id_v4 + hi-res mean z 0.364 / 0.541. Final seven-day table
(scripts/v3_summary.py): eval mean ARI old whole-crop MD 0.298, id_v2 0.342, id_v3 0.378-0.379, id_v4 0.373, hi-res DINO
0.373; val 0.465 / 0.498 / 0.515-0.549 / 0.536 / 0.538. Three independent scorers (448-px identity, 1024-px identity, their
group-hard variants) plateau at eval ARI ~0.37-0.38 with per-day oracles 0.30-0.36 on the three big days: the remaining
error is within-group look-alikes that global part embeddings at either resolution do not separate; the learned fusion
over all 19 channels has unstable weights (id_v3 body -0.9, id_v4 body +1.2: collinear channels) and is not used.
Defaults set in cluster_day.py: thr-offset 0.0, max-link-cost 1.5, cannot-link fin, group window 120 s / 10 pairs.

## 2026-10-19 afternoon — signatures: the most distinctive regions per whale (qualitative first, user request)

cluster/signature.py: per whale token (hi-res DINO tokens, PCA-64, 64 x 64 grid of the 1024-px instance chart) a
distinctiveness d = repeatability (best match in the burst's other charts at the same chart position) - commonness (the
5th-best other tracklet's match at that position: rarity across the day's whales); per tracklet and part the top-24
tokens with spacing = the signature; links = mean of the best 8 aligned signature matches, symmetrised. Edge shapes
(cluster/shape_sig.py): fin outline (finshape.outline_descriptor on the isolated silhouette) and fluke trailing edge
(lower boundary over the span, residual from a quadratic sweep), per-point robust z against the day's population =
uniqueness; links = uniqueness-weighted outline distance. Web app: Whale-Review-Web /signatures (backend/signatures.py,
frontend/signatures.html + js/signatures.js): tracklet list, charts with the distinctiveness heat (one colour scale per
day), signature points, edge outlines coloured by uniqueness, best links with point correspondences drawn across the
pair, catalog label shown as a check only.
2025-02-27 (burst nodes, 135 tracklets), tracklet-pair AUC: hi-res global 0.840; token signature 0.689 (fine-tuned tokens),
0.662 frozen last block, 0.536 frozen block 8; per-chart centred tokens 0.645 (hr) / 0.622 (frozen) but mark-like maps
instead of a whole-animal blob; fin-shape 0.494, fluke-shape 0.464 (chance: outlines are mostly right on fins, but the
descriptor is dominated by the waterline / base corners and partial flukes). Single shots now get the day's typical
repeatability instead of 1.0. Links are ranked by z(global) + z(signature); the signature points are the explanation.
Canonical-coordinate token matching (signature.py --canon: token centre -> full-res px -> crop -> CSE map; mirror / flip of
the charts applied; 95 % of whale tokens mapped): tracklet-pair link AUC 0.506 at 0.5 m, i.e. chance, against 0.645 with
chart-position alignment on the same tokens. The CSE map's real-photo precision (mirror geodesic disagreement ~1.8 m) is
far below the 0.5 m a token match needs; the pose-normalised charts align parts better than the map does. Chart-position
matching stays the default; --canon waits for a more precise map (clicked real points, run05).
Native-resolution tokens (parts_xr: fin / fluke charts at 3072 px = native for most photos; cluster/local_feats.py: 1024-px
tiles, stride 768, whale tokens only, 16 chart px per token = 4x finer than the 1024-px pass): 2025-02-27 705 charts, 1.67 M
whale tokens, 437 s. Token-signature link AUC with the dino_hr backbone tokens (centred): 0.561 vs 0.645 at 1024 px: finer
tokens from the identity-trained backbone do not describe marks better; the trained local descriptor (train_local.py) is
the test that matters. signature.distinctiveness restructured per tracklet (one matmul per chart x tracklet) for the
10x larger token banks.
cluster/train_local.py (user request: ML local descriptor from cross-burst positive pairs with same-region negatives of
other animals): DINOv3 ViT-B/16 last 2 blocks + token head -> 128-d; batches of P animals x 2 charts from different bursts
(one part per batch, one day at a time), 1024-px crops at native scale (fin / fluke from parts_xr 3072 px, body from
parts_hr 1536 px), shared crop window + 48-px jitter, +-5 deg, scale 0.9-1.1, photometric jitter, blur, noise, no flips;
InfoNCE per anchor token: positives = the partner chart's whale tokens within 3 % of the chart at the same position that
agree in CSE canonical coordinates within 1 m (the sanity check), negatives = all other animals' tokens (incl. the same
region) + the same animal's tokens beyond 12 % of the chart; val = chart-pair AUC (mean over anchors of the best aligned
cosine). Smoke test: 1.6 s/step at P=4; the untrained head already gives fin 0.85 / fluke 0.76 / body 0.54 on 52 pairs.
local_v1 (8 epochs x 300 steps, P=6) queued behind parts_xr (scripts/local_v1_chain.sh), then native-resolution
descriptors + signatures for the 7 days (web app variant signature_local_v1).
local_v1 (train_local.py, 8 epochs x 300 steps, P=6, 0.84 s/step; first launch died on a data-loader cache race, relaunched
with an atomic cache write): its own validation (val days, chart pairs same animal / other burst vs other animals of the
day, score = mean over whale tokens of the best cosine within 3 % of the chart) reaches body 0.98, fin 0.98, fluke 0.97
AUC by epoch 7 (0.91 / 0.98 / 0.98 after one epoch). The signature pipeline on the same descriptors (native-resolution
tokens, 7 days) gives link AUC 0.51-0.64 (hr tokens 0.49-0.62): the signature matched with the loose chart radius
(fin / fluke 30 %, body 50 % of the chart) and only the 24 "distinctive" tokens, i.e. not the way the descriptor was
trained. Testing the 3 % radius and a dense all-token link score (the training metric) at day level.
local_v1 diagnosis: the model's validation metric on crops is 0.97-1.0, the same metric from the stored full-chart tokens
0.51-0.60, and the day-level dense link 0.59 at the training radius. Cause: the sampler gave each animal pair its own crop
window (positives shared it, negatives did not), so the position information in the ViT tokens separated the animals by
crop geometry instead of by marks (a leak in both the loss and the validation). Fix (local_v2): one window per batch for
every chart, 3 % jitter per chart, positive radius 5 %; the chain's signature step now matches at the training radius
(--align-scale 0.17) and ranks by the dense all-token score.
local_v2 (shared crop window per batch, 3 % jitter, r_pos 5 %, 8 epochs, 40 min): honest validation pair AUC rises 0.60 ->
0.66 (best epoch 5: body 0.71, fin 0.67, fluke 0.61; hard-negative accuracy, true counterpart vs the same region of another
animal, only 0.14-0.18). Day-level signature-only link AUC with its native-resolution descriptors (radius 5 %): 2024-12-18
0.618 (hr tokens 0.553), 2025-02-19 0.555 (0.492), 2025-01-15 0.701 (0.576), 2025-02-27 0.708 (0.583), 2025-03-14 0.659
(0.616); the dense all-token score 0.67-0.70; the hi-res global embedding stays ahead (0.68-0.84). The trained descriptor
is the first thing that moved the signature link (+0.04 to +0.13), but at the token level "same region of another whale"
is still nearly as close as the true counterpart: 16-px tokens of grey skin carry little mark information; marks are sparse.
local_v2 day-level complete: 2024-12-26 signature-only link AUC (next line from the chain log) added to the six days above;
the trained descriptor beats the hi-res tokens on every day, the global embedding stays ahead on every day.
  tracklet-pair AUC: signature 0.641 (16110)  hr_global 0.764 (16110)  combined 0.655 (16110)  dense 0.655 (16110)  shap
local_v3 launched (conv 3x3 context head, anchors drawn and weighted by local contrast, 4 blocks tuned, 16 epochs, validation
and dense scores on the top 20 % of anchor matches); the Signatures page got a click-to-mark tool (data/<day>/marks.json via
/api/sig/<day>/marks) so researchers can record the regions they identify by: future positives and the only direct check.
cluster/edge_match.py written: gradient-snapped trailing edges (fin: tip -> base on the trailing side; fluke: lower boundary),
chord-frame deviation + multi-scale curvature, banded DTW; evaluation = pair AUC on labelled charts with a 'good edges' subset.
Edge matcher (edge_match.py) on 2025-02-27: fin trailing edges at chance for every descriptor and even for same-burst pairs
(consecutive frames of one whale: AUC 0.48-0.59; tip-anchored variant 0.48-0.53), i.e. the extracted fin edge is not stable
from frame to frame; cutting the blob with the body map's fin band and a sharpness filter change nothing. Fluke lower edges:
same-burst 0.69, cross-burst 0.63 (deviation profile), weak. Conclusion: the shape channel needs its own redesign (a fin /
fluke edge detector at native resolution with stable endpoints, finFindR / CurvRank style), not another descriptor on top
of the segmentation mask.
Encounter-neutral time term (saved id_v4 day scores; bursts chained within 120 s = an encounter; time term 0 inside it):
2024-12-26 0.297 -> 0.343/0.361 (the CG day: fewer wrong merges), 2024-12-18 0.609 -> 0.507 (fragmentation), eval mean
0.373 -> 0.355, val 0.536 -> 0.542; a mild push apart (-0.25) inside encounters: eval 0.312 / val 0.569. The time pull is
right for lone animals and wrong inside competitive groups; a rule that switches it off only where an encounter shows
several animals (two instances in a photo, two dorsal fins) is the candidate, not a global change.
local_v3 (conv 3x3 context head, anchors drawn and weighted by local contrast, 4 blocks tuned, 16 epochs, 97 min): honest
validation pair AUC (top-20 % anchor matches) 0.742 at epoch 10 (body 0.72, fin 0.79, fluke 0.71; all-anchor mean 0.741),
local_v2 0.661: the context and the mark-weighted anchors are worth +0.08. Mark-anchor hard-negative accuracy 0.08 -> 0.24
during training. Day-level signatures with these descriptors follow (signature_local_v3).
local_v3 day-level (signature_local_v3, 7 days): see scripts/signature_summary.py; on 2025-02-27 signature 0.694 / dense 0.693
(local_v2 0.708 / 0.694): the +0.08 of the descriptor's own validation (fin-box crops) does not transfer to whole-chart,
all-part, all-pose day links. Assessment delivered to the user 2026-10-19 21:40: keep the identity backbone, charts, day
pipeline and web tool; redesign (1) the mark signal with researcher clicks (tool in place), (2) the shape channel with a
dedicated edge detector trained from the synthetic rig, (3) within-group clustering as an assignment per encounter.

## 2026-10-19 night — pair verifier (teacher) and distilled embedding (student), user question

Every labelled-pair model so far was an embedding scored by a cosine (id_v2-v4 ArcFace + supcon, dino_hr, the token
descriptor). cluster/pairnet.py adds the model the user asked about: a cross-attention pair verifier that sees both
charts (DINOv3 tokens at 512 px -> 256-d + Fourier chart-position + part token -> 3 layers self/cross attention, whale
tokens only, symmetric pooled head -> P(same)), trained with BCE on all pairs of batches of 6 animals x 2 charts of one
part from one day (cross-burst positives, same-day negatives); and a student (WhaleIdModel, id_v4 init) trained to
reproduce the teacher's P(same) through sigmoid(a cos + b) on the same pairs (relational distillation, 0.3 label mix),
i.e. the teacher's pairwise judgement forced into an aligned per-individual embedding. Evaluation on identical pairs
(cross-burst positives vs same-day negatives, per part) for teacher, student and id_v4, val days for selection and eval
days for the record; the student then runs as a day-clustering channel (scripts/pairnet_chain.sh, overnight).
pair_teacher v1 (matcher from scratch, then from the id_v4 backbone): val pair AUC 0.45-0.59 over 4 epochs, loss at the
chance level of the balanced BCE (1.26), while the same model overfits one batch to loss 0.001 in 30 steps: the
cross-attention matcher learns nothing that transfers from ~20k pairs per epoch (LightGlue-type matchers train on
millions). Rebuilt as a residual verifier: logit = scale * cos(id_v4 embeddings) + bias + cross-attention correction
(last layer zero-initialised), so it starts at id_v4's decision (0.84 on these pairs) and the question becomes whether
seeing both charts adds to the embedding.
pairnet results on identical cross-burst pairs (same-day negatives, 80 batches): val teacher 0.858 / student 0.865 / id_v4
0.848 (5010 pairs); eval teacher 0.720 / student 0.740 / id_v4 0.719 (5259 pairs). The residual verifier adds ~0.01 over
the embedding it starts from (seeing both charts buys little at this data size); the distilled student (id_v4 init, trained
only to reproduce the teacher's pair probabilities through sigmoid(a cos + b), label mix 0.3) ends ABOVE both its teacher
and id_v4 (+0.017 val, +0.021 eval): the soft pairwise targets act as a better-regularised pair objective than ArcFace +
supcon. Day-level clustering with the student as the channel follows (_student).
Distilled student as the day-clustering channel (bursts, offset 0): eval 0.358 (2024-12-18 0.623 = best so far on that day,
2024-12-26 0.300 = its oracle, 2025-02-19 0.243, 2025-03-21 0.264), val 0.571 (best val so far; id_v4 0.536, id_v3 0.549).
Pair AUC +0.02 does not move the eval mean: the big days sit at their oracle thresholds, so the limit is the ranking of
within-group look-alikes, which all embedding variants share.

## 2026-10-20 — failure analysis and the within-group problem (user: find the failing areas and target them)

cluster/failure_analysis.py (student scorer, 7 days): 63-100 % of wrong links join two animals of one true group (98-100 %
on the big CG days); if every predicted cluster were split correctly by animal, eval ARI would be 0.53-0.68 instead of
0.24-0.62. Appearance AUC across groups 0.71-0.88, within groups 0.57-0.75 (within Pairs 0.53-0.63, CG 0.61-0.78). Hard
photos (top-3 neighbours all other animals) are 20-42 % of the big days' photos, not smaller or worse framed than the
rest, 50 % of their best wrong neighbour is a group mate. Bound with true groups + appearance-only inside: eval 0.35-0.79.
Within-group cross-burst AUC per channel (mean of 7 days): student fin 0.73, hr fluke 0.74, student body 0.68, id_v4 fin
0.71, local/LightGlue 0.56-0.59. Multi-instance photos are 5-13 % of scored photos; scoring single-instance photos only
changes ARI by < 0.01, so the one-label-per-photo issue is minor.
cluster/two_stage.py (encounters by time -> appearance-only linkage inside, fins/flukes weighted over bodies -> strict
cross-encounter merges), tuned on val: theta_in 1.0, theta_out 2.0, encounter 300 s: val 0.605 (best so far; single-stage
student 0.571), eval 0.345 (student 0.358, id_v3 0.379); the second-best setting val 0.574 / eval 0.369. The structure of
the clustering is worth ~+0.03 on val and nothing on eval: the limit is within-group appearance. Next: native-resolution
fin / fluke specialist embedding selected on within-group AUC (train_id.py --parts-tag parts_xr --center-crop 1536 --size 1024).
fin_xr (native-resolution fin / fluke specialist from DINOv3, 4 blocks, group-hard, 1024 px): within-group val AUC 0.653 ->
0.632 over 2 epochs (same-day 0.74-0.76) against the student's 0.775 (fin 0.772, fluke 0.779) and id_v4's 0.758 on the
same protocol: stopped. Resolution alone does not buy within-group separation for an embedding; retrying as a fine-tune
of the student at native resolution (fin_xr2, init pair_student, backbone lr 1e-5, selected on within-group AUC).
Shape-only charts (cluster/sil_charts.py -> parts_sil: R = fin silhouette / fluke mask, G = whale mask, B = fin band, box of
the silhouette + 35 %, 512 px) for all split days; sil_v1 = train_id on them (fin, fluke; group-hard; selected on the
within-group AUC) queued after fin_xr2 (scripts/sil_chain.sh), followed by its channel (idsil:), the two-stage run with
shape fins / flukes and the single-stage student + shape fusion. Tests whether fin / fluke shape alone carries within-group
identity that the texture embeddings miss.
fin_xr2 (student fine-tuned at native resolution on fin / fluke, 8 epochs, selected on within-group AUC): val within-group
0.771 best (student 0.775 on the same protocol); day-level within-group AUC fin 0.694 mean vs student 0.730, fluke 0.668 vs
0.670. Single-stage student + specialist fusion: eval 0.367 / val 0.594 (student alone 0.358 / 0.571, noise); two-stage with
specialist fins: val 0.555 / eval 0.30-0.36. Third negative result for resolution: dino_hr (1024 on 1536), fin_xr (scratch,
native), fin_xr2 (fine-tune, native) all fail to separate group mates better than the 512-px embedding.
sil_v1 (shape-only embedding on the segmentation silhouettes): within-group val AUC 0.52-0.58 over 5 epochs and falling:
stopped. The silhouettes themselves are the problem: the segmentation runs at 1280 px, so a fin edge at 3072-px chart
scale is a staircase. Re-running seg_r04_ft4 on the native-resolution fin crop (box of the silhouette x 2, max side 1024)
gives much tighter contours (IoU with the old mask 0.83-0.98 on clean fins, the difference being the old mask's coarseness;
fails on a 600-px blurry fin and on spray). This is the input the shape channel and the edge matcher needed.
Edge matcher on the refined (native re-segmented) masks, 2025-02-27: fin same-burst AUC 0.58-0.59, cross-burst 0.52-0.54;
fluke same-burst 0.65-0.69, cross-burst 0.55-0.61. Mask precision was not the limit: the tip -> base path and its chord frame
change between consecutive frames (the base is wherever the morphological opening cuts), so the descriptor is unstable
regardless of the edge quality. Closing the shape line for this session; a shape channel needs its own detector with
stable anchors (tip, fin-back junction, notch) - a separate project. Refined masks kept (cluster/reseg.py) for it.
Group-hard pair chain (batches filled with one catalog group; validation on within-group pairs): teacher = id_v4 exactly
(val 0.844 / within-group 0.791 both; eval 0.690 / 0.663 vs 0.690 / 0.662); student 0.828 / 0.758 val, 0.688 / 0.652 eval,
i.e. below the mixed-day student (0.864 / 0.740 on the mixed protocol) and below id_v4 within groups. Day clustering with it:
eval 0.376 / val 0.542 (the plateau). Native re-segmentation of all fin / fluke charts finished (RESEG_ALLDONE).
Round summary: within-group confusion is the failure; two-stage clustering, three resolution variants, two shape variants,
refined masks, pairwise supervision and group-hard distillation all leave within-group pair AUC at 0.65-0.79 and eval ARI at
0.35-0.38. The 512-px texture / outline evidence is exhausted; the input has to change (researcher marks, an edge
detector with stable landmarks, the run05 renders for the body map).

## 2026-10-20 afternoon — researcher pair labels, run05 individual variance, fin / fluke landmark detector (user request)
Pairs page (Whale-Review-Web /pairs, backend/pairs.py, frontend/pairs.html + js/pairs.js): the catalog-labelled, clustered
days stay as they are; a researcher sees two photos of the main instance from different bursts, zooms / pans each, clicks
the region in A and then in B that decided the call (numbered crosshairs, photo fractions), ticks the cue (dorsal shape,
fluke shape, barnacles, scars, pigmentation, size, other) and presses Same (S) / Different (D) / Can't tell (U) / Skip (N);
the next pair loads at once. Queues: "same" = catalog same-animal cross-burst pairs the student scored lowest (misses),
"different" = same-group other-animal pairs it scored highest (confusions), "mixed" alternates. Records go to
Whale-Review-Web/data/<day>/pair_marks.json (never touched by the matcher). Verified with Playwright (docs/pairs_page.png).
Factory (Whale-Image-Data-Factory) for run05: individuals.py widened (adult dorsal height 0.5-2.0, sweep -25..40 cm, new
trailing-edge wear dorsal_edge_cm 0-10 with 0-3 nicks; fluke span 0.85-1.2, sweep -30..40, notch 0-30, edge wear 0-20 with
0-6 nicks, tip loss 15-60 cm on one side with p 0.15), skin.py variance widened (darkness 0.55-1.5, mottling, 0-8 rakes,
0-23 round scars, 0-11 barnacle clusters). New library: 36 adults + 8 calves, seed 5 (old 12 + 4 kept as
individuals_run04.json / skins_run04/). project.edge_chains: fin trailing edge fin_te_00-11, leading edge fin_le_00-11
(height bins above the back line) and fluke trailing edge fluke_te_00-23 (span bins) as extra keypoints with uv / state in
every meta from now on (48 chain points on the canonical rig; run04 metas do not have them). shotlist.py gained
individual_cycle and pose_cycle (review renders: every adult in turn x surface / back_arch / fluke_up).
Review render r05_individuals (config run05_individuals_test.json: 1920x1080, one frame per shot, 108 shots = 36 adults x 3
poses, close framing, deep_blue look, calm / light seas, 10 batches, MRQ_Batch_AA2) running; tools/individuals_sheet.py
draws keypoints + chains per individual for the user's review before the run05 full render (still needs the go-ahead).
Landmark detector (model_fine_tuning/landmarks/parts.py, new): one DINOv3 ViT-B/16 + dense decoder for fin and fluke crops
(window around the part's landmarks, 1.4-3.0 x extent, capped at 1.15 x the frame): heatmaps for 56 landmarks (fin tip /
front / back + 24 fin chain points, fluke tips / notch / tail base + 24 fluke chain points; chain channels are "missing"
on run04 and carry no loss), a 4-class state per landmark, and full-resolution own-silhouette / seam / mask maps from the
factory contour label (BCE + dice, pos_weight 10). run04 index: train 4226 part crops (1379 fin / 2847 fluke), val 896.
Queued behind the render on the GPU (scripts/part_landmarks_chain.sh: postprocess + review sheet, then parts_v1 8 epochs,
val sheet docs/part_landmarks_parts_v1_val.png, predictions on 2025-02-27 and 2025-01-15 into days_all/<day>/part_landmarks_parts_v1/).
parts_v1 result (run04, 8 epochs, ~3 min/epoch): val PCK@0.05 of the part extent fin 0.376 / fluke 0.458 (PCK@0.1 0.544 /
0.566), visible-state F1 0.93 / 0.85, own-silhouette edge F1 (2 px) 0.53 / 0.62, seam edge F1 0.79 / 0.90, mask IoU 0.90 /
0.89; still rising at the last epoch. On real charts (2025-02-27 fins) the predicted own-edge map follows the fin + back
silhouette and the seam map the waterline cleanly (days_all/<day>/part_landmarks_parts_v1/sheet.png); the chain channels
are untrained on run04 (no chain labels) and must be ignored until the run05 fine-tune. Checkpoints
experiments/part_landmarks/parts_v1/{best,last}.pt; val sheet docs/part_landmarks_parts_v1_val.png.
Factory finding while reviewing the fin chains: the 4k asset carries a thin vertical sliver (31 vertices, 10 x 10 x 22 cm)
on the back just ahead of the dorsal fin's leading edge. Inside a full-height fin it is hidden; with the widened
dorsal_height (0.5-2.0) it sticks out of low fins as a tall hooked blade that the landmarks do not describe (tip landmark
up to 130 px below the rendered apex on the review frames). apply_rest now presses it flat onto the back (rest-frame box
rule, clamp to the local back top); the review set is being re-rendered as r05_individuals2. Also: edge_chains for the fin
now walk the fin's side-silhouette envelope from the tip to each junction (stops at the saddle / base line) instead of
height bins over a flank percentile; the frames render at 1280x720 whatever the config says (the MRQ preset fixes it).
Second factory finding (review render 2): the dorsal_height / dorsal_sweep / trailing-edge wear block of
factory/individuals.py::apply_rest never touched the dorsal fin. Its window (joint59 - 60 .. joint56 + 40 in the rig's
rest frame) lies entirely behind the fin (tip at rest x ~88, between joint56 and joint55), so "dorsal_height" scaled the
hump of the tail stock instead: a tall-fin individual rendered as a tall rounded back with the fin on its flank, the
landmarks stayed on the (unchanged) fin and the review frames showed the tip 40-95 px below the silhouette apex. This
also holds for run04 (its dorsal_height 0.8-1.3 varied the hump, never the fin). Fixed: window joint56 - 30 .. joint55 +
65, height measured above the midline ridge line interpolated across the fin (not above the flank), hmax over the fin
window only; the peduncle girth now fades out by joint56 so it no longer inflates the fin. Isolation test (one parameter
at a time, rest frame): dorsal_height 2.0 -> tip +26 cm, junctions unchanged; 0.5 -> -13 cm; sweep 30 cm -> tip -22 cm
along the body; peduncle 1.1 -> fin untouched. Third review render r05_individuals3.
Dorsal fin edits, final form (factory/individuals.py apply_rest + WhaleModel.dorsal_region): the fin membership is read
off the canonical mesh (height above the back-junction -> front-junction base line, 125 vertices, tip 36 cm; vertex
indices are shared with the rig's rest mesh) and the edits act along rest-frame directions ("up" = base midpoint -> tip,
"along" = back -> front junction) because the rig's rest pose is a curled, pitched whale where neither z nor x is the fin
axis. Isolation test: dorsal_height 2.0 -> tip +36 cm along the fin axis, 0.5 -> -18 cm, junctions fixed; sweep 30 cm ->
tip -30 cm along the base line; wear / nicks move the rear side forward along the base line. The previous attempts (flank
percentile base, midline chord, shoulder bins) all failed in the rest frame for that reason.
Review render 3 (fin edits fixed): the fin landmarks and chains now sit on the fin in every checked frame. What remains
is a property of the asset + poses, not of the labels: the asset's dorsal fin (36 cm at height 1.0) rises from a hump
whose top ("dorsal_fin_front", 1 m ahead of the tip) is only ~20 cm lower, so in the surface / back_arch poses (arch apex
near the hump) the silhouette apex is the hump and the fin reads as a step on its rear slope unless dorsal_height is
well above 1. Worth the user's eye on the review pages: whether the low-fin individuals look like real "knob on the hump"
animals or whether the hump itself should be flattened / the arch apex moved aft for run05.
Deliverable: Whale-Image-Data-Factory/docs/run05_review/individuals_sheet_p01..p06.png (r05_individuals3; 108 frames, 36
adults x surface / back_arch / fluke_up, close-ups with the chains). Chain visibility: fluke_te 848 / 864 points visible in
fluke_up frames, fin_te 237-375 and fin_le 318-404 of 432 in surface / back_arch frames. QA flags 10 / 108.

## 2026-10-20 night — the phantom dorsal fin (user: "df_tip is where df_back should be"), spine_end, run05 launch
The user was right. The 4k rig's dorsal fin is the small hooked blade (45 vertices, 22 cm above the back ridge, 20 cm
chord) that I had taken for a mesh artifact and pressed flat. The canonical frame (WhaleModel.register_to: per x-slice
height / width mapping onto the K9239 reference) fabricates a 36 cm peak at the K9239 fin position (0.39 of the body
from the tail) by stretching the 4k back's slice there; project.landmarks picked that phantom as dorsal_fin_tip, so every
dorsal-fin label on the 4k model since run04 was on the smooth back 45 cm behind the real fin, and the fin edits of
individuals.py scaled the phantom region. Fixed: WhaleModel.dorsal_fin() reads the fin off the rig's own rest mesh (the
thin dorsal-midline blade standing > 8 cm above the running-median back ridge), gives tip / back / front junctions and the
two 12-point edge chains by height level (the trailing edge is near-vertical, so height bins, monotone in x), and
landmark_indices() overrides the canonical picks; dorsal_region() feeds the same blade to apply_rest (dorsal_height 2.0
-> tip +22 cm along the fin axis, sweep along the base line, wear / nicks on the blade). The flattening is gone. The
run04 parts_v1 detector was therefore trained on wrong fin-tip labels (its fin PCK is meaningless; the edge maps are fine).
New keypoint spine_end: the dorsal midline point at the narrowest cross-section of the tail stock (0.153 of the body from
the tail end), i.e. where the spine's dorsal ridge ends before the blades spread; with the notch it gives the fluke's
midline axis and its visibility tells the fluke's upper side from its underside (user request). Not yet in the Dinov3
KEYPOINTS list (18 names; models index by name, so the extra meta entry is harmless until a run05 fine-tune adds it).
Verification: 12-shot surface render r05_verify (fin tip vs silhouette apex), then tools/run05_overnight.sh: review render
r05_individuals4 -> docs/run05_review pages, then build_dataset run05 (10 000 shots, seed 6, AA2) with a Zen DDC janitor.
Disk: run03/raw and run04/raw moved to /var/tmp/sat3737-whale-raw/ (symlinks left) to make room on /home.
Fin chains (user: they zigzag): the per-height "rearmost / foremost blade vertex" picked side-face vertices of the 10 cm
thick blade, alternating faces level by level. Now the chains follow the blade's midline edge loop (|y| < 1 cm, the
vertices where the two faces meet, down to the ridge line), resampled every ~2 cm along the mesh edges with quarter-step
samples (index lists -> mean of vertices, so the points lie on the rendered outline). Trailing edge: tip (-135, 146) ->
base (-138.5, 125); leading edge: tip -> front junction (-117, 130). Review pages regenerated from r05_individuals4 with
these chains (docs/run05_review); the run04 label refresh restarted with them; run05's postprocess (end of the build) uses
them. run05 status at 01:55 UTC: 40 / 249 batches rendered (~5.3 min per batch of 50 shots, ETA ~18 h).

## 2026-10-21 — run05 render stopped at the disk floor (214 / 249 batches), finishing + training queued
/home hit the 30 GiB diskguard floor at batch 214 (run05 raw = 26 GB so far; the dataset will need ~27 GB more). Moved
run01, run02 and the three old review renders (~23 GB) to /var/tmp/sat3737-whale-raw/ (symlinks left in place).
tools/run05_finish.sh (detached, log data/renders/run05_finish.log): waits for the move, renders the remaining 35 batches
(~3 h), postprocess (16 workers), factory.dataset, QA sheet, then touches run05/dataset/.finished.
scripts/run05_train.sh (detached, log experiments/run05_train.log) waits for that marker, then: split
(splits/run05.json, min visible 800 px, flagged dropped) -> seg_r05_ft4 (30 ep) -> canon_r05_ft4 (40 ep, the run04
"plain" recipe) -> parts_v2 (landmarks/parts.py on run05 + the refreshed run04; 57 channels incl. spine_end; 10 ep) ->
val sheet + predictions on 2025-02-27 / 2025-01-15. CSE (cse_train.sh) and the real-photo evaluation are not queued.
run04 label refresh done 04:02 UTC: every meta now carries the real fin tip, 48 chain points and spine_end.
run04 re-label, take two: the first refresh (01:53-04:02 UTC) re-posed run04 with the NEW individual library (36 + 8,
different i01-i12) and the new fin edits, so its geometry no longer matched the frames (visible_inside_amodal p10 0.68,
3319 shots flagged visible_outside_amodal). individuals.py now has a legacy mode: WF_INDIVIDUALS=<json> picks the
library (individuals_run04.json) and WF_LEGACY_DORSAL=1 restores the run04-era dorsal block and peduncle fade verbatim.
Test on batch_000: visible_inside_amodal median 1.000 / p10 0.999, projection covered 1.000, real fin tip 2 / 2 / 15 px
from the apex. Full legacy refresh running (log data/renders/run04_label_refresh.log, marker dataset/.refreshed_legacy).
Disk (user-authorised): run01 and run02 deleted; run03 to be moved to /var/tmp/sat3737-whale-raw/run03 (symlink) once
the run05 render has finished (moving it earlier would push the root disk under the engine's 30 GiB floor).
Automatic chain after the render (2026-10-21 21:20 UTC, 244 / 249 batches): run05_finish.sh (postprocess, split, QA,
.finished) -> run05_train.sh (split, seg_r05_ft4, canon_r05_ft4, parts_v2 + val sheet + predictions on 2025-02-27 and
2025-01-15) -> run05_after_train.sh -> run05_evaluate.sh (run05 models on the run05 test split and on the real photos of
2024-12-18, run04 models on the run05 test split as the baseline, summary in experiments/run05_eval_summary.txt).
Not queued (decisions first): CSE retraining on run05, re-running the day pipeline with the new seg / canon models.
run04 legacy re-label finished 20:59 UTC (marker dataset/.refreshed_legacy).

## 2026-10-21 evening — web-ocean assessment for a run06 water upgrade (user question)
NOAA_whales/web-ocean (MIT): stand-alone WebGPU Tessendorf FFT sea (6 cascades 1103 m .. 0.26 m, JONSWAP + Donelan-Banner
+ swell, choppiness), Jacobian whitecaps with drifting / decaying foam maps, LEAN shading, Hillaire sky; no objects,
no interaction. Ported the simulation to numpy (factory/fftocean.py: spectrum, evolution, sampling with displacement
inversion, whitecap history; sea states -> wind/fetch/swell with Hs 0.48 / 0.88 / 1.52 / 2.72 m for calm / light /
moderate / rough, whitecap cover 0.3 % moderate, 1.8 % rough). Matched water-only web-ocean renders beside three run05
frames (headless Chromium WebGPU works here) show what we lack: the irregular multi-scale wind-sea texture (our 16-wave
Gerstner sum reads as corduroy at long lenses) and open-sea whitecaps (our crest foam is off). Integration route for
run06: bake cascades + foam per batch to textures, add a WPO / normal / whitecap layer to a project copy of
Water_Material_Ocean by MaterialEditingLibrary scripting, labels through fftocean.height; splashes and poses untouched
(1-2 days + labels + QA). Whale-water interaction: factory/iwave.py (iWave-style linear waves with the body's waterline as
a moving obstruction) on a run05 pose gives bow wave / wake / foam streak and slap ring waves (prototype calibration);
per-shot 64 m texture into the same material and the same label call. Not a free-surface solver: sheets / droplets stay
with the Niagara splashes. Write-up: Whale-Image-Data-Factory/docs/run06_water/README.md (+ three figures).
Training status at 23:30 UTC: seg_r05_ft4 epoch 7 / 30; run03 moved to the root disk; /home 55 GB free.

## 2026-10-22 — run05 training chain + landmark-driven clustering queued (user: focus on run05, individuals, keypoints -> day clustering)
seg_r05_ft4 done (30 ep, 1.8 h): best val dataset IoU 0.933. canon_r05_ft4 training (40 ep), then parts_v2 (landmark
detector, run05 + re-labelled run04, 57 channels), then run05_evaluate.sh (after-train waiter).
New: cluster/shape_channel.py = fin / fluke SHAPE channel from the detector's landmarks: chain points in a landmark frame
(fin: back -> front junction chord; fluke: tip -> tip span, mirrored match for the underside view), RMS distance over
points visible in both, z-scored into channels.npz as shape:<name>|fin / |fluke; with --labels the cross-burst pair AUC
(all / within catalog group) and a closest-pairs sheet. Synthetic unit test: AUC 1.0, same-animal RMS 0.02 vs 0.46.
scripts/run05_landmarks_days.sh (detached, waits for run05_train "done"): parts_v2 predictions on all 7 days, shape
channels, then cluster_v3 runs _student_shape (id:pair_student + shape), _v4_shape, _shape_only, and the v3_summary table
against _student / _v4id. Val days decide; eval days are only reported. Log experiments/run05_landmarks_days.log.
Water (web-ocean) work parked per the user: docs/run06_water/README.md holds the assessment and prototypes.
