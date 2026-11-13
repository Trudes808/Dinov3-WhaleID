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

## 2026-10-23 — after the restart: run05 results, the landmark shape channel is a null result, GPU driver down
Machine restarted 2026-10-22 ~23:45 UTC into kernel 6.8.0-147; the NVIDIA DKMS module (550.163.01) exists only for
6.8.0-139 / -146, so nvidia-smi fails until `sudo dkms autoinstall && sudo modprobe nvidia` (headers for -147 are
installed) or a reboot into 6.8.0-146. All queued work had finished before the restart:
- canon_r05_ft4 (40 ep, best val 0.747): run05 test surface error median 0.28 m vs the run04 model's 0.46 m on the same
  split; real photos 2024-12-18 outline IoU above water 0.229 vs 0.237 (run04 model): no real-photo gain, as with run04.
- seg_r05_ft4 = seg_r04_ft4 on the run05 test split (dataset IoU 0.933 both): segmentation is saturated.
- parts_v2 (landmark detector, run05 + re-labelled run04, 10 ep): val PCK@0.05 fin 0.63 / fluke 0.61 (v1: 0.38 / 0.46),
  visible-state F1 0.92 / 0.86, own-edge F1 0.59 / 0.64. On real charts the tips / notch / fin tip land sensibly
  (conf 0.3-0.6) but the 24-point edge chains clump or scatter (synthetic -> real gap at chain scale).
- shape:parts_v2 channel (chain-point descriptor): pair AUC 0.50-0.54 on every day, within-group 0.44-0.53 = chance;
  clustering with it is worse (student 0.571 val / 0.358 eval -> student+shape 0.497 / 0.343; shape only 0.415 / 0.336).
Next (CPU only while the GPU is down): cluster/outline_shape.py = the refined native silhouette (reseg) cut at the
detector's anchors (tips / notch; fin tip / junctions): the old edge matcher with stable endpoints.
Zoomed second detector pass (landmarks/parts.py predict --refine: a window of 2.6 x the first pass's landmark extent
around the part, CPU inference ~70 charts/min): real fins now fill the detector input and the outline / tip are sharp
(part_landmarks_parts_v2r). cluster/outline_shape.py = refined native silhouette cut at the detector's anchors (fin: tip
+ junctions, top 70 % of the fin height; fluke: tips through the notch), compared after a per-pair Procrustes similarity
alignment. Val days, fins (config: anchor conf >= 0.05, hidden allowed, anchor within 12 % of the chord):
2025-02-27 89 / 214 fins, cross-burst AUC 0.66, within-group 0.70; 2025-01-15 47 / 104, 0.73 / 0.69; 2025-03-14
151 / 331, 0.62 / 0.66. Same-burst AUC only 0.73-0.77 (extraction jitter, not shape ambiguity). Stricter anchors give
0.72-0.77 within-group at 17-25 % coverage. Flukes: 10-20 usable per day (tips rarely all visible), no signal yet.
Written as channel oshape:parts_v2r|fin / |fluke on all 7 days; clustering with it next.
Upright detection (predict --upright: undo the chart builder's rotation / mirror / vertical flip from the stored M and
flags before the two passes, map landmarks back; user noticed the rotated charts): fins unchanged in AUC (within-group
0.52-0.72, ~45 % coverage), flukes gain coverage (19-127 usable per day vs 10-79) and a weak signal (cross-burst AUC
0.61-0.64 on the two days with many pairs, 2025-02-19 and 2025-03-21). Same-burst AUC 0.6-0.8 everywhere: extraction
jitter still dominates. Channel oshape:parts_v2u written on all 7 days.
Clustering with the canonical-chart outline channel (oshape:parts_v2r) by the default mean-z fusion: student 0.571 val /
0.358 eval -> student+oshape 0.502 / 0.349, oshape only 0.414 / 0.330: the noisy channel dilutes the embedding. Next:
the learned fusion (fuse.py on the val days) with and without oshape:parts_v2u.
Learned fusion (fuse.py on the val days, --no-missing): the student channels alone give fused pair AUC 0.817 / 0.880 /
0.854 (val) and 0.781 / 0.794 / 0.769 / 0.739 (eval); adding oshape:parts_v2u|fin / |fluke leaves every number the same
(weights +0.075 and -0.008; the outline channel's own pair AUC 0.58-0.76 sits far below the student fin channel's
0.77-0.91 and is correlated with it). Conclusion for this round: the fin / fluke outline as extracted from the
detector's anchors + the refined mask does not add to the identity embeddings. The bottleneck is extraction jitter
(same-burst AUC 0.6-0.8), i.e. anchor placement on real photos and the mask edge, not the distinctiveness of the shapes.
Options from here: real-photo landmark clicks (a click page next to the pairs page) to fine-tune parts_v2 on the days;
the detector's own edge map at the zoomed pass instead of the segmentation mask for the outline; landmark-aligned fin
crops for the identity embedding (GPU). Clustering with the calibrated student fusion (_fS) running for the record.
Pairs page rebuilt as the real-photo labelling tool (Whale-Review-Web /pairs, user request): true pairs of a day side by
side (model misses first), named keypoints per photo (10 names, keys 1-0, A/B ticks), paired "region of conclusion"
boxes (R, drag in A then in B, numbered), verdicts same / no obvious visual similarity / different / unsure / skip, next
pair on save. Records in Whale-Review-Web/data/<day>/pair_marks.json with keypoints and regions as photo fractions.
Verified end to end with Playwright (keypoints + region + save, test record removed). Server restarted after the reboot.
Next: an exporter from pair_marks keypoints to detector training labels (photo fractions -> chart pixels through the
parts_xr transform M) and a real-photo fine-tune of parts_v2 once the GPU is back.
Pairs page additions (user): a "same burst" checkbox (record field same_burst) and a compressed day timeline under the
photos (/api/pairs/<day>/timeline: all photos with time / camera / burst / identity; ticks, burst bars per camera lane,
gaps capped at 8 s, A and B markers, same-animal photos green, hover tooltip, A/B capture times and gap in the pair
bar). Playwright-verified (record saved with same_burst, test record removed).
Pairs page, pre-labelling additions: all 28 days with charts + catalog labels are offered (train days without model
scores in random order; split tag in the day list), order selector hardest / random on the scored days, "catalog label
looks wrong" flag, previous-pair re-edit (B; replaces the saved record), keypoint placement tooltips, own photo route
/img/pairs/<day>/photo. Counts: 33 day folders, 11 378 photos with timestamps, 10 564 catalog-labelled, 393 animal-days;
the 7 split days alone: 3 051 photos, 123 animals, 733 bursts, 54 379 true cross-burst pairs, 151 506 confusable pairs.
Playwright-verified (train-day photos load, keypoints, previous/edit replaces, test records removed).
Fluke tips are now anatomical (2026-10-23); label files are empty so nothing to migrate.
2026-10-31 pairs tooling, after the researcher's first 13 labels on 2024-12-18 (all "no similarity"; many hardest true
pairs are fin vs fluke, and the researcher's "same burst" ticks were 40-85 min apart, i.e. "same interaction"):
(1) several whales per photo (+ whale, shift+Q/W; record other_whales), (2) "no keypoints visible" per photo (Q/W;
no_keypoints; every verdict but skip needs keypoints or the flag on both photos), (3) a consecutive-frames queue
(mode=sequential: adjacent photos of one burst in day order, keypoints of an already-labelled photo carried over as
"known"; pair_type sequential), (4) a second time level: interaction = run of photos with no gap over 15 min across
cameras (2024-12-18: 9 interactions over 7.3 h, matching the catalog groups), same_burst + same_interaction checkboxes
pre-ticked from the timestamps, interactions as bands on the timeline, (5) a separate /bursts page for the cross-burst
linkage: two photo sets (bursts or interactions) of one animal (or two animals of one group) with filmstrips, browse
both until a shared mark is found, box it, verdict per pair of sets (pair_type burst_link, level, set_a/b, chosen photos).
Shared labelling engine frontend/js/pairtool.js. Playwright-verified on both pages (sequential save + carry-over,
validation, flags, link save + next unlabelled + previous + level / mode switch); test records removed, the
researcher's 13 labels untouched. Server restarted (0.0.0.0:8765).
2026-10-31 (later) pairs page re-centred on back-to-back shots. The researcher's 52 labels so far were all cross-burst
(default queue "true pairs, hardest first"): 47 no similarity, 5 same with regions, so almost no tied regions. Measured on
2024-12-18 (330 instances, 16 animals): consecutive photos of one animal are 0.2-1.4 s apart inside a burst, 63 chain
pairs are 1.5-60 s re-surfacings (all same camera), 31 gaps exceed 60 s. New default queue mode=near: each animal's photo
chain (day order, all cameras) cut at gaps > NEAR_GAP = 60 s into runs, walked run by run; orders spaced (frames >= 1 s
apart, near-duplicates skipped, runs end on their last frame; 156 pairs), gaps (re-surfacings first), all (283), random.
Records stay pair_type sequential; same_burst / same_interaction pre-ticked per pair (no longer forced). When B ends a
run the pair bar links the two bursts / interactions on /bursts (deep link ?ident&set_a&set_b). /bursts: "next
unlabelled" takes the hard gaps (> 60 s) first; consecutive sets already tied by a back-to-back label count as covered
(covered in /api/pairs/<day>/bursts). Playwright-verified; test records removed, the 52 labels untouched.
2026-10-31 (later) both label pages re-laid out for laptop screens (user: photos "quite small on a small screen"): the
photo panes take the page, everything else a thin strip (shared frontend/css/pairtool.css): one-line pair bar with
ellipsis (the same-burst / same-interaction / catalog-wrong boxes live there), 58 px timeline (T hides it), two-row
tool strip (short keypoint labels + tools; cues + verdicts), keys / instructions only on H. New fit modes in pairtool.js: V zooms both panes to the catalogued whale's bbox
(<= 1.5 screen px per image px), F fits the photo, "auto" zooms to the whale on every load; settings persist in
localStorage. Measured with Playwright: at 1366x768 the two panes are 72 % of the page (80 % with the timeline hidden,
was ~35 %), 80 % at 1920x1080; tool strip 65 px on both pages at every width.
2026-11-01 first use of the researcher's labels as a test (user: "how many do we need to run any tests to see if the
labeling is worth doing?"). 2024-12-18 so far: 183 records (131 back-to-back: 82 same with a tied region, 31 no marking,
1 unsure, 17 skip; 52 cross-burst: 5 same, 47 no similarity), 181 photos with 982 keypoints, 191 region pairs, 2.2 h of
labelling at ~83 records / h. parts_v2 vs the researcher's keypoints (scratchpad eval/landmark_error.py: photo
fraction -> full px -> chart via parts_xr M, mirror / flip, vs part_landmarks_parts_v2; error in the 2560-px frame,
PCK of the labelled part extent): fin 314 visible predictions, median 33 px on a 163 px extent, PCK@0.1 0.27 (synthetic
val 0.63): dorsal_fin_tip median 10 px PCK@0.1 0.64, dorsal_fin_back 35 px / 0.13, dorsal_fin_front 121 px / 0.01
(either a failure or a definition mismatch to check on the sheet); fluke 87 visible, median 61 px on 573 px, PCK@0.1
0.49, left / right tip swapped in 15 of 37 fins-with-both-tips (side ~chance). Verdict vs student pair score:
back-to-back same vs no-marking AUC 0.73 (82 / 31). Conclusion: the real-photo landmark gap is large, a fine-tune
pilot is worth running; it needs labels on train days (none yet): ~300 photos on 2025-01-02 + 2025-03-18 (~4 h),
2024-12-18 stays the measure (eval day, never trained on).
2026-11-01 pilot labelling set up (user: "set me up to label those images ... show progress along top"): pilot train
days 2025-01-02 (982 photos, 317 spaced back-to-back pairs, 96 gaps) and 2025-03-18 (755 photos, 224 spaced pairs,
111 gaps), both without model scores (day order only), listed first on /pairs; target 300 photos with keypoints
(WHALE_PILOT_DAYS / WHALE_PILOT_TARGET). Top-bar progress widget on both pages (/api/pairs/<day>/progress: spaced
pairs done / total, photos with keypoints, gaps linked, pilot count per day). Next, once the pilot count is reached:
fine-tune parts_v2 on the pilot-day labels (hold out a fifth for epoch selection), measure on the 2024-12-18 labels
against PCK@0.1 fin 0.27 / fluke 0.49, after checking the dorsal_fin_front definition on the prediction sheet.
2026-11-02 real-photo fine-tune pilot A (user: "is it possible to try with what we have"). landmarks/parts.py gained
real_rows() / RealPartDataset (the researcher's clicks from Whale-Review-Web pair_marks.json mapped into the parts_xr
charts; only the clicked points are supervised, chains / dense maps / states missing), per-sample state_w / dense_w in
losses(), and train flags --real-days --real-only --real-repeat --real-val-fraction --real-state-weight --days-root.
Run scripts/parts_ft_pilot.sh parts_v2_ft_pilotA --real-only --w-dense 0: init parts_v2 best, 2025-01-02 clicks = 122
part rows (106 fin / 16 fluke; train 87 fin + 11 fluke x4, hold-out 24), 12 epochs, lr head 1e-4 / backbone 1e-5,
3 min on the RTX 4000 Ada. Scored on 2024-12-18 (never trained on; scratchpad eval/landmark_error.py, PCK@0.1 of the
clicked part extent, all predictions): fin 0.25 -> 0.35 plain / 0.40 with --refine (median 37 -> 26 / 19 px; tip 0.59
-> 0.72, back 0.11 -> 0.19 / 0.31, front 0.01 -> 0.10 / 0.15, still 107-160 px off = definition to check); fluke 0.46
-> 0.60 best / 0.65 last (median 70 -> 38 / 28 px; notch 0.45 -> 0.87, spine_end 0.67 -> 0.78, tips ~0.5 with the
left / right side still at chance: 18 of 37 swapped). Internal same-day hold-out fin PCK@0.1 0.75 vs 0.35-0.40 on the
other day: the day-to-day gap dominates, so the second pilot day matters more than finishing the first. Conclusion:
labelling pays off already at a third of the target; flukes are the scarcest (16 labelled).
Predictions: days_all/2024-12-18/part_landmarks_parts_v2_ft_pilotA_{best,last,bestr}; checkpoints
experiments/part_landmarks/parts_v2_ft_pilotA. Not yet done: a mixed real + render fine-tune (dense maps kept), the
fluke side, the dorsal_fin_front definition check on the sheet.
2026-11-02 fin front base assessed (user: "I think we likely have it in differing places") with scratchpad
eval/fin_front.py / fin_front_zoom.py on the 104 fins of 2024-12-18 that have all three clicks: the researcher clicks
the front base where the leading slope flattens into the back (median 154 px from the tip in the 2560 frame) and the
back base at the bottom of the trailing notch (78 px); parts_v2 (run05 rig labels = the fin blade's base from
whalemodel.dorsal_fin(), project.py's +-0.05-0.06 L ridge points only as fallback) puts both bases ~60 px from the tip,
symmetric, i.e. the blade base under the tip. So the "front" miss is a definition difference, not detection. Decision:
the real-photo clicks define the landmark from here (palette help rewritten); the renders should move dorsal_fin_front
to the hump's forward base in the next label pass (walk forward from the tip along the dorsal profile until the slope
flattens). Second finding: the chart builder's head side is wrong for ~21 % of fins (the clicked front is on the chart's
left), and the fine-tuned front lands on the wrong side in 18 % (baseline 12 %): the head direction has to be learned
from the fin shape on real photos. Pairs page: new default order "need" (fluke pairs first, then the least-labelled
animals in steps of 10, then day order; 2025-01-02 has 86 fluke pairs, 25 labelled; 2025-03-18 78, none), fluke
counts in the progress widget and a link to the pilot day with the fewest labels.

2026-11-02 later: the render label fixed for good (user: "make sure the label is fixed on the synthetic data permanently").
The rig's back profile shows why the blade-root label could never match: ahead of the blade the dorsal ridge runs straight
on (level in the straight clip poses, -30 deg in the curled rest pose), there is no hump and so no "slope flattens into
the back" point to find on the mesh. The researcher's clicks (200 photos: 2024-12-18 + 2025-01-02) put the front at a
median 2.2 x |tip - back junction| from the tip (IQR 1.6-2.8, ~80 deg from the tip -> back direction). Factory
`WhaleModel.dorsal_fin()` now returns `front_base`, the dorsal-midline point `DORSAL_FRONT_RATIO` (2.0 in the rest frame
= 2.1 at rest, 2.3 in the straight clip poses) fin heights ahead of the tip (quarter steps along the mesh edges);
`landmark_indices()` uses it for `dorsal_fin_front`. The blade root still ends the `fin_le` chain and anchors the
individual edits, so renders are unchanged. `factory.postprocess --keypoints-only` recomputes keypoints in place (posed
mesh rasterized again for the depth buffer: the saved depth PNG flips edge keypoints' states) and refuses shots whose
other keypoints do not reproduce; dry runs on 400 shots of run05 and run04 (run04 with WF_INDIVIDUALS=individuals_run04
WF_LEGACY_DORSAL=1) reproduced everything but the front bit-for-bit. Full relabel of run05 + run04 via
`tools/relabel_fin_front.sh` (log data/renders/relabel_fin_front.log, previous entries kept in
dataset/keypoint_prev_dorsal_fin_front.json, meta field `keypoint_revision` = 2026-11-02-fin-front). Fin `part_index`
caches deleted (the fin bbox moved). parts_v2's synthetic fin-front PCK is not comparable with later models; the shape
channel's fin frame (back -> front) is ~2x longer. Next: retrain the parts detector on the relabelled runs (+ real clicks).
Outcome check after the relabel (run05, 5773 visible fins): image-plane |front-tip| / |back-tip| median 1.41 over all views,
1.78 on side views (IQR 1.3-2.4) against the real clicks' 2.23 (1.6-2.8); the close-up tall-fin individuals sit lower
(~1.5) because the individual edits raise the blade while the point stays put. Kept ratio 2.0: on the renders it lands
where the leading slope meets the back (the visual rule the researcher applies), see scratchpad finfront/crops_new.png;
WF_DORSAL_FRONT_RATIO can move it if a later comparison says so. Old labels: dataset/keypoint_prev_dorsal_fin_front.json.

## 2026-11-03 — safe clusters: the burst / sequence stage, built to be pure (user: "focus on the burst clustering first")
The researcher's observation (2026-11-02): identity travels through a few decisive part views; most photo pairs of one
animal cannot be linked by looking. The failure analysis agrees (63-100 % of wrong links join two animals of one group).
So the pipeline is split: a stage that clusters the capture sequence with very high purity, then a part-based link stage.
New: cluster/safe_clusters.py (per-instance features: EXIF exposure / focal from the RAW, the canon model's direct
rotation / ppm / head side, Lab colour of whale and water, lit-side asymmetry; candidate pairs; gbt or logistic model;
union-find with one instance per photo per cluster; purity / completeness metrics with the sequence-only oracle) and
cluster/safe_tracklets.py (stage one = tracklets with the student body cosine in the link cost; stage two = gbt on
tracklet pairs: gap, continuity of the last / first frames, max / mean appearance over instance pairs, sizes).
Findings on the way: (1) a photo-pair model is dominated by the time gap (importance 0.30 of 0.87 AUC) and merges
within-burst switches: at any threshold it is no purer than 1.5-s bursts (eval: 85 impure photos at 0.9). (2) The
wrong links at a safe threshold are almost all multi-whale frames: the largest mask alternates between two animals or
two whales surface at the same spot, and the catalog's single label flips (val sheets in the scratchpad). Instance-level
nodes with photo exclusivity and the appearance embedding fix most of it. (3) Across gaps > 1.5 s the same-animal share
of consecutive same-camera pairs is 30-60 % (groups) and the tracklet-pair AUC 0.75-0.88: at P >= 0.98 essentially no
cross-gap link is made (recall 0-6 %); the remaining fragmentation (6.7-6.9 clusters per animal vs the 300-s oracle's
2.0-2.4) is the part-based stage's job, as the researcher described. (4) Stage-one tracklets with the embedding at cost
1.0 are perfectly pure on eval (0 / 2146) but fragment to 13.6 per animal; cost 1.5 (4 impure, 9.8 per animal) plus the
re-link is the better frontier on val. Grid on val (cost 1.0 / 1.5 x skip 3 / 5 x thresholds): cost 1.5, skip 3, P 0.98.
Result (photos in impure clusters / clusters per animal): val 8 of 878 (single-whale photos 4 of 799), 6.9; eval 22 of
2146 (14 of 2002), 6.7. Baselines: 1.5-s bursts val 18 / 5.7, eval 79 / 5.5; plain tracklets val 5 / 8.6, eval 6 / 9.4.
Per day (eval): 2024-12-18 6 impure at 5.8, 2024-12-26 16 at 8.6 (bursts 72 at 6.3), 2025-02-19 0 at 5.8, 2025-03-21 0
at 6.5. Pair AUC of the stage-two model 0.84-0.91 on val / eval. Outputs DAY/safe/{photos,clusters}.json for all 28 split
days (train days in-sample), model experiments/safe_clusters/safe_model.pkl (= track_gbt_c1.5_k3.pkl), sweeps and
per-day reports in experiments/safe_clusters/. The five small excluded days have no parts_hr charts yet, so no clusters.
Next: the link stage on these clusters (fin / fluke verifiers, cross-view part pairs), and the researcher's cluster
linking on the bursts page fed with these clusters.

## 2026-11-03 later — the strict tier (no false positives), the multi-whale audit, and cluster linking for the researcher
User: clusters must be low-false-positive so the researcher can link them and the links become training data for a
cluster linker; the multi-whale scenes are the hard part. The remaining wrong links of the P 0.98 setting were all
frames with two whales (the catalog's single label flips) plus one spurious instance (a person on the boat). Strict
tier = P >= 0.995 AND no end frame with several whale instances: zero wrong stage-two links on all 7 labelled days
(val + eval, 3024 photos); the 4 + 4 photos still in impure clusters are stage-one tracklet assignments inside
multi-whale bursts (cost 1.5; cost 1.0 fragments more and the re-link then makes 1 + 3 errors). Cost: 8.5 (val) /
9.2 (eval) clusters per animal instead of 6.9 / 6.7. Audit on true small clusters (1.5-s bursts holding >= 2 catalog
animals = the multi-whale scenes): val 10 bursts / 77 photos, 7 split into pure clusters; eval 17 / 212, 16 split
pure; train 73 / 594, 65 pure. Every day's safe/clusters.json now carries, per cluster, the catalog animals (labelled
days) and the number of multi-whale photos, plus `suggestions` for the reviewer: stage-two candidates with P >= 0.2
between different clusters and, within 15 min, the 5 closest-looking clusters per cluster by the student embedding
(max cosine over instance pairs, best part), strongest first (300-1800 per day). safe_tracklets: --strict, audit
(report.audit), suggestions; model experiments/safe_clusters/safe_model.pkl at P 0.995, cost 1.5, skip 3.
Whale-Review-Web: /bursts gained level "cluster" (GET /api/pairs/<day>/clusters; one pseudo-animal "clusters"; the
tracked instance boxed; G walks the suggestions then consecutive clusters; catalog hidden unless ticked; records level
cluster with cluster_rev; other same-labels across clusters count as tied). Playwright-verified; server restarted.
Cluster level, usability (2026-11-03, user: "how do I best use this?"): keypoints optional there (validate option
keypointsOptional), Link found needs one region pair; the page keeps a union-find of the saved same-verdicts (+ ties)
and the different-verdicts between those animals, skips settled pairs in the suggestion walk and shows "so far n
clusters, m photos" behind A and B. Playwright-verified (different without keypoints, same with a drawn region pair,
settled pair display; test records removed).
Bug fixed (user, 2026-11-03): on /bursts a region pair drawn on photos (A1, B1) stayed on screen and in the payload when a
filmstrip moved to another photo, so a save could pair boxes with the wrong photo keys. pairtool.js now binds keypoints
to their photo (per-pane store keyed by the instance key) and region pairs to the two photos they were drawn on; other
photos' pairs are hidden and not saved, a waiting box is dropped when its pane changes photo. Playwright-verified
(keypoint chips follow the photo, waiting box survives a B switch, hidden pair blocks the save and returns). The "no
keypoints" complaint on Link found was the pre-change script still loaded in the browser (static files are no-cache).
What the cluster links look like (2026-11-03, researcher's observation: a link is a small patch, dorsal shape / scars /
colour or barnacles on one fluke tip + pigmentation, plus general colour and the surrounding high-contrast marks).
Checked against the 108 "same" cluster links saved so far on 2025-01-02 (scratchpad analysis, read-only): 97 have
exactly one region pair, 10 two, 1 three; keypoints placed on 1 of 108. Median box 4.7 % of the image width (~380 px
at full res), 3.4 % of the instance box area (p25 1.5 %, p75 9.4 %). Cues: dorsal+pigment 22, fluke+pigment 19,
barnacles 14 (+9 with fluke / pigment / scars), scars 12, dorsal+scars 11, none 5. Mapping the 240 boxes into the
parts_hr charts (the chart affine M maps full-res pixels, i.e. 2560-frame x scale_full, into the 1536 chart; 300/300
instance centres land inside): dorsal fin 69 (+11 on its edge), fluke 60, body 58 (+16 edge), 26 far from every
chart of the tracked instance. Confirmed links are all same-camera with median dt 0 s (p75 1 s, max 104 s): the queue
walks suggestions by P, so the easy within-burst joins come first and the cross-gap links that matter for stage 2 are
still ahead. The 26 far boxes: 19 sit inside the OTHER instance of a two-instance frame, and the three frames looked at
are not two whales but one whale split (fluke in two lobes, 184A4979; fluke + pectoral, 184A5507) or a false instance
(the person on the bow tracked as 5-frame cluster 396, 184A5537-41). The researcher's links are right at the whale
level; the record's instance key is the cluster's tracked one, so region boxes must be attributed to the instance
whose bbox holds them when they are used for training. Two-instance frames: 57 on the day, 13 of them joined into one
whale by the links so far (lower bound). Consequences noted for stage 2: a two-scale patch verifier (high-res decisive
patch + context of marks + instance colour) trained on these region pairs; and a fragment / false-instance filter
before stage 1, since those frames are what the strict tier blocks and the researcher now re-joins by hand.
Cluster-link queue reordered (2026-11-03, user: "the labels you need the most first"): Whale-Review-Web
backend/pairs.py `_queue` now orders the suggestions of GET /api/pairs/<day>/clusters for the reviewer instead of
passing the model's order (sequence P, then appearance): tier "across a gap" (clusters further apart than 1.5 s, what
the sequence model cannot resolve and the linker must learn) before "within a burst" (splits at multi-instance
frames, mostly fragments / false instances); each tier built in passes that give every cluster one look before any
gets a second, strongest evidence first (P, else half the appearance cosine). 2025-01-02: 1491 across / 324 within;
the first pass is 179 pairs over 358 of 434 clusters; 18 of the first 100 were already labelled. Each suggestion
carries gap, tier, rank; the header shows "suggestion r/n (tier)", the progress bar counts the across-gap tier, G
again moves on without a label (queue wraps). ?order=model keeps the old order. Playwright-verified on 2025-02-24
(rank 1 across a gap on load, G walks 2, 3, 4; no records written). Server restarted while idle (last save 13 h before).
New verdict `likely_same` on the safe-clusters level (2026-11-03, user: many split pairs have no linkable mark but are
plainly the same whale's run: whole image alike, same background, same pigment, clusters sequential or nearly so).
Whale-Review-Web: VERDICTS + likely_same; /bursts cluster level gets the button "Likely same (by sequence)" (key L,
hidden on the other levels); no region or keypoints needed; it labels the pair (the queue moves on, the progress
widget counts it) but does not join the clusters in the page's graph, so nothing is implied through it. Training
semantics: a soft positive for the sequence / safe-cluster model (the researcher's judgement from the same cues the
model has: time, colour, exposure, position), NOT a mark pair for the stage-2 patch verifier. Playwright-verified
(hidden on the bursts level, L saves a cluster record with verdict likely_same and empty regions / keypoints, prev
reopens it, record deleted). Server restarted between saves (last save 4 min earlier).
Check-in (2026-11-03 midday): 2025-01-02 has 265 cluster records (162 same, 69 no mark, 19 different, 15 likely
same); 136 since the reorder, all across gaps (58 at 1.5-60 s, 29 at 60-300 s, 49 beyond 300 s). 80 same links now
cross a gap > 1.5 s (cues: dorsal shape 57, pigmentation 46, scars 26, fluke shape 12, barnacles 7). The 15 likely-same
pairs sit at gaps of 1.6-4.6 s (one 12.5, one 144) with sequence P 0.41-0.91: the strict tier's cost, as expected.
Same links join 251 of 434 clusters into 89 animals. Researcher's third observation: besides marks in matching places,
the PROFILE (contour) of the fluke or the dorsal fin links clusters across views: fluke front/back and top/bottom,
dorsal fin left/right. Stage 2 therefore gets a contour channel (fluke trailing edge, fin profile) matched with
mirror / flip invariance; the parts_hr charts already hold silhouette, fin_band, fluke masks with view_side, mirrored,
underside, flipped_v flags. The cues dorsal_shape / fluke_shape mark the training pairs for it.
Queue v2 (2026-11-03 evening, user: "reorder in terms of the labels you need"): Whale-Review-Web `_queue` now scores
each across-a-gap suggestion inside a coverage pass by (1) catalog on labelled days: pairs the catalog calls one
animal first (positives with a region pair are what the linker lacks; negatives come free from the catalog), (2) the
pair's need from the chart view flags (`_views`, cached in data/<day>/cluster_views.json: fin view_side < 0 = left,
fluke underside): cross-view pairs (fin left/right, fluke top/underside = the contour channel's training pairs)
before same-view part pairs before body-only, (3) evidence (P, else half the appearance cosine). 2025-01-02: of the
1491 across-gap suggestions 726 are catalog-same; needs: fin left/right 496, body only 490, fin same side 427, fluke
top/underside 238, fluke same face 127, both 35. The next 100 unlabelled pairs in the new order are 90 catalog-same
and 70 cross-view (was: evidence order). ?order=blind ignores the catalog, ?order=model keeps the model order. The
header shows the need after the tier. Playwright-verified on 2025-02-24 (rank 1 "across a gap, fin left/right", G
walks on, nothing written); server restarted while idle (last save 7 h 44 min earlier).

## 2026-11-11 — Photo Labeling page: how the catalogue is made by hand, a link ranker and a boundary model for it (user: step back and reimagine the tool)
User: the pairwise link labelling serves the models; what would serve the researcher who has to turn a day of photos
into a labelled Lightroom catalogue? New page Whale-Review-Web /labeling (README there, "Photo Labeling view"): a day
goes through Encounters -> Animals -> Review -> Lightroom, the safe clusters are the unit that is sorted ("stacks"),
the models order and propose, the researcher decides. Read-only forensics of Test.lrcat first (how the work is done
today; details in that README): files renamed after the cull (71 % of frames kept by the camera numbers), Title = one
animal id per photo, collections day / "Group N type" / "<day>_GnnAnn role", passers-by "<day>_Onn", group numbers from
the field log, Mother A01 / Calf A02 / Escort A03, 16 "same as" notes, best-of sets (Green tail stock / fluke, Yellow
flank / dorsal), fluke shots cropped and exported; the batch Title stamps time the sorting: a CG of 200-300 photos and
7-8 animals 60-90 min, an MCE 15-45 min; a median day is 394 kept photos, 4 groups, 14 animals; CGs hold 51 % of the
photos. Encounters are not clean time blocks (236 runs in 28 days, 44 returns, 5 % of group changes under 7 s apart).
New: cluster/label_assist.py (fit | simulate) writes Whale-Review-Web/backend/labeling_model.json.
Pair model = logistic on [sequence P of the suggestion, student cosine per part (max over instance pairs, with
presence flags), log gap (+ capped at 60 s)], fitted on the 9 592 within-encounter cluster pairs of the val days.
Animal score = its closest stack (max); mean of the top 2-3, max + mean, noisy-or and max + most recent were all
within 0.01 on val and are not used. Stacks of an encounter sorted one by one in capture order, catalog as the
researcher: true animal first / in the top three val 0.843 / 0.987 (leave-one-day-out), eval 0.786 / 0.970 (863
stacks, 91 animals, 2 146 photos); recency alone 0.70 / 0.91, appearance alone 0.73-0.75 / 0.95-0.96.
Runs (the following open stacks whose link to the unit so far is >= tau, no other animal closer, taken along and
shown): eval decisions 863 -> 710 at tau 0.95 (offered stacks right 0.944) -> 532 at tau 0.9 (0.890); the page's own
replay (encounters cut as in the catalog, earlier links as hints): 551 decisions for 2 167 photos (3.9 photos each,
138 per day), first choice right 0.641 and top three 0.946 of the 460 decisions with the animal present, 330 of 367
run stacks right, 2 163 of 2 167 photos end with the catalog's animal. Per decision (after the runs took the easy
continuations) no score threshold gives a trustworthy "sure": the best class (top >= 0.7, margin >= 0.35) is 16-19 %
of decisions and right 0.868 eval / 0.871 val; the rest: first choice 0.494 / 0.605, top three 0.805 / 0.815, 14 %
new animals (their top score overlaps the existing animals': median 0.59 against 0.72). The page therefore pre-selects
but never decides, and says so. 6 decisions per animal against the sequence-only oracle's 2-2.4 units: the cross-gap
linker (stage 2) is what would cut the count.
Boundary model (gaps between sightings = runs of photos <= 120 s apart; logistic on log gap, log displacement from
the RAWs' EXIF GPS, speed): fitted on 446 gaps of the 21 train days, held-out 185 gaps of val + eval AUC 0.829 (the
gap alone 0.769), at P >= 0.35 precision 0.71 / recall 0.57. A proposal only; a time or distance threshold alone
needs 2-4 x more segments than groups for 99 % purity (gap > 300 s: 429 segments for 113 groups), since a followed
group dives for 10 min and drifts 500 m.
Finding on the cluster-link labels: joining the reviewer's "same" links of 2025-01-02 transitively gave 44 piles, 11 of
them mixing catalog animals (one of 60 stacks / 197 photos across 7 animals): 19 of the 266 cluster-level "same"
verdicts (and 1 sequential) join clusters whose catalog animals differ, none flagged catalog_wrong; pair_marks.json
record indices 77, 241, 297, 309, 310, 326, 328, 329, 331, 364, 550, 615, 625, 672, 677, 719, 729, 804, 809, 811
(e.g. 809 = G01A01 / G01A02 of a pair, 9 s apart). Either the catalog or the verdict is wrong there: check them
before the region pairs train the stage-2 verifier (7 % of the positives). On 2024-12-18 and 2025-03-18 every pile was
catalog-pure. The page uses links as hints only (the linked animal first, the linked stack offered in the run).
Same-frame links are no contradiction test: 7 of 12 piles with two instances of one photo were catalog-pure (one
whale split into two instances).
All 28 labelled days through the page's replay: see Whale-Review-Web README (train days are in-sample for the student).
Not done: Lightroom plug-in for the collections (XMP sidecars carry Title / label / keyword only), a cull assistant
(only kept photos exist here), the field log as the source of the encounter cuts, processing a new day from the page.

## 2026-11-11 later — the researcher's cluster links on 2025-01-02: what they did, what they train, where the pipeline stands
User: "Ok I have manually labeled most of those pictures, lets see what we can do with them for training and then
reevaluate where we are at." Read-only analysis: nothing in the pipeline, the review pages or the label files was
changed. Whale-Review-Web data/2025-01-02/pair_marks.json (saved 2026-11-11 15:36) holds 956 records. 842 are cluster
pairs of the bursts page (744 of the 1491 across-gap suggestions and 98 of the 324 within-burst ones): same 266,
no_similarity 381, likely_same 98, different 83, unsure 14. Same links: 184 across a gap > 1.5 s, 129 beyond 60 s, 84
beyond 300 s; 288 region pairs (246 links with one, 18 with two, 2 with three); cues dorsal shape 140, pigmentation 119,
scars 61, fluke shape 59, barnacles 41. 114 are sequential photo pairs of the pairs page (2026-11-01, instance keys, no
level field): 80 same (48 between two safe clusters, all within 60 s; 32 inside one cluster) and 34 no_similarity. By
the record time stamps the 956 records took about 5.1 h of active labelling (sum of the gaps <= 2 min; median 13 s per
cluster pair, 22 s before a same record, 9 s before a no_similarity one). Scripts, per-pair score tables, logs with the
full tables and contact sheets are in experiments/stage2_labels/ (file names below refer to that folder).
Outcome. (1) Stage 1 is unchanged: strict safe clusters, 8.5 clusters per catalog animal on val, no wrong automatic link
on the 7 val and eval days (2026-11-03). (2) The links themselves did the most: 12.83 -> 3.87 pieces per catalog animal
on the day. (3) 20 of the 314 same links contradict the catalog. Joined transitively they put half of the catalog photos
into mixed piles; without them the piles are pure to 0.9 %. They need the researcher before the region pairs train
anything. (4) likely_same marks the interaction, not the animal. (5) One day of verdicts does not train a linker or a
mark verifier that beats the existing scores, in grouped CV on the day or on the val days; the day's catalog labels
train a better linker than the verdicts. (6) The boxes buy a where-to-look model that saturates after about 6 animal
groups and transfers. Matching on its windows, averaged by rank with the student, finds about a third more catalog-same
pairs at precision 0.95 on the val days and about half more at 0.90; no gain on the in-sample labelled day. (7) Crops at
the researcher's boxes reach 0.97 against other boxes but 0.55 against the best-looking patch of another whale:
max-pooled patch similarity does not decide "different", that needs aligned part comparison. This replaces the
"two-scale patch verifier trained on the region pairs" plan of 2026-11-03. (8) Nothing reaches automatic-link precision
across gaps (at best about a quarter of the needed merges before the first catalog-wrong one, AUC 0.60-0.70 on val):
stage 2 is a ranker for the reviewer for now, which is how the Photo Labeling page above uses links and scores. (9)
About 70 % of the remaining fragmentation is at gaps up to 60 s (about 40 % within 1.5 s), where a mark is seldom
visible and the evidence is continuity; in 1.5-60 s a linker trained on one day's catalog labels is ahead of the
untrained scores on val (mean AUC 0.78 against 0.73 at best; a lead, not an established result), and catalog labels
exist for every train day. (10) The instance stage limits the measure and the top of every ranking: people and gear
tracked as whales on wide-angle frames, fluke fragments, multi-whale boxes.
State of the day (state.py, state2.py, state3.py, piles.py, audit.py). Pieces per catalog animal = piles (connected
components of the links) holding at least one cluster of the animal by cluster majority, mean over the 30 animals, a
pile with two animals counting for both; 385 of 434 clusters carry an animal, 975 catalog photos. Safe clusters 12.83.
Cluster-level same links (266): 168 piles, 110 of them a single cluster, 5.43. All same links (266 + the 48 photo-pair
links between two clusters = 314): 120 piles, 44 of more than one cluster, 3.87. These are the "44 piles, 11 mixed" of
the Photo Labeling entry above; the first pass here had used the cluster-level records only. Catalog oracle joining
every catalog-same pair within a window: 1.5 s 7.93, 60 s 4.23, 300 s 2.47, 15 min 1.20, 1 h 1.00 (sighting runs per
animal: mean 1.2, max 3): the links are already below what joining every catalog-same pair within 60 s would give.
Links against the catalog (piles.py, conflicts.py, conflicts.log, sheets/conflict_0..4.jpg). 20 same links join clusters
whose majority animals differ: 19 cluster-level ones (426|427, 121|123, 58|85, 51|70, 207|263, 209|263, 61|119, 22|27,
66|89, 329|331, 294|297, 44|116, 191|194, 119|167, 30|47, 262|270, 264|266, 133|179, 3|9) and one photo-pair link
(80|86); they are the 20 record indices of the Photo Labeling entry. None carries a catalog_wrong tick (5 ticks
elsewhere), most are inside group G03, and several look like catalog errors by eye (61|119, 119|167). Joined
transitively they decide the purity. With all same links 11 piles mix catalog animals (179 clusters, 51 % of the catalog
photos) and 211 of 975 photos (21.6 %) are outside their pile's majority animal; the largest pile has 60 clusters, 197
photos and 7 animals (G03A04 69 photos, G03A01 49, G03A05 41, G03A03 30 and three more). With the cluster-level links
alone: 14 mixed piles, 39 % of the photos, 104 photos outside. Without the 20 links: 140 piles (57 of more than one
cluster, the largest of 25), 3.97 pieces per animal, one mixed pile (14 clusters, G05A04 18 photos and G05A07 3, bridged
by a cluster without a catalog animal), 9 of 975 photos (0.9 %) outside the majority, 6 of them already inside the safe
clusters. The piece count hardly depends on the 20 links, the purity depends on them entirely. likely_same is not an
identity link: all same links + likely_same joined give 66 piles and 2.73 pieces, but 12 mixed piles with 79 % of the
catalog photos and 378 photos (38.8 %) outside the majority (one pile of 70 clusters across 6 animals of G05, one of 66
clusters across 7 animals); without the catalog-contradicted links (20 same, 20 likely_same) 95 piles, 2.90, 5 mixed
piles, 6.7 % outside. It marks the interaction, as the label was defined on 2026-11-03.
What is still apart (state3.py, all same links). 438 catalog-same pile pairs remain: 357 never shown, 48 judged no
similarity, 32 likely same, 1 different. The links plus every catalog-same pair within 1.5 s give 2.23 pieces per
animal, within 60 s 1.50, within 300 s 1.40, within 15 min 1.20 (without the 20 conflicts 3.97 -> 2.30, 1.70, 1.53,
1.20): more than half of the missing merges have a same-animal piece within 1.5 s and about three quarters to four
fifths within 60 s. Queue: 1815 suggestions, 918 without any label, 411 of them between different piles (116
catalog-same, 212 catalog-different); the links plus every catalog-same open suggestion end at 2.37. By gap:
within-burst (up to 1.5 s) 130 open (59 catalog-same, 10 different, 61 unknown), confirming the catalog-same ones alone
gives 2.53; 1.5-60 s 71 open (2 / 57 / 12), 3.73; 60 s-15 min 210 open (55 / 145 / 10), 3.70. The across-gap queue was
ordered catalog-same first (queue v2), so what is left of it is mostly catalog-different; the yield is now in the
within-burst suggestions, which the queue shows last. Of the 167 catalog-same pile pairs within 15 min, 76 have neither
a suggestion nor a label.
Verdicts against the catalog (cluster majorities; the catalog is hidden on the page unless ticked). Catalog same /
different / unknown: same 217 / 19 / 30, likely_same 59 / 20 / 19 (75 % of the known ones catalog-same), no_similarity
106 / 159 / 116 (28 % of the 383 labelled catalog-same pairs show no linkable mark), different 1 / 60 / 22, unsure 0 /
14 / 0. By gap (gapbins.py; share same / likely same / no mark / different): up to 1.5 s (98 pairs) 84 / 0 / 16 / 0 %;
1.5-5 s (82) 9 / 51 / 28 / 7 %; 5-15 s (82) 20 / 29 / 44 / 6 %; 15-60 s (112) 29 / 11 / 49 / 11 %; 60-300 s (173) 26 / 7
/ 50 / 15 %; 300-900 s (295) 28 / 3 / 56 / 12 %. On a catalog-labelled day the verdicts mostly repeat the catalog (217
of the 236 cluster-level same links between clusters with an animal); what they add is the place of the evidence (the
boxes), which catalog-same pairs show no mark, 30 links to clusters without a catalog animal, and the 20 disagreements.
A linker trained on the verdicts, labelled day (dataset.py, linker_feats.py, linker_cv.py, evalx.py). Grouped CV, 5
folds over 52 animal groups; target same = 1, different and no_similarity = 0; pairs across a gap > 1.5 s: 184 same
against 83 explicit different, against the no-mark pairs, and against the 240 other-animal negatives (different, or no
mark between catalog-different clusters). The student is in-sample on this day. Columns: AUC against different / no mark
/ other animal, AP, same links ranked above the 1st / 2nd / 4th / 6th highest negative. Student fin max 0.789 / 0.840 /
0.816, 0.802, 18 / 25 / 46 / 73; student best part 0.724 / 0.822 / 0.810, 0.800, 35 / 38 / 59 / 64; stage-1 model P
(in-sample) 0.764 / 0.810 / 0.828, 0.829, 58 / 65 / 71 / 73; gbt on the student appearance features 0.753 / 0.841 /
0.824, 0.812, 7 / 31 / 41 / 57; gbt on all existing pair features 0.740 / 0.828 / 0.800, 0.795, 7 / 31 / 44 / 52.
Training on the verdicts with the existing features does not beat the single scores it is built from.
Patch matching, the researcher's "small area that matches" (windows.py, windows_feat.py, feat.py, match.py, mil.py).
Window bags: square windows of 100 to 1150 px in 7 sizes at half-side stride on every whale instance (mask fraction >=
0.2), cropped from the 2560-px day images and resized to 224, frozen DINOv3 ViT-B/16 block-11 cls token, unsupervised
PCA-whitening to 128-d fitted on the day; best match over window pairs within one size step, instance score = max,
cluster score = mean of the row maxima; 254,156 windows on the day. All windows: 0.688 / 0.862 / 0.804, 0.762, 14 / 14 /
37 / 39. A learned window-pair scorer (diagonal bilinear + unary terms, positives = windows of linked clusters, hard
negatives mined from other whales) is not better: 0.661 / 0.882 / 0.799, 0.748, 0 / 13 / 32 / 37.
What the boxes are worth. (a) Crops at the researcher's boxes on both sides (regions_feat.py, regions_eval.py,
regions_head*.py), 288 positives against 77,210 box pairs of other animals: raw frozen cls 0.912, unsupervised
PCA-whitening 0.957-0.975, WCCN from the labels 0.972-0.980, a contrastive head 0.971. The labels add about 0.01 over
unsupervised whitening. (b) That measure was too easy: it compares a boxed mark with other boxed marks (box_vs_best.py).
Against what a matcher faces, the best-looking window pair between two different whales, the similarity at the
researcher's own boxes (median 0.58) gives AUC 0.554, and 0.40 against explicit different verdicts (best window pair of
other-animal pairs: median 0.52, of different verdicts 0.62). The boxed pair is the top-scoring window pair in 2.8 % of
the links and in the top 10 in 25 %. Best pair against best pair: 0.831 / 0.709. Max-pooled patch similarity finds
agreement somewhere; it does not see a difference at the corresponding place. (c) Where to look (mil.py, sal_curve.py,
sheets/saliency_sheet.jpg): logistic regression on the whitened window features, positives = windows with IoU >= 0.3 to
a box, negatives = windows of the same instance with IoU < 0.05 (401 boxed instances, 27 animal groups). On held-out
groups a box is among the top 1 / 5 / 10 / 20 predicted windows in 0.372 / 0.708 / 0.791 / 0.873 (chance 0.050 / 0.201 /
0.331 / 0.507). Learning curve, top 1 / 5 / 10: about 2 training groups 0.232 / 0.503 / 0.633, 6 groups 0.286 / 0.642 /
0.750, 16 groups 0.389 / 0.689 / 0.786: most of it is there with 6 animal groups, so more boxes on the same day add
little. Matching only the top-10 predicted windows per instance: 0.727 / 0.897 / 0.844, 0.797, 8 / 22 / 38 / 48. (d)
Mark search from one box (marksearch.py; candidates = the clusters within 15 min, median 74; 392 queries from links
across a gap): the linked cluster is ranked 1st / top 5 / top 10 in 0.217 / 0.566 / 0.719, the in-sample student best
part in 0.334 / 0.778 / 0.898. (e) likely_same against the other pairs without a link across a gap (soft.py, 98 against
448): the time gap alone gives AUC 0.832, trained models 0.79-0.80, and inside likely_same no model tells catalog-same
from catalog-different (0.51-0.53).
Combining, labelled day (combine.py, labelled_nowhale.py; same CV columns). Rank mean = mean of the normalised ranks of
the available scores, nothing trained. Student fin + where-to-look windows 0.797 / 0.913 / 0.868, 0.853, 36 / 41 / 68 /
75; fin + body + windows 0.792 / 0.897 / 0.863, 0.857, 38 / 61 / 65 / 72; gbt on the student features + window scores
0.751 / 0.888 / 0.850, 0.832, 11 / 19 / 56 / 75. The two families are complementary: the window score separates a link
from "no mark" (0.90 against 0.79-0.84 for the student), the part-aligned student fin channel separates a link from
"different" (0.79 against 0.73 for the windows), and a trained combiner does not beat the rank mean. Against the catalog
over all pairs of the day within 15 min (2665 same, 11,124 different) the in-sample student needs no help: same pairs at
precision 0.95 / 0.90 / 0.80: student fin + body 255 / 289 / 460, fin + body + windows 199 / 263 / 392, where-to-look
windows alone 50 / 56 / 118.
Out-of-day check against the catalog (transfer.py, transfer_eval.py, transfer_eval_nowhale.py, kruskal.py; val days
2025-01-15, 2025-02-27, 2025-03-14, where the student is out-of-sample; whitening and the where-to-look model fitted on
2025-01-02 only). All cluster pairs within 15 min across a gap > 1.5 s, truth = catalog majority animals. Columns: mean
precision of each day's top 100; catalog-same pairs inside the longest ranking prefix (at least 10 pairs) with precision
>= 0.95 / 0.90 / 0.80, summed over the days; correct merges before the 1st / 3rd / 6th catalog-wrong merge when the
clusters are merged in score order (the stop is an oracle, so an upper bound for any fixed threshold); mean AUC. With
the non-whale clusters of the next paragraph left out (2627 same, 4561 different, 365 merges needed): student best part
0.86, 204 / 280 / 574, 52 / 102 / 126, 0.685; all windows 0.84, 187 / 307 / 509, 61 / 78 / 115, 0.613; where-to-look
windows 0.84, 232 / 366 / 622, 59 / 121 / 155, 0.610; rank mean student best + windows 0.88, 280 / 415 / 667, 75 / 123 /
164, 0.660; fin + body + windows 0.89, 270 / 453 / 728, 98 / 134 / 155, 0.678; fin + body 0.84, 200 / 297 / 625, 47 / 98
/ 125, 0.695. The student fin channel exists for only 23-73 % of the pairs. Linkers trained on the labelled day (gbt, 3
seeds, non-whale pairs dropped): on the researcher's verdicts (672 pairs) student + windows 0.82, 208 / 346 / 613, all
pair features + windows 0.82, 66 / 211 / 527; on the day's catalog labels (13,298 pairs) student + windows 0.85, 171 /
248 / 589, all + windows 0.90, 171 / 238 / 463. They find fewer catalog-same pairs at each precision level than the
untrained rank mean; the catalog-label linker with all features is level on top-100 precision only. Gap > 60 s (2060
same, 3971 different): student best 98 / 134 / 330, where-to-look windows 117 / 148 / 338, fin + body + windows 154 /
289 / 435. As is, with the non-whales (2715 / 4773, 377 merges needed; transfer_eval.log): student best 0.82, 161 / 197
/ 403, 43 / 78 / 114, 0.673; fin + body + windows 0.84, 166 / 308 / 708, 69 / 80 / 155, 0.666.
Tracked non-whales (wrong_sheet.py, wide_sheet.py, nowhale.py). All 12 top-ranked catalog-different pairs on 2025-03-14
are the crew member on the bow in an orange life vest, tracked as a whale instance on wide-angle frames and listed under
G02A01 / G02A02 / G02A03 (sheets/wrong_2025-03-14_1..3.jpg). The photo's catalog animal goes to the cluster of the
largest instance, so the person inherits the animal, and the whale in the frame gets none or was not detected. By eye
over the wide-angle clusters (focal length < 100 mm; sheets/wide_*.jpg): 2025-03-14 16 of 204 clusters (person on the
bow 45, 47, 48, 49, 54, 55, 56; deck emblem 50, 53; a person's head 64, 65, 66; camera gear 63, 67; boxes over person
plus whale 51, 62), 12 with a catalog animal while the real whale of those frames (clusters 46, 52) has none; 2025-01-02
9 to 11 of 434 (344, 349, 363, 364, 367, 396, 399, 401, 402, and the splash fragments 147, 148), 5 under G05A04;
2025-02-27 none (its 5 wide-angle clusters are whales); 2025-01-15 has no wide-angle frames. Telephoto clusters were not
checked. Signature, not validated as a filter: focal length 28-40 mm, box height / width 1.6-2.6, instance colour far
warmer than the water (b* difference +13 to +19). 64 of the 842 labelled pairs touch such a cluster (47 no_similarity, 6
same, 6 likely_same, 5 different; same links 396|399, 399|400, 367|369, 357|367, 366|396, 366|401): the tracked instance
of those records is not the whale, so they should not train an appearance model as they are. Leaving them out moves the
labelled-day table by 0.00-0.03 and changes no conclusion (labelled_nowhale.log).
Per val day, non-whales left out (top-50 precision, same pairs at 0.95 / 0.90 / 0.80; student best | where-to-look
windows | fin + body + windows). 2025-01-15 (429 same, 568 different; 101 clusters, 16 animals): 0.80, 21 / 25 / 45 |
0.74, 20 / 20 / 36 | 0.74, 11 / 35 / 37. Weak for every score; its top catalog-different pairs are real whales, mostly
O02 against O03, and the first one (clusters 5|8, 11 s apart) looks like one fluke (sheets/wrong_2025-01-15_1..3.jpg).
2025-02-27 (1438 / 1599; 156 clusters, 18 animals): 0.98, 140 / 172 / 358 | 1.00, 154 / 254 / 461 | 1.00, 155 / 273 /
511. It carries most of the sums; its first catalog-different pairs are fluke fragments and a multi-whale splash box.
2025-03-14 (760 / 2394; 172 clusters, 18 animals): 0.90, 43 / 83 / 171 | 0.98, 58 / 92 / 125 | 1.00, 104 / 145 / 180; as
is (848 / 2606; 184 clusters): 0.70, 0 / 0 / 0 | 0.82, 0 / 12 / 98 | 0.72, 0 / 0 / 160. The rank mean of student best +
windows beats the student at 0.95 on two of the three days (16 / 160 / 104 against 21 / 140 / 43) and at 0.90 on all
three (33 / 254 / 128 against 25 / 172 / 83). Pieces per catalog animal pooled over the val days (429 clusters, 52
animals): 8.25 at the start (8.48 with the non-whales), 1.23 for the catalog oracle within 15 min; merging in score
order up to each day's first catalog-wrong merge: student best 7.25 (52 of the 365 needed merges, 14 %), best + windows
6.81 (75, 21 %), fin + body + windows 6.37 (98, 27 %); up to the 6th wrong merge 35 %, 45 %, 42 % of the needed merges.
Where the fragmentation sits (band.py, gapbins.py, gap_merge.py; val days, non-whales left out). Catalog oracle by
joining window, pooled: start 8.25, every catalog-same pair within 1.5 s 5.56, 60 s 3.06, 300 s 1.98, 15 min 1.23. Of
the excess pieces 37 % close within 1.5 s, 34 % in 1.5-60 s, 15 % in 60-300 s, 10 % in 300-900 s (labelled day 41 %, 31
%, 15 %, 11 %). Catalog same-rate of cluster pairs by gap: val days up to 1.5 s 0.94 (176 same / 12 different; per day
0.98, 0.93, 0.92), 1.5-5 s 0.81, 5-15 s 0.50, 15-60 s 0.42, 60-300 s 0.47, 300-900 s 0.24; labelled day 0.94 (190 / 13),
0.52, 0.23, 0.25, 0.19, 0.18. The within-burst part is the known price of the strict tier (1.5-s bursts gave 5.7
clusters per animal with 18 impure photos on val, 2026-11-03). Joining every pair up to T s apart and nothing else
(correct / wrong merges by catalog majority, pieces per animal): val days 0.5 s 74 / 1, 6.83; 1.0 s 126 / 6, 5.83; 1.5 s
141 / 9, 5.54; 5 s 174 / 25, 4.85; 60 s 230 / 115; labelled day 0.5 s 88 / 6, 1.5 s 140 / 14. Within-burst pairs are a
94 % suggestion class, not an automatic rule. The photo's catalog animal goes to the cluster of the largest instance, so
two whales alternating as the largest instance of a burst read as catalog-same and the within-burst same-rate is an
upper bound; the researcher's verdicts point the same way (82 of 98 within-burst pairs same by a mark, none different).
The short band 1.5-60 s on the val days (band.py; 567 catalog-same and 590 catalog-different pairs). Columns: mean
top-50 precision, same pairs at precision 0.95 / 0.90 / 0.80, mean AUC. Time gap alone 0.85, 48 / 78 / 201, 0.644;
student best part 0.81, 117 / 147 / 273, 0.688; where-to-look windows 0.76, 121 / 149 / 264, 0.620; rank mean fin + body
+ windows 0.83, 114 / 169 / 323, 0.686. Linkers trained on 2025-01-02 (gbt, 3 seeds, non-whale pairs dropped): catalog
labels with the time and geometry features only 0.67, 0 / 11 / 50, 0.630; catalog labels with all pair features +
windows (13,298 pairs) 0.89, 136 / 212 / 314, 0.781 (per day 0.768, 0.746, 0.829); catalog labels of the band only (1584
pairs) 0.87, 92 / 186 / 317, 0.748; the researcher's verdicts (672 pairs) 0.78, 102 / 167 / 271, 0.625. In this band a
linker trained on one day's catalog labels with time, geometry, student and window features is ahead of every untrained
score on mean AUC (0.781 against 0.725 at best, the rank mean with the time gap) and at precision 0.95 and 0.90, level
at 0.80. Per day at 0.90 it finds 67 / 117 / 28 pairs against 57 / 127 / 31 for the best untrained score of each day
(time gap, fin + body + windows twice): about as good as the best single score on each day, which no untrained score
manages, and not better than it (AUC on 2025-02-27 0.746 against 0.767 for fin + body). A lead worth training on all
train days, not an established result; the verdicts do not help. At longer gaps the rank mean stays ahead at the top:
60-300 s (1257 / 1410)
0.83, 159 / 220 / 420, 0.651 against the catalog-label linker's 0.78, 28 / 59 / 408, 0.690; 300-900 s (803 / 2561) 0.76,
76 / 103 / 126, 0.691 against 0.65, 34 / 48 / 92, 0.667. Merging in score order over the pairs up to 60 s apart (270
merges possible), correct merges before the 1st / 3rd / 6th catalog-wrong merge: time gap alone 86 / 143 / 172 (86-89
before the first over 200 random orders of equal gaps); catalog-label linker of the band 70 / 108 / 132; fin + body +
windows 53 / 112 / 136; verdict linker 45 / 93 / 139; catalog-label linker on all pairs 43 / 86 / 131; student best 27 /
93 / 118. Shortest gap first gets about as far before the first wrong merge as the best appearance ranking over all
pairs within 15 min (86 against 98); on the labelled day it fails at once (1 / 28 / 50): an ordering, not a rule.
Caveats. The student is in-sample on 2025-01-02 and the CV folds, though grouped by animal, sit inside one day. The
labelled pairs are the suggestion queue (picked by the stage-1 model and the student, catalog-same first across gaps),
not a random sample. The catalog is a noisy reference and its animal goes to the largest instance of a photo. 2025-02-27
carries most of the val sums and 2025-01-15 is weak for every score. The non-whales were found by eye among wide-angle
clusters only. The rank-mean combination and the gap-band results were looked at on the val days, so they are not
confirmed on untouched days (the eval days were not used). The first-wrong-merge stop is an oracle stop. The verdict
target ("a link was found") is not the same as catalog-same. The 114 photo-pair records enter the state of the day only,
not the training probes. The short-band linker has one training day. No backbone was fine-tuned on these labels: every
probe used frozen features (student embeddings, frozen DINOv3 windows) with whitening, heads or gbt on top.
Offered, not started: (a) a review of the 20 catalog-contradicted links (sheets exist) and of the O02 / O03 pairs on
2025-01-15, with the outcome written as catalog_wrong ticks or changed verdicts; (b) the open within-burst suggestions
first in the queue, or a bulk-confirm mode for them (130 open, 3.87 -> 2.53 on this day if the catalog-same ones are
confirmed); (c) a short-gap continuity linker trained on the catalog labels of all train days (time, geometry, student,
where-to-look windows), tuned on val; (d) a false-instance filter before stage 1 and catalog assignment to the instance
that is the whale; (e) the rank mean of student and where-to-look window score as the order of the bursts queue and as a
feature of the Photo Labeling ranker, plus pile-to-pile suggestions; (f) the next labelling on a val day (a measure with
the researcher's own verdicts) or on other train days rather than more boxes on 2025-01-02; (g) stage 2 as aligned part
comparison (fin, fluke, contour) plus where-to-look patches.
Files: experiments/stage2_labels/ holds the scripts, ds.pkl (labelled records, folds, region boxes), the per-pair tables
feats_<day>_all.pkl and transfer_raw_<day>.pkl for the four days, the score pickles, the logs and sheets/. The window
feature caches (3.8 GB) were deleted: before re-running a feature script run windows.py DAY, windows_feat.py DAY,
match.py 2025-01-02 pcaw128 and regions_feat.py there. The evaluation scripts (state.py, state2.py, state3.py, piles.py,
audit.py, conflicts.py, linker_cv.py, labelled_nowhale.py, kruskal.py, nowhale.py, transfer_eval.py,
transfer_eval_nowhale.py, band.py, gapbins.py, gap_merge.py, soft.py) run from the folder as is, with the repo's .venv
python -u, OMP_NUM_THREADS=1 and PYTHONPATH = the repo root.

## 2026-11-11 evening — Photo Labeling round 2: frame traces, passers-by, the dorsal -> fluke series (user + researcher notes)
Researcher's notes (via the user): green label = scarring studies (chosen by hand), yellow = other studies; a photo
of land without a whale = the boat switches groups; a photo of the boat railing after a series = the role of the
animal just photographed (red female / nuclear female, green male / escort, blue calf); animals are matched by the
series of shots that runs from the dorsal fin to the fluke of one dive. User's scenario: flag photos taken
opportunistically of another whale in another direction, by a scene / lighting change.
Marker photos: none in the kept set. The 97 photos without a whale instance are distant whales and splashes, the 291
frames at <= 45 mm are tagging / biopsy documentation (a person on the bow: not for publication), and no photo of
the 11 111 has more than 9 % saturated red or 0.1 % saturated green (a railing close-up). They are culled.
Frame traces: the kept files keep the camera's file name (...-184A4470), so a hole in the counter between two kept
photos of one body = deleted frames (label_assist.frame_traces; a body without any hole all day = unknown). 10.4 % of
the 11 314 consecutive kept pairs have a hole. P(animal changes) with no hole / one deleted frame: 0-2 s 0.02 / 0.60
(n 20), 2-10 s 0.60 / 0.90 (n 121), 10-60 s 0.50 / 0.73, 60-300 s 0.61 / 0.73, > 300 s 0.54 / 0.77. Group changes:
84 % have a hole (median 2 frames) against 3.1 % of the steps without an animal change; at gaps > 300 s a group
change follows 6 % of the gaps without a hole, 38 % with one deleted frame, 57 % with 2-3.
Boundary model with the traces (del_some, log1p_del, del_none added; label_assist fit): held-out AUC 0.829 -> 0.846,
at P >= 0.35 precision 0.71 -> 0.74, recall 0.57; 1 group change among the 55 held-out sighting gaps without a hole.
Link model: the trace as a pair feature (frames deleted between two stacks) does nothing (val first choice 0.843 ->
0.830, eval 0.786 -> 0.789); demoting the animal photographed just before a hole: val 0.853 -> 0.843, eval 0.786 ->
0.791, not used. As a run stop (a run does not cross a hole): offered run stacks right eval 0.890 -> 0.911, val 0.855
-> 0.870 (label_assist simulate); on the page's replay eval 329 of 357 (0.922, was 0.899), 552 decisions, val 0.868.
Passers-by during an encounter (label_assist passers; scratch scripts for the scene part): scene descriptor = median
Lab of 12 horizontal bands of the frame + brightness / saturation / glare (15 s for all days). Consecutive photos of
one camera, "the group changes" against "it does not" inside gap bands: profile distance AUC 0.59-0.65, range change
0.60-0.65, exposure 0.50-0.57, water colour 0.51-0.60; a burst against the bursts around it (5 min): profile outlier
0.70, range 0.69; a learned combination (gbt, leave-one-day-out) 0.63. At the stack level inside catalog encounters
(4 027 stacks with >= 3 group stacks within 5 min, 30 passers-by = 0.7 %): isolation (1 - best link to those stacks)
AUC 0.958, farther 0.70, scene outlier 0.68; flags: scene outlier >= 3x 5 of 294 (2 %), farther 14 of 329 (4 %),
isolation >= 0.4 18 of 158 (11 %, recall 0.60), isolation and farther 8 of 53 (15 %); adding scene or range to
isolation in a logistic changes nothing (AUC 0.96). The camera swings as far between members of one group; a group
animal photographed once is as isolated as a passer-by. 80 % of the 585 passer-by photos are > 5 min from any group
photo (the encounter cuts isolate them); small isolated sightings are passers-by only 20 % of the time (base 10 %),
so no automatic "O" stretches either. The page shows the scene (before / whole frame / after insets, large viewer) and
tags "stands alone · passer-by?" with those odds; no scene flag.
Dorsal -> fluke series (chains of stacks of one camera, <= 8 s apart, link >= 0.9, fin before fluke): 700 on the 28
days, 97.3 % hold one catalog animal, 65 % of the 471 animals have one (77 % have any fluke photo). Sorted first
(stack by stack, catalog as researcher): eval first choice 0.789 -> 0.777, top three 0.968 -> 0.965; the true animal
already shows a same-view photo 0.885 -> 0.904 and both dorsal and fluke 0.751 -> 0.845; 66 of the 91 new animals are
then made in the series pass. An option on the page ("ID series first"), off by default.
Also: coastline basemap for the boat track from OpenStreetMap (Overpass natural=coastline, stitched to 42 land rings,
8.7k points, 67 kB; all 10 617 photo positions fall in the water), large photo viewer, railing colours on the roles.
Open: an unculled day would let the page read the markers themselves (land -> cut, railing colour -> role, railing
-> end of a series).

## 2026-11-11 night — the five steps after the first labelled day: review queue, burst-first queue, non-whale filter, short-gap linker, scored queue
Request (the user, after the entry "2026-11-11 later" above): "Ok lets do 1-5, start with 1 and I can go through and
examine those 20 while you build the other steps". The five steps = offers (a) to (e) of that entry: (1) the 20 links
that disagree with the catalog queued for review on the bursts page, (2) the open within-burst suggestions first in the
queue, (3) the short-gap linker trained on the catalog labels of all train days, (4) a filter for non-whale instances
ahead of the clustering, (5) the averaged ranking as the suggestion order of the bursts page and as a hint on the Photo
Labeling page. All five are built; 1, 2, 4 and 5 changed the review page and the production clusters, so this entry also
says what changed on disk.
Outcome. (1) Live on the review page: the review queue for the 20 disagreements, the within-burst tier first with S
needing no box there, and the queue ordered by the models (short-gap tier, then the rank mean) with the catalog out of
the order. (2) The non-whale filter is in production: 210 of 11 810 instances left out on the 28 days, every day
re-clustered, still no wrong automatic link on any labelled day; every flag on the val days and on the eval days is a
real non-whale by eye. (3) The researcher's saved links follow their photos across the re-clustering; the label file is
untouched. (4) The short-gap linker trained on the catalog labels of the train days is no longer only a lead: with 11 of
the 21 train days it is ahead of every untrained score on each val day in the band 1.5-60 s (AUC 0.83 against 0.73, 314
against 197 same pairs at precision 0.90); the fit on all 21 days was still waiting for the feature extraction when this
was written (see the addendum). (5) The window score does not help the Photo Labeling page's ranking (one more correct
first choice in 378 held-out decisions), so that page was left as it is. (6) Window features, where-to-look scores and
scored candidate links exist per day in DAY/safe/, made by one module (cluster/link_scores.py).
Step 1, the review queue (Whale-Review-Web backend/pairs.py _review_items, frontend bursts.js / bursts.html; README
"Safe clusters level, changes of 2026-11-11"). The safe-clusters level of /bursts has a yellow "review n" button
(address &review=1) that walks the saved "same" links whose two clusters carry different catalog animals (the majority
label of each cluster; a tie made by a Pairs-page label counts while the pair has no direct link): 20 items on
2025-01-02, the same 20 as in that entry, in day order. Each opens with the catalog shown, the saved boxes loaded and,
per cluster, the group the other saved links put it in with this link and the links still to review left out (so a
single odd cluster among many of one animal reads differently from two established groups; without the 20 the groups are
pure, e.g. 18 clusters G03A01 against 25 clusters G03A04 for the link 44|116). S the link stands (the catalog-wrong box
and an optional "which animal" list say more), D different, N no linkable mark after all, U can't tell, G or X leave it.
Saving replaces the record and adds review = {kind "catalog", time, animal?, was: the whole first record with its
index}; the boxes of a withdrawn link move to review.was; a tie gets a direct cluster record next to it and a direct
"different" outweighs the tie in the page's graph. Item 9 (80|86) is not a disagreement of two labels: cluster 86
carries 5 photos of G03A03 and 5 of G03A04 and the two linked photos are both catalogued G03A04, so the question there
is whether that safe cluster holds two whales. No pair has a tie and a direct label, and no "different" verdict sits
inside one of the researcher's own piles (checked with and without the 20). Schema additions are additive (review,
same_by, model_scores); the label file was not touched by me: 956 records, saved 2026-11-11 15:36:13, same md5 before
and after every install. Browser tests against a sandbox copy of the app with its own copy of the label file (scratchpad
review/test_review.py: 32 checks, among them record replaced in place, first version kept whole, no other record
changed, tie handled, reload, normal saves untagged, other levels untouched).
Step 2, within-burst first (order "burst", now the fallback order). The 2026-11-03 queue put the pairs across a gap
first; the page now starts with the splits inside bursts (two clusters at most 1.5 s apart) in day order. For those
pairs "Link found" saves without a region pair, because the sequence of frames is the evidence and usually no mark is
visible: the record carries same_by "sequence", joins the clusters like any same link, and can never be a positive for a
mark model (no region). Across a gap S still needs the boxed mark, and "Likely same" stays the soft call that joins
nothing (it marks the interaction, as that entry found). On 2025-01-02 the tier holds 323 pairs, 119 of them open under
the current links; by the catalog the stage-one suggestions of that tier without a shared photo are 185 same / 13
different there and 176 / 9 on the val days (95 %), and the pairs whose clusters share a photo (two instances of one
frame: fragments of one whale or two whales) have almost no catalog label, so those are the researcher's call. The
earlier orders stay reachable (&order=needed is the order the 842 records were collected under, &order=burst,
&order=blind, &order=model).
Step 4, the non-whale filter (model_fine_tuning/cluster/link_scores.py instances / not-whale, safe_tracklets.py
--filter; experiments/stage2_linker/nw_*.py, nw_labels.json, sheets/nw_*.jpg; model
experiments/safe_clusters/not_whale.pkl). Input: one square crop per instance (its box plus 15 %), frozen DINOv3
ViT-B/16, unit CLS and unit mean patch token, standardised; a logistic regression (C 0.01, balanced). Labels by eye from
contact sheets in three rounds of looking at what the current probe was least sure were whales among the unlabelled
train instances (unlabelled telephoto instances count as whales): 150 not-a-whale instances on 14 train days (crew on
the bow in orange vests, the biopsy crossbow, the deck and its emblem, camera lenses and gear in the foreground,
railings and poles, a tag on a pole, drones), 84 whales, 70 left out as unclear, most of them masks that merge a person
or boat part with a whale next to the bow. The probe started from the 14 instances of the nine clusters found by eye on
2025-01-02 and already flagged 31 of the 32 known instances on 2025-03-14 with nothing flagged on the other two val
days. Final probe, threshold 2.0 chosen on the val days: 34 of 34 by-eye non-whales flagged on 2025-03-14 (the 32 known
and two blurred straps at 110 mm that the first by-eye pass, wide-angle clusters only, had missed), none on 2025-01-15
and 2025-02-27, lowest labelled score 2.57 against 0.35 for the highest whale (a whale beside a sailboat). In-sample on
the train days all 150 labelled non-whales score above 3.98 and the by-eye whales at most 0.23; 15 of the 70 merged or
unclear masks are above the threshold and are dropped with their whale, which is the known cost (the fix belongs in the
mask stage). Eval days, looked at once after the freeze, for the report only: 11 of 2 290 instances flagged, all 11 not
whales by eye (the crew member with the tagging pole, a drone, a tag on a pole, a blurred pole), and no clear miss among
the next 37 by score. Flagged in all: 210 of 11 810 instances (train 165, val 34, eval 11). Splash fragments and blows
are not targets of the filter (they belong to a whale).
Clustering with the filter (safe_tracklets --filter reads DAY/safe/not_whale.json): flagged instances form no tracklet,
a frame with a person and a whale counts as a single-whale frame for the strict tier, and when the dropped instance was
the largest one (~0) the photo's catalog label goes to the largest instance left. The stage-one model is unchanged
(safe_model.pkl, strict tier, P 0.995). Without and with the filter (report_{val,eval,train}_{nofilter,filter}.json):
wrong automatic links 0 and 0 on every split; photos in impure clusters val 4 and 4 of 878, eval 4 and 4 of 2 146, train
12 and 12 of 7 986; clusters per catalog animal val 8.47 to 8.29, eval 9.18 to 9.10, train 10.00 to 9.85; days without a
flagged instance come out identical cluster for cluster. Production switched for all 28 days (clusters 5 170 to 5 040,
suggestions 18 440 to 17 870); the earlier files are kept as DAY/safe/clusters_v1.json and the clustering's name in the
review records is now "safe_model|P0.995|cost1.5|strict|not_whale_v1". On 2025-01-02: 434 to 420 clusters (18 instances
left out, 12 clusters gone, 2 pairs of clusters merged by links the strict tier had blocked), 12.83 to 12.63 pieces per
catalog animal alone and 3.87 to 3.80 with the researcher's links.
The researcher's links across the re-clustering. Cluster ids are renumbered by a re-clustering, and the 842
cluster-level records name clusters by id under the clustering in cluster_rev. The page backend now carries every saved
link over to the clusters its two photos are in (pairs._link_sets; the photo keys were always the durable statement) and
lets a new verdict replace the carried record (pairs._replaces); the label file is not rewritten. On 2025-01-02 778
links carry over and 64 drop out: 62 name an instance that is no longer tracked (38 no-mark, 9 different, 8 same, 7
likely-same verdicts between or with non-whale clusters) and 2 same links whose photos now share a cluster. One pair has
two carried records (no-mark and likely-same). The 20 review items are unchanged. The Photo Labeling page maps earlier
links by photo key as well and its own end-to-end test passes on the new clusters (tests/labeling_e2e.py). The analysis
pickles of experiments/stage2_labels refer to the earlier ids (clusters_v1.json).
Window features for every day (model_fine_tuning/cluster/link_scores.py: windows, look, scores, pairs; the functions of
stage2_labels/{windows,feat,match,transfer}.py ported, nothing else changed). Per day: windows.npz (windows whitened to
128 d with the whitening of 2025-01-02, experiments/safe_clusters/window_whitening.npz), inst_feat.npz,
window_scores.npz (D_sal from the ten most salient windows per instance by the where-to-look model, D_all from all
windows; where_to_look.pkl refitted in the module: 576 boxes on 401 instances, 2 920 windows on a box against 83 819
off; the reference day uses its out-of-fold saliency). Port check against the pickles of the earlier analysis on four
days: same finite mask, mean absolute difference 2e-5 to 2e-4, correlation 0.9999 or better (a handful of entries move
by up to 0.14 where two windows tie for the tenth place). 11 810 instances, about 2.9 million windows; the extraction is
bound by the GPU at roughly 560 crops per second (three workers side by side reach about 670). A first attempt with
three workers and the loader settings of a single one filled /dev/shm with queued crops (each queued batch of float32
crops sits there) and killed loader workers; the loader now queues two photos per batch with four workers, and the
result files are written through a temporary file.
Step 3, the short-gap linker (experiments/stage2_linker/linker.py, tables.py; link_scores.pair_table). One row per pair
of safe clusters within 15 min: time (gap, spans, sizes, same camera), the continuity between the earlier cluster's last
frame and the later one's first (safe_clusters.pair_features), the student cosines over all instance pairs per part, the
four window scores; label = same majority catalog animal; gradient-boosted trees (depth 3, 300 rounds, three seeds),
checked on the val days in the band 1.5 s < gap <= 60 s (586 same, 574 different pairs on the new clusters). Untrained
scores there: time gap alone AUC 0.642, student best part 0.690, rank mean with the time gap 0.725; same pairs at
precision 0.95 / 0.90 / 0.80: 125 / 162 / 299 for the student, 121 / 197 / 332 for the rank mean. Trained on the
reference day alone (the lead of the "later" entry, reproduced on the new clusters): AUC 0.739 with band pairs, 0.786
with all pairs to 15 min (142 / 223 / 337). Trained on 11 train days (4 833 band pairs, 1 834 same; the other ten days
were still in the feature extraction): AUC 0.825 with per-day values 0.788 / 0.813 / 0.872 against 0.69 / 0.77 / 0.73
for the best untrained score of each day, 273 / 314 / 440 same pairs at precision 0.95 / 0.90 / 0.80, and 108 correct
merges before the first catalog-wrong merge over the pairs within 60 s against 88 for the time gap alone and 69 for the
rank mean with the time gap (273 needed). Variants on the same 11 days: trained on all pairs to 15 min 0.823 (229 / 326
/ 427); appearance as within-day ranks 0.817; without the student features 0.803 but 308 / 335 / 411; without the window
scores 0.813 (228 / 328 / 415); time and continuity only 0.709 to 0.722 (177-203 / 222-234 / 276-286). So more days is
what the model needed, the appearance features carry about 0.10 AUC on top of continuity, and student and window scores
are worth about the same here. The final fit on all 21 days, the choice of the variant and the eval read follow in the
addendum below.
Step 5, the scored queue (link_scores.py pairs -> DAY/safe/link_scores.json; pairs._queue order "rank"). Per day the
file lists every pair of clusters inside one burst, each cluster's four best partners up to a minute away (by the
linker's P, lp) and its five best up to 15 min away by the rank mean (rm: the mean within-day rank of the student's fin
and body cosines and the where-to-look window score); 2 802 candidates on 2025-01-02, 3 128 with the stage-one
suggestions merged in. The page shows the within-burst tier in day order, then "short gap" by lp, then "across a gap" by
rm; a saved record keeps the pair's scores (model_scores), which will measure the ranking on the researcher's own
verdicts. The rank mean on the val days with the new clusters, all pairs beyond a burst within 15 min (2 692 same, 4 542
different): same pairs at precision 0.95 / 0.90 / 0.80 are 277 / 480 / 787 against 224 / 328 / 640 for the student's
best part, correct merges before the first catalog-wrong merge 102 against 52 of 368 needed; beyond the band (gap > 60
s) 171 / 302 / 435 against 107 / 171 / 356. (Eval read: addendum.) The hint for the Photo Labeling page was tested
before touching that page (label_hint.py): its pair logistic (sequence P, student cosine per part, gap) refitted
leave-one-val-day-out with and without the window score (the best D_sal over the instance pairs of two stacks and
whether it exists) gives first choice 0.854 against 0.857 and top three 0.989 against 0.989 over 378 decisions with an
existing animal; no gain, so labeling.py and labeling_model.json are unchanged. Inside an encounter, with the sequence
model and the closest stack of each animal, that ranking has little left for a mark score to add; the window score earns
its place across gaps on the bursts page.
Caveats. The catalog is the reference for the filter's clustering check, the linker and the rank mean, and 20 of the
researcher's links say it is wrong in places. The linker's student features are in-sample on the train days (the student
was trained on them); the variant without them is within noise of the full one on val, which bounds the harm. The linker
and the tier limits were chosen on the three val days; the eval read is the only untouched one. The filter's labels are
one person's by-eye calls from 250-px tiles; 15 flagged instances are masks that merge a person with a whale. The page's
order was not tuned for the fewest wrong merges: within a burst it is day order, because the researcher looks at every
pair anyway. Pair tables and scores were built on the new clusters; numbers in the "later" entry were on the earlier
ones with the by-eye non-whales removed by hand, so they differ slightly (band pairs 586 / 574 against 567 / 590).
Files. Code: Dinov3-WhaleID model_fine_tuning/cluster/link_scores.py (new), safe_tracklets.py (--filter);
Whale-Review-Web backend/pairs.py, frontend/js/bursts.js, js/pairtool.js, bursts.html, tests/bursts_e2e.py (new, 37
checks on a temporary copy of the labels), README. Models in experiments/safe_clusters/: not_whale.pkl,
window_whitening.npz, where_to_look.pkl (+ where_to_look_oof_2025-01-02.npy), link_gbt.pkl. Per day in
experiments/days_all/DAY/safe/: clusters.json (re-made), clusters_v1.json (before), not_whale.json, inst_feat.npz,
windows.npz, window_scores.npz, link_scores.json. Experiment folder experiments/stage2_linker/: nw_*.py with
nw_labels.json and sheets/, report_*_{nofilter,filter}.json, tables.py and tables/, linker.py with its logs,
final_eval.py, label_hint.py, the extraction and scoring drivers, staging_days/ (the staged re-clustering, symlinks).
For a new day: safe_clusters features, link_scores instances, link_scores not-whale, safe_tracklets cluster --strict
--filter, link_scores windows (GPU), link_scores scores, link_scores pairs.

## 2026-11-11 night, addendum — the linker on all 21 train days, the eval read, the researcher's review of the 20, five second looks
Outcome. (1) The short-gap linker is confirmed three ways: AUC 0.82 on the val days, 0.82 on the eval days (read once),
0.84 on the train days with each day held out (higher than every untrained score on 19 of 19 days). (2) The
where-to-look window score is NOT confirmed as a ranking aid: its gain on the val days is absent on the eval days. (3)
The researcher reviewed the 20 links against the catalog: 11 kept, 9 withdrawn; no contradiction left inside the links;
five spots are queued for a second look, each with its reason on the page. (4) The scored queue is live for all 28 days;
on the train days it uses linkers fitted without that day.
The linker on all 21 train days (linker.py, linker_21.log; 8 931 catalog-labelled pairs in the band 1.5 s < gap <= 60 s,
2 698 same). Val days, band (586 same, 574 different): chosen variant = fitted on the band pairs with the features as
they are (time, continuity, student cosines, window scores): AUC 0.819 (per day 0.763 / 0.819 / 0.876) against 0.690 for
the student's best part and 0.725 for the rank mean with the time gap; same pairs at precision 0.95 / 0.90 / 0.80: 268 /
314 / 432 against 125 / 162 / 299 (student) and 121 / 197 / 332 (rank mean). The other variants are within noise of it
(AUC 0.796 to 0.822: trained on all pairs to 15 min, appearance as within-day ranks, without the student, without the
window scores), so the plainest was taken; time and continuity alone give 0.692 to 0.725. Ten more train days did not
move it (0.825 on eleven days): the model is limited by its features, not by the number of days. Saved as
experiments/safe_clusters/link_gbt.pkl (link_gbt_v1).
Each train day held out (linker.py --loo, linker_loo.log; 19 days with at least 30 band pairs, 2 671 same and 6 228
different pairs): mean AUC 0.844 against 0.687 for the student's best part, 0.700 for the rank mean of fin and body and
0.592 for the time gap; higher than each of them on 19 of 19 days; same pairs at precision 0.95 / 0.90 / 0.80 summed
over the days 570 / 1 049 / 1 627 against 451 / 583 / 939 (student). The reference day held out: 0.806. The student's
cosines are in-sample on these days (it was trained on them), which flatters the student rows and the linker's student
features alike. In this band the window score does not help on the train days (rank mean with it 0.644 against 0.700
without).
Eval days, read once after everything was frozen (final_eval.py eval, final_eval_eval.log; 2024-12-18, 2024-12-26,
2025-02-19, 2025-03-21). The linker in its band (1 152 same, 1 857 different): AUC 0.819 (0.816 / 0.773 / 0.848 / 0.839)
against 0.665 for the student's best part, 0.676 for the rank mean of fin and body and 0.593 for the time gap. The
precision counts say less there: 2024-12-18 has 567 same of 629 band pairs, so every score clears precision 0.90 on it
by the base rate; on the other three days the linker finds 61 same pairs at precision 0.90, the time gap alone 51, the
student none, and on 2024-12-26 (352 same, 1 289 different) no score reaches 0.90. So the ranking carries over; the
confident head of the ranking is thin.
The rank mean with the window score on the eval days does not repeat its val result. All pairs beyond a burst within 15
min (3 444 same, 9 424 different): AUC 0.659 against 0.683 for the student's best part and 0.689 for the rank mean of
fin and body alone; same pairs at precision 0.95 / 0.90 / 0.80: 203 / 332 / 857 against 247 / 419 / 845 (student).
Beyond the band (2 292 same, 7 567 different): 56 / 65 / 138 against 34 / 72 / 179 (student) and 54 / 74 / 156 (fin and
body), top-100 precision 0.62 against 0.65 and 0.66; the window score alone has AUC 0.574 (0.604 on val). On the val
days the same comparison was 171 / 302 / 435 against 107 / 171 / 356. So the claim of the "later" entry and of the
paragraph above ("the rank mean finds more catalog-same pairs at precision 0.90 / 0.95 than the student alone") holds on
the three val days only. The where-to-look model has boxes from one day and one researcher; its benefit is unproven. The
page's across-a-gap order still uses the rank mean with the window term: it was not changed after the eval read (eval
days do not tune), the difference is small either way, and each saved verdict now carries the pair's scores
(model_scores: rm, lp, p, app, win), so the researcher's own verdicts will settle whether the window term stays.
The page. DAY/safe/link_scores.json exists for all 28 days (889 to 2 789 candidate links per day). On the 21 train days
the short-gap P comes from the fit that left the day out (experiments/safe_clusters/link_gbt_loo/DAY.pkl, picked up by
link_scores pairs; "held_out" in the file), so no catalog label of a day enters that day's order; the student embedding
was still trained on those days. Val, eval and new days use link_gbt.pkl.
The researcher's review of the 20 (2026-11-11 19:26 to 19:33, user ST, all 20 saved from the review queue). Kept as
same, 11: 22|27, 58|85, 61|119, 80|86, 119|167, 133|179, 207|263, 209|263, 264|266, 294|297, 412|413. Withdrawn to
different, 9: 3|9, 30|47, 44|116, 51|70, 66|89, 121|123, 191|194, 262|270, 329|331. The catalog-wrong box and the
which-animal list were not used and no note was written; the first version of every record is in its review.was. State
after it (current clusters, links carried by their photos): 248 same, 83 different, 343 no-mark, 91 likely-same and 14
unsure cluster links plus 48 ties from Pairs-page labels; no different verdict inside one of the researcher's groups;
125 groups; 12.63 pieces per catalog animal from the clusters alone, 3.83 with the links; 76 of 969 catalog photos sit
outside the majority animal of their group (211 of 975 with all 20 links before the review, 9 without them).
What the 11 kept links say against the catalog (my reading). Three attach a single-photo cluster to an established group
of another animal (119: one photo G03A04 between G03A06 clusters, tied by two kept links; 264: one photo G03A03 with
G03A06 x23; 179: one photo G03A06): one catalog label each. Two say the passer-by O03 (10 photos) is G03A03, the kind of
statement the catalog itself makes in its "same as" notes (G05A03 same as G03A01, O06 same as G03A09 on this day). Four
join two established catalog animals: 58|85 (G03A08 x12 with G03A01 x21), 22|27 (both animals of the pair G02, 6 and 14
photos), 412|413 (G06A01 x12 with G06A03 x4, two clusters 0.7 s apart) and 294|297 (G03A01 x17 with G05A07 x11). The
largest mixed group (G03A04 x69 with G03A03 x30) is different in kind: every link on its path joins two photos with the
same catalog label. It is held together by clusters 86 (10 frames in 2 s: the catalog gives the first five to G03A03 and
the last five to G03A04) and 81 (3 frames in 1 s: G03A04, then G03A03 twice), bursts with two whales side by side in the
frame. Either the tracked instance changes whale inside the burst (these are the day's impure safe clusters) or the
catalog's single label per photo moves between the two whales while the tracked one stays. A link is a statement about
two photos; taken at the cluster level it joins everything in both clusters, so one mixed cluster merges two animals
without any wrong link. Consequence for the next stage: chain and score the researcher's links by photo key wherever a
cluster carries two catalog animals, and let the Photo Labeling page (which moves single frames) decide such clusters.
Second looks (Whale-Review-Web pairs._review_items, _review_asks; data/2025-01-02/review_extra.json). The review list
has two more sources. A "different" verdict between two clusters the catalog gives to one animal goes on it by itself
(sense "different"): 1 of the 83 different verdicts (65 agree with the catalog, 17 have no label on one side), 39|110,
both G03A01, 887 s apart, a group of 18 clusters against a group of 3. And an ask file lists pairs to look at again with
the reason shown on the page; an ask stays open until the pair is saved after it was asked. Four asks: 80|86 and the tie
81|86 (is the boxed whale one animal through the bursts of clusters 86 and 81, answer in the note), and 294|297 with
329|331 (the first kept, the second withdrawn, although the other links put 329 in one group with 294: the catalog's
G05A07 is now in two of the researcher's groups, 11 photos joined to G03A01 and 7 called different from it). Five open
items, review list 22. tests/bursts_e2e.py covers the three sources (44 checks, on a temporary copy of the labels).
Open. The five second looks. 125 open within-burst pairs on 2025-01-02 (66 the catalog calls same, 6 different, 53
without a label). The across-a-gap order (window term or not) on the researcher's verdicts. A second labelled day,
preferably a val day, where the student is out of sample and the where-to-look model meets boxes it was not fitted on.
The mask stage for the 15 flagged masks that merge a person with a whale.

## 2026-11-11 night — Photo Labeling round 3: the Animals step animal by animal (user: first stack = group 1 animal 1, closest stacks first, hard ones last)

Asked. The user opened step 2 and saw a stack on the left and nothing on the right (the first stack of an encounter has
no animal to compare with). Direction given: the first stack is automatically animal 1 of the group; then the closest
stacks come first; one animal is collected, then the next, until only the hard stacks are left.
Simulated first (scratch script on the page's own P matrix, stacks, scopes and run rule, the catalogue as the
researcher; 24 fit days = train + val, eval days not looked at). Baseline, stack by stack in capture order: 2 416
decisions, 4 026 looks (animals held against a stack before the answer; a new animal = all listed). It reproduces the
page's replay (556 decisions on the eval days after today's re-clustering, 552 before).
- A pure "is it this animal?" pass per animal is expensive: every stack asked about its closest animal down to link 0.5
  = 6 061 decisions, 59 % of the answers a no; down to 0.3 = 8 634. Calibration of that question in animal-first order:
  link >= 0.9 a yes 93-96 %, 0.8-0.9 73 %, 0.7-0.8 49 %, 0.6-0.7 27 %, 0.5-0.6 14 %. Stack by stack, with every earlier
  animal present, a link >= 0.7 is right 89 % (val, 60 % of stacks): the same score means less when the stack's own
  animal does not exist yet. The link model is sure about stacks close in time, not across surfacings.
- Margin to the second animal and the appearance-only cosine add little to "is the closest animal right" (val AUC 0.797
  link alone, 0.807 with margin, 0.830 with appearance margin): no calibrated shortcut there.
- The next animal's first stack cannot be automatic: the open stack least like the animals so far is a new animal 55 %
  of the time below link 0.3 and 75 % below 0.15. Only the first stack of an encounter is certain.
- Runs must grow in both directions once the order is not the capture order (forward only: 3 009 decisions at
  threshold 0.8; both directions: 2 519).
- Threshold for "same whale?" 0.9 / 0.8 / 0.7: 85 / 76 / 62 % yes, 2 352 / 2 519 / 2 955 decisions, 4 122 / 4 141 /
  4 259 looks. Taken: 0.8, "new animal?" below 0.15.
Built (Whale-Review-Web frontend/js/lab_core.js nextTask / queueFor / buildUnit / refuseUnit, lab_animals.js rewritten,
backend state section `no`, labeling_model.json `collect`). Order of work per encounter: first stack = first animal
(saved without a question); "same whale?" for the open stack closest to the animal being collected while link >= 0.8
and that animal is the stack's closest, then any other animal's closest; "next animal?" for stacks below 0.15 to every
animal, capture order; the rest last, closest first, "which animal?". Screen: animal left, stack in question right,
animals it could be far right, "coming up" queue. space yes, N no (saved; a stack that can be none of the animals
starts a new one in the same step), 1-9, A new, O, X, S, D done with this animal.
Measured through the page's catalogue replay (4 eval days, held out): 611 decisions for 2 167 photos (153 per day; stack
by stack 556, +10 %), 19 automatic first animals, 231 "same whale?" (154 yes = 67 %; val 82 %, train 74 %), 11 "next
animal?" (9 new), 369 "which animal?" (of 317 stacks named: first choice 57 %, top three 91 %), runs 316 of 343 stacks
the same whale (92 %), 2 163 of 2 167 photos with the catalogue's animal (99.8 %), 993 looks against 948. Val 332 / 323
decisions, train 2 204 / 2 093. All 28 days: 3 147 decisions for 11 106 photos (112 per day).
Reading. The order does not save work (3-10 % more key presses, the same comparisons); it changes its kind: 4 decisions
in 10 are a yes / no on one animal, the naming comes last with every animal's photos present, and no screen is empty.
The "clear lead" class of round 1 is gone from the page: with the confident stacks taken by "same whale?" it is 25
decisions on the eval days and right 60 %. What would cut the count is unchanged: a link model that holds across
surfacings. The saved "no" answers (stack, animal, link score) are the hard negatives for it.
tests/labeling_e2e.py: 56 checks, pass on 2025-02-27, 2024-12-18, 2025-01-02, 2025-03-14; `--frontend DIR` runs it on a
copy of frontend/ (the live page serves frontend/ from disk; this round was built in a copy and copied over).

## 2026-11-11 late — Photo Labeling round 4: the researcher's first session with the animal-by-animal step (five asks)

The user sorted 2024-12-23 group 1 on the new step and sent five things while working. All in Whale-Review-Web.
1. Confirmed photos should join the left panel: the animal's strip now holds all its photos (best comparable first,
   the ones the last decision added marked "added", the rest as taken); before it showed one photo per stack.
2. "Sometimes the photo is very blurry": the page shows the pipeline's 2560-px copy (convert_raw: half-size decode,
   resized) of 5496-8192 px frames. Main whale box in the copy on 2024-12-23: median 724 px, 25 % under 393 px, 10 %
   under 214 px; the whale view enlarges up to 1.6x, so a quarter of the whales were stretched. New route
   /img/labeling/<day>/hires/<key>.jpg cuts the whale (box + margin) from the RAW with the pipeline's own decode
   (rawpy use_camera_wb, full size; 0.76-0.98 s, cached), capped at 3200 px: 3.2x the pixels for a 151-px whale, mean
   colour shift to the copy under 1 level, so it is drawn over the copy without a seam. Fetched only when the view
   stretches the copy (> 0.9 screen px per copy px). The embedded full-size JPEG of the RAW was faster (0.03 s) but is
   the camera's rendering and, for the CR2 files, a slightly different crop: not used.
3. Removing an animal after a mislabel: "remove animal" (photos back to the open stacks, one undoable patch) and "this
   stack out" for a single wrong stack; freed numbers are used again; animal ids are no longer reused in a session
   (after a merge the next new animal took the merged one's id: an undo or a kept "no" could meet the wrong animal).
4. Review -> Animals "does not load a left-hand image": not reproduced on a copy of the saved day with the live code
   (round trip, the merge in Review, the automatic first animal of group 2 all load). Two defects found on the way
   and fixed: the pane answered a second request for a photo still loading with the photo drawn before (old photo
   under the new box), and a photo that failed to load left the pane blank with no retry (now one retry, then a
   message and click-to-retry; the live server had been restarted under the open page at 20:26). A complete group now
   says so and offers the group in work / the next open group on Enter.
5. `R`: yes with the match marked. A box on the same feature in both photos, saved with the decision in the state
   (`links`, the Pairs page's region form). Purpose: positives with a place to look, the label the link model lacks.
tests/labeling_e2e.py: 67 checks, pass on 2025-02-27, 2024-12-18, 2025-01-02, 2025-03-14.

## 2026-11-12 — Photo Labeling round 5: flukes first, the stack on the left, a fluke > dorsal > other similarity (user, after a day's use)

Asked. Start an encounter with the stacks of flukes ("the easiest to say with certainty new animal"), then dorsal fins,
then the remaining stacks by score; the current stack on the left, the animal it is most similar to on the right with a
similarity score; that similarity recalculated as a combination of fluke, dorsal and the rest, in that priority.
Checked first (scratch simulation on the page's matrices, the catalogue as the researcher; eval days not looked at).
- The order costs nothing: flukes > dorsal fins > rest with the link score = 315 decisions on the val days, the same as
  capture order (315), train 2 141 against 2 153; most new animals turn up in the fluke phase (train 180 of 248).
- The model's parts, pair AUC inside encounters on the val days: fluke cosine 0.67 (0.63 across gaps > 60 s), dorsal
  fin 0.73, body 0.70, the link score 0.78. A fluke is what a person is surest of; it is the embedding's weakest part.
- So each part is calibrated at the stack-animal level on the val days (logistic; fluke: cosine 0.9 -> 0.65, dorsal:
  0.9 -> 0.75; "other" = body cosine, sequence P, gap) and the three are blended over the parts both show. First choice
  right, val days, each scored with the calibration of the two others (276 stacks named): link score 77.9 %, blend
  3:2:1 77.2 %, 1:1:1 76.1 %, 6:3:1 75.4 %, other only 72.5 %, strict fluke-else-dorsal-else-other 72.6 %, a fitted
  logistic on all parts 73.2 %. Train days (embedding in-sample): blend 3:2:1 81.7 %, link score 73.9 %. Taken: 3:2:1.
- A low similarity is no sign of a new animal (top blend < 0.2: new 34 % val, 48 % train; < 0.15: 45 / 52 %): the page
  no longer proposes "a new animal" by itself, it notes that nothing is very similar.
Built (Whale-Review-Web). backend: pairs.bin carries six matrices (link, appearance, fluke / dorsal / body cosines,
sequence P); labeling_model.json `blend` (calibrations, weights) and `order` (measured). lab_core.js: rank() computes
the blend per animal (parts kept for display), nextTask() gives the fluke / dorsal / rest order, upcoming(). The
Animals screen: stack in hand left, most similar animal right, similarity with its three parts, the stack opens on its
fluke (or fin) and the animal is shown by the same part; N brings the next most similar animal. Saves are retried for
a few seconds when the server does not answer (a restart under the open page used to stop it).
Measured through the page's catalogue replay, eval days (held out): 542 decisions for 2 167 photos (136 per day), first
choice right 323 of 470 named (68.7 %), top three 95.5 %, 72 new animals (49 in the fluke phase, 23 dorsal, 0 rest),
runs 317 of 343 (92 %), 2 163 of 2 167 photos with the catalogue's animal, 938 looks. By phase, first choice: flukes
61 %, dorsal fins 71 %, rest 90 %. Against the earlier orders on the same days: capture order 556 decisions / 948
looks, animal by animal (2026-11-11 evening) 611 / 993. Val 315 (323, 332), train 2 017 (2 093, 2 204).
Reading. The researcher's order is also the cheapest of the three measured. The weak point is the fluke phase (first
choice 61 %): the animals are few and new there, and the fluke embedding is poor. A fluke-specific embedding (same view,
underside pattern) is the model work this points to; the `R` region links and the saved "no" answers are its labels.
tests/labeling_e2e.py: 68 checks, pass on 2025-02-27, 2024-12-18, 2025-01-02, 2025-03-14.

## 2026-11-12 later — Photo Labeling: two animals side by side before a merge (user)

Asked: beside Review's "merge into…" a "compare with…" that pulls two animals up side by side in the Animals step,
the animal treated as the stack, the one picked as the first candidate, the others in the order of similarity, yes =
merge. Built in Whale-Review-Web (frontend only): Review sets S.compare, the Animals view builds a unit from all the
animal's stacks (rank() over the set = animal-to-animal similarity, the same fluke 3 : dorsal 2 : other 1 blend), space
merges into the animal shown (mergeAnimals carries the "not this animal" answers over to the animal that stays), N
keeps the two apart (refuseAnimals: a "no" on the animal's first stack; the merge hints skip such pairs), R merges with
the match marked, esc returns. tests/labeling_e2e.py: 72 checks, pass on 2025-02-27, 2024-12-18, 2025-01-02, 2025-03-14.

## 2026-11-12 afternoon — the researcher's first full day on the Photo Labeling page (2024-12-23) against the catalog
Request (user): "I just did a day 2024-12-23 as a test run of the new labeling research flow. Can you compare my results
to what the catalog ground truth says and highlight what I mislabeled or failed to consolidate into one animal? I want
to find where my errors were so that we can brainstorm ideas on how to improve". Read-only analysis of Whale-Review-Web
data/2024-12-23/labeling.json as saved at 14:48 (rev 48; snapshot, scripts, logs and contact sheets in
experiments/labeling_runs/2024-12-23/). Nothing on the page or in the state was changed.
Outcome. No photo was put with a wrong animal: each of the researcher's 10 animals holds photos of one catalog animal
only. All errors are failures to consolidate: the catalog has 6 animals (singer G01A01 x30; pair G02A01 x20, G02A02 x4;
mother G03A01 x30 with calf G3A02 x3; passer-by O01 x8), the researcher made 10. Four splits, 10 of the 93 sorted photos
in a split-off piece: (1) singer: photos 14-15 (09:31:53, 09:32:46) as a second animal next to 27 photos; (2) pair:
photos 30, 31, 35 as one animal and 34, 38-53 (17 photos) as another, both G02A01; the true second whale (4 photos) was
found; (3) mother: photos 54, 59, 83, 84 as one animal and 24 photos as another; the calf was found; (4) photo 85
(12:24:45, the mother again 37 min later) as animal 1 of a new group, behind a group cut the page proposed (P 0.49) and
the Encounters step left in place (cuts null). The two rejected photos are a blurred distant whale (16) and a drone
(86): the catalog files them under the encounter's animal, the rejections are sound. At 14:57, after the snapshot, the
researcher merged split (3) with the page's new side-by-side merge view; three remain.
The decisions (48 logged steps, about 45 active minutes, median 43 s per step). Of the 7 "new animal" answers that were
not automatic, 5 were wrong (two of them undone later by a merge), and in all 5 the page's first candidate was the right
animal, with similarity 0.55 to 0.72; the 2 right ones had 0.51 and 0.09. No "same whale" answer was wrong. The errors
go one way: when unsure the researcher starts a new animal, which keeps the animals pure and leaves the consolidation
undone.
What the photos show (sheets split_*.jpg, fins_*.jpg). Pair: the fin outline of the two pieces is the same at fin scale
(falcate, blunt tip with a small nick); what differs is the lighting and the wet or dry look of the back, the distance
and the phase of the surfacing (flat pass against arching before the dive). It was the first decision after a 17-hour
break. Mother: the second animal was started from the fluke photos 73-77 held against an animal known only by a
broadside dorsal photo (54): no part in common, so "new" was the only possible answer; the page's fluke-first order
makes this by construction whenever the encounter's automatic first animal is a dorsal stack. Later the flat broadside
fins (54, 59, 83, 84) were linked to each other with boxes and the rest (pectoral fins, the back seen from behind,
flukes) to each other; photo 80 in the large piece shows the same hooked fin as 83 and 84 from the same side but was
sorted by its neighbours in time, never held against them. Singer: photos 14-15 show the other flank, flat and far, with
a low blunt fin; the 27 others show the arched back from the left or from behind with a pointed fin. From the fin alone
this one is not safely decidable; the field log's group type (a singer is alone) decides it.
What the page had and did not use. Its merge hints (lab_core.js mergeHints, Review step) only show animal pairs whose
link score is at least 0.85; the four true pairs score 0.74, 0.61, 0.55 and, across the false cut, 0.46, so none was
shown. Ranked instead of thresholded, the three splits inside an encounter are pairs 1, 2 and 3 of the 7 animal pairs
inside encounters, and over all 45 animal pairs of the day the four are at ranks 1, 2, 3 and 8. The group type was never
entered (it would have capped singer, pair and mother-calf at 1, 2 and 2). No frame of the pair's encounter shows two
whales, so co-occurrence says nothing here. One day, a train day (the student embedding is in-sample), a first session:
a lead for what to build, not a measurement.
Ideas for the brainstorm, not built: (a) a ranked merge check at the end of every encounter (the closest animal pairs
side by side, most similar first, a few questions) in place of the 0.85 threshold; (b) a second look before a "new
animal" is created while the first candidate scores above about 0.5: both fins or flukes at the same scale and heading;
(c) the group type from the field log as a soft cap on the animals of an encounter; (d) no "same whale?" question when
the stack and the animal share no part: park the stack until a dorsal-to-fluke series or a later stack bridges, or sort
dorsal stacks before lone fluke stacks; (e) no automatic first animal behind an uncertain cut: ask "seen earlier today?"
with the closest animals of the earlier encounters (the catalog's "same as"); (f) for the model, a fin-outline channel
that survives posture, flank and lighting. (a) and (c) would each have caught all three splits inside encounters on this
day; each needs the catalogue-replay simulation with a researcher who splits by view before it is built.

2026-11-12, same day: the dashed box around the detected whale became four corner brackets set 4 px outside the box
(lab_pane.js, all panes), with `B` / "marks" on the Animals page to hide them: the researcher found the line beside a
fin edge made the edge harder to compare. tests/labeling_e2e.py 73 checks.

## 2026-11-12 late afternoon — Photo Labeling: merge hints by animal-to-animal similarity; the group type; where questions lack a shared part
Request (user, after the comparison of the 2024-12-23 test run with the catalog): how is the group type determined;
"improve the merge hints, determine a threshold that seems more reasonable, its a hint after all it can be ignored"; a
merge check in Review where "full animals can be compared against each other collectively by fluke, dorsal, and other";
what a "second look" would be (discussion); saving a question until there is a shared part "may be handled by our new
fluke, dorsal, other order". Built: the merge hints (Whale-Review-Web frontend only, no backend restart). Measured: the
rest. Work in experiments/labeling_runs/merge_hints/ (fetch.py reads every day's payload and pair matrices from the
page's API; sim.py, sim2.py, user_day.py, shared_part.py, second_look.py; logs).
Outcome. (1) Merge hints now compare whole animals: similarity = every stack of the smaller animal against the whole
other animal by the blend of step 2, averaged (animalSim); a pair is hinted at 50 % or more, or at 25 % or more when
each is the other's most similar animal of the encounter; the row shows fluke / dorsal / other and opens the
side-by-side view. On the 2024-12-23 run as saved at 14:48 this shows exactly the three split whales (28, 32, 37 %) and
nothing else; the earlier rule (link of the closest pair of stacks >= 0.85) showed none. (2) Simulated on the labelled
days, two split whales in three are shown for about one wrong hint per encounter; on the eval days the gain over the
earlier rule is small. (3) Under the page's current order a stack meets its own animal without a fluke or dorsal fin in
common in about 1 % of the questions, a third of what capture order gives: the order handles most of it. (4) Correction
to the entry above: the test run was sorted with the earlier animal-by-animal order (its log has "collect" steps until
14:48), so the fluke stack held against a dorsal-only animal came from that order, not from flukes-first.
The score. First simulation (sim.py): every catalogue animal sorted right except one left in two pieces (fluke stacks
apart, one fin side apart, apart at the longest gap, one stack apart, one to three stacks apart), against pairs of
different animals of the encounter. AUC of the candidates on the val days for pieces of one to three stacks: link score
of the closest pair 0.82, best match per part (the blend over the two sets, what the side-by-side view shows) 0.81,
best-match mean per part 0.85, the stack average 0.87 (0.91 for single stacks); train days 0.90 / 0.85 / 0.93 / 0.95.
The stack average also gives far fewer wrong hints at a given share of split cases shown. But the three real pairs of
the test run score 0.28 to 0.37 on it, below any threshold that simulation would pick (0.40 to 0.50): a researcher
splits where the pieces look different, random splits are easier.
The threshold. Second simulation (sim2.py): the catalogue plays a researcher who never joins two whales (as in the test
run: no wrong "same") and starts a new animal for a stack of a whale already there with P = 0.2 x clip((0.75 -
similarity) / 0.5, 0, 1), stacks in the page's order; this gives 1.65 (val), 1.78 (train), 1.83 (eval) animals per
catalogue animal, the test run had 1.67. Rules compared by the share of split whales with a true pair shown and the
wrong hints per encounter (5 runs per day). Val: link >= 0.85 (before) 38 % at 0.62; similarity >= 0.40 52 % at 1.08; >=
0.25 78 % at 3.01; >= 0.50 or >= 0.25 when each is the other's closest (chosen) 65 % at 0.71, 58 % of the hints right,
no encounter with more than 3 wrong; the same rule on the best-match blend 55 to 62 % at 0.7 to 1.0. Train: before 52 %
at 0.87, chosen 69 % at 0.85. Eval, read once after the choice: before 55 % at 1.08, chosen 60 % at 1.18 (half the hints
right). So the mutual-closest clause is what reaches the low-scoring real splits without flooding large groups; a flat
low threshold costs four times the wrong hints for a fifth more reach. The hints across encounters (appearance >= 0.85,
"same as") are unchanged: the one such case of the test run (one photo behind a false cut) scores below several wrong
pairs on every score tried.
The page (Whale-Review-Web frontend/js/lab_core.js blendOf, animalSim, MERGE, mergeHints; lab_review.js hint rows;
labeling.css; backend/labeling_model.json "merge" with the numbers above; tests/labeling_e2e.py five checks, 78 in all,
pass on 2025-02-27, 2025-01-02, 2024-12-18; label_assist.py fit keeps the merge block). The backend was not restarted
(another session was editing backend/labeling.py), so the page runs on the defaults in MERGE, the same numbers; the
model file takes over at the next restart. Installed as exact string patches on the live files, tried first on a private
copy of frontend/ with a copy of the 14:48 state.
Group type. In the catalog it is the tail of the collection set name ("<day> Group N <type>"): 33 MCE, 30 CG, 18 Pair,
15 MC, 8 Single, 6 Singer, 2 MYE, 2 Trio and 4 free-text ones on the 28 days. Animals per group by type: Single 1 (8 of
8), Singer 1 (5 of 5 photographed), Pair 2 (15 of 18; 1 twice, 3 once), MC 2 (12 of 15), MCE 3 (28 of 33; 2, 4, 5), Trio
and MYE 3, CG 2 to 20 (median 6). So the type caps the animals of 85 to 100 % of the groups that are not competitive
groups: a soft cap. Whether the type is written in the field log at the time or settled later from the photos is not
known here (the catalog only holds the result); the page today proposes a type from the animals found ("type: Trio ?" on
the test run's mother-calf group with three animals), which cannot catch a split. The user will ask the researchers.
Shared part (shared_part.py; catalogue as the researcher, per catalogue encounter). Stacks whose whale is already an
animal when they come up, with no fluke and no dorsal fin in common with any stack sorted to it so far: page order 1.5 %
(val, 6 of 390), 0.9 % (eval), 0.8 % (train), always a dorsal stack meeting an animal known by flukes only, 7 to 12 % of
the whales have one such moment; capture order 3.8 / 3.0 / 2.7 %, a quarter of the whales. Another 2 to 5 % share a
dorsal fin only from the other side. On 2024-12-23 there is no such stack under either order. Parking was not built: the
order leaves about 1 %, and such a pair now comes back as a merge hint.
Second look (second_look.py; for the discussion, not built). With the page's similarity of a stack to its most similar
animal: a bar at 0.5 would stop 11 to 18 % of the true new whales for a second look (about three a day on the eval days)
and would cover the half of the true matches that score above it; a bar at 0.4: 16 to 34 % and two thirds. The test
run's splits scored 0.33 to 0.52 as whole animals, so a second look at the moment of the answer covers at most half of
such errors; the merge hints now cover the same pairs later with all of each animal's photos.

## 2026-11-12 — view class: fluke / dorsal fin / other straight from the pixels (user: no mesh needed, just find flukes and dorsal fins so the stacks that show them come first)

Asked: instead of the 3D mesh mapping, is a plain classifier enough to say confidently which photos show a fluke,
which a dorsal fin, and call the rest "other", to order the Photo Labeling stacks? (A second session built the order of
work itself the same day: flukes, dorsal fins, the rest. This entry is the labels that order runs on.)
Before: the page took the fin / fluke flag of an instance from the parts_hr charts, i.e. from the synthetic-trained
canonical map (fin = >= 400 px in the fin band of the body coordinate, fluke likewise). Against a blind by-eye random
sample: three-way accuracy 0.875 (train + val, 368) / 0.841 (eval, 132); it calls a fin on almost any back (precision
0.72-0.79), misses a fifth of the flukes (recall 0.80-0.81) and has almost no "other" (3 % of instances).
Built: model_fine_tuning/cluster/view_class.py (features | predict | fit | report), model
experiments/safe_clusters/view_class.pkl (view_class_v2), per day DAY/safe/view_feat.npz and DAY/safe/view_class.json
(instance key -> P fluke, P dorsal fin); experiment folder experiments/view_class (README there has every table).
Frozen DINOv3 ViT-B/16 cls + mean patch token of two crops of the instance box (448-px square; a shape-keeping crop,
long side up to 672), PCA 128, RBF SVM per answer (fluke yes / no, fin yes / no), probability by a logistic on
out-of-fold scores of a random sample. Nothing fine-tuned; 10-20 s of GPU per day.
Labels: 945 instances labelled by eye from contact sheets (by Claude, not the researcher; marginal cases marked and
kept out of loss and accuracy): 432 random on the 24 train + val days, 144 random on the 4 eval days (blind, before any
score), 216 hard cases, 151 audit tiles; plus the researcher's keypoint clicks of the Pairs / Bursts pages (627
instances on three days; a click is attributed to the instance whose box holds it) and the 296 photos of the
catalogue's fluke collections. 1 360 labelled instances on train + val train the model. On the 39 instances with both
a by-eye label and researcher clicks: absent agrees 36 / 36; of 36 clicked parts the eye said yes 28, marginal 4, no 4.
Result, three-way accuracy on clear cases: train + val random sample, day-grouped CV 0.967 (356 / 368); eval random
sample 0.970 (128 / 132; per day 35 / 35, 32 / 34, 31 / 32, 30 / 31). Eval per answer: fluke precision 0.971, recall
0.985; dorsal fin 0.965 / 0.982; at P >= 0.9 both 100 % precise (59 flukes, 54 fins; recall 0.88 / 0.96). Against the
researcher's clicks + catalogue flukes of the eval days: 93 of 97 flukes, 113 of 117 fins, one false call. "Other" is
the weak class (8 of 10 on eval, 27 of 32 in CV): pectoral fins and tail stocks called flukes, humps called fins.
Eval days were read twice for this model (first version: logistic on the cached 224-px inst_feat, 0.939 on the same
sample; then the final one); the recipe was chosen on the train + val CV (0.890 -> 0.927 on the hard-case-heavy set).
Synthetic data (user asked mid-way whether both are used): tested, not in the shipped model. The 16 917 run04 + run05
shots with exact labels from the render metadata, same features (vc_synth.py): synthetic only 0.62 / 0.70 three-way
on real photos (all labels / random sample; 0.72 / 0.82 with each domain centred), real + synthetic within +-1 point
of real only at today's label count (first version 0.890 / 0.954 -> 0.888-0.893 / 0.957-0.965; final recipe 0.927 /
0.962 -> 0.916-0.924 / 0.948-0.967); with 50-100 real labels it is worth +2 to +3 points (N 50: 0.806 -> 0.839).
Synthetic data still feeds the step before: the masks, hence the instance boxes, come from the synthetic-trained models.
On the page (Whale-Review-Web backend/labeling.py: view_class.json replaces the chart's fin / fluke bits when the day has
one; WHALE_VIEW_CLASS=0 switches back; fin side / fluke face still from the charts): 866 of 5 040 stacks change phase
(chart fluke 2 092 / fin 2 581 / rest 367 -> 1 988 / 2 349 / 703). Blind audit of 54 changed stacks on the eval days:
the classifier right on the disputed part 39, the chart 9, 6 marginal; a coin flip (8 : 7) where the classifier finds
a part on a stack the chart had as neither, 31 of 33 elsewhere. Catalogue replay (new tests/labeling_replay.py; chart
-> classifier -> classifier with gated similarities): decisions eval 542 -> 541 -> 541, val 315 -> 312 -> 311, train
2 017 -> 2 030 -> 2 030; first choice right eval 0.696 -> 0.696 -> 0.717, val 0.775 -> 0.795 -> 0.805, train 0.813 ->
0.814 -> 0.822; looks eval 929 -> 925 -> 902. Gated similarities (WHALE_VIEW_GATE=blend, default): the ranking's
fluke / dorsal similarities only count instances the classifier says show the part; the link model's inputs are
untouched (gating those too, =all, gives 3-5 % fewer decisions through longer runs at run precision 0.907 against
0.922 on eval: needs the link model refitted, not shipped). The gating variant was picked with the eval replay in
view. Merge hints (thresholds of the same day, sim2.py on the new inputs): unchanged within noise (split whales
reached eval 0.60 -> 0.60, train 0.69 -> 0.71, val 0.65 -> 0.58; wrong hints per encounter 1.18 -> 1.21, 0.85 -> 0.87,
0.71 -> 0.78). tests/labeling_e2e.py: 78 checks pass on the four days with the classifier on; bursts_e2e passes.
Reading: the simpler problem is solved to the level the order of work needs; it does not by itself cut the number of
decisions (the order costs nothing and saves nothing, as measured when it was introduced). What it changes is that
the fluke phase holds flukes, the fin phase fins, and the "rest" pile is real and twice as large. The lever on the
decision count is still the similarity inside a phase: the fluke similarity is the embedding's weakest part, and a
fluke-specific matcher now has a clean input set (4 058 instances at P >= 0.9, with the 296 catalogue flukes among them).
Not done: the researcher has not checked the by-eye labels; no box of where the fluke / fin is (the classifier says
whether, not where); the labeling_model.json "order" numbers are still the chart-flag ones; live server not restarted.
Aside: backend/pairs.py thumb() writes through a fixed temporary name, so two processes (or two requests) making the
same thumbnail at once can fail one of them with FileNotFoundError (seen with two test servers in parallel).

## 2026-11-12 afternoon — Whale-Review-Web redesign: Library, Model Interpretability, Labeling Assistant (user)

Asked: a clean Lightroom-like Library for peeking through a day's photos (the old model / label tools out of it), the
"Review Excluded" module repurposed as "Model Interpretability" (a flowgraph of how a day is processed, every step a tab
with a plain explanation, inputs, outputs and interactive visuals: masks, 3D clustering with popups), and "Photo
Labeling" renamed "Labeling Assistant". The user asked for the lead session to plan and review and for Opus sub-agents
to write the code; three ran in parallel on disjoint files against one written brief.
- Library (`/`): any processed day; grid / loupe / compare / survey, filmstrip, Navigator, Catalog, Folders, Collections
  (read-only mirror of the Labeling Assistant's sorting), Histogram, camera Metadata; no model output. backend/library.py,
  frontend/js/lib_*.js, css/library.css, tests/library_e2e.py (92 checks).
- Model Interpretability (`/interpret`): overview flow diagram with the day's numbers, nine steps: photos, whales
  (outline wipe sliders), body map (photo coloured by body position linked to a 3D standard whale), parts (cut-outs,
  walls of every fluke / fin), fingerprints (3D cloud of the part embeddings, PCA or t-SNE, hover = the whale, click =
  five look-alikes, colour by stack / time / own sorting / catalogue), stacks, encounters (clue breakdown per pause),
  similarity (heat map + the 3:2:1 arithmetic), workflow. backend/interpret.py + interpret_models.py,
  frontend/js/interp*.js, tests/interpret_e2e.py (63 checks) + interpret_models_e2e.py (43).
- backend/app.py slimmed: no day loaded at start-up, routers mounted when present; the old review app (model clusters,
  approve / exclude, pose editor) moved to attic/whale_review_v1/, nothing deleted. /pairs, /bursts, /signatures answer
  but are no longer linked. Module picker on every page: Library | Model Interpretability | Labeling Assistant.
All four suites pass together (library, interpret, interpret_models, labeling). One thing the page makes visible on
labelled days: with the cloud coloured by the catalogue, 81 % of the dorsal-fin cut-outs of 2025-02-27 have the same
animal as their closest look-alike from another stack.

## 2026-11-12 evening — view-aware matching: what the fluke / dorsal label does for the match score and the merge hints (user: is it in the assistant now; can it improve the match score, the suggestions in Animals and the merge suggestions in Review?)

Status. Live since 16:39: another session restarted the review server at 16:38:50 with backend/labeling.py as it
stood (the view classifier's labels, gated similarities, match.bin); the frontend patch (lab_core.js, two edits) went
onto the live files at 16:39:10. Checked read-only on port 8765: 2025-02-27 view_model view_class_v2, ranking on
match.bin, phases fluke 32 / fin 78 / rest 46; 2024-12-23 (a train day) classifier labels, ranking on the part cosines.
Work in experiments/view_match (README there has the tables); new module model_fine_tuning/cluster/view_match.py.
Which comparisons carry information (stack pairs inside catalogue encounters, AUC of the student's part cosine): fin
cosine over every pair with two fin charts 0.724 (val), over pairs where the classifier says both show a fin 0.779,
over the rest 0.658 (train 0.779 / 0.829 / 0.694); fluke on train 0.717 / 0.756 / 0.652, on val too few to tell. A
third of the fin comparisons were fin against no fin. 11-13 % of the stacks that show a fluke have no fluke chart.
View-specific similarity: for the instances that show a part, the classifier's frozen features centred on the day,
PCA-whitened (128), within-animal whitened (shrink 0.3), reduced by within-day LDA to 32 dimensions; fitted on the
catalogue animals of the 21 train days (fluke 2 937 instances / 233 animals, fin 4 133 / 269), one projection per train
day without it. Stack pairs out of fold, val days (neither model saw them): fin 0.811 against 0.777 for the student
(0.823 / 0.770 beyond 10 min; the gap alone 0.56), fluke 0.799 against 0.657 on the 283 pairs both score. Train days
(student in-sample): fin 0.744 / 0.836, fluke 0.744 / 0.757.
On the page (catalogue replay, first choice right; decisions unchanged at 541 / 311 / 2 030): chart flags -> classifier
labels -> + gated similarities -> + fin similarity blended (shipped): eval 0.696 -> 0.696 -> 0.717 -> 0.751, val 0.775
-> 0.795 -> 0.805 -> 0.831, looks eval 929 -> 925 -> 902 -> 873; all 7 held-out days up at the last step (eval 56, 141,
75, 67 -> 58, 149, 78, 70 of 71 / 208 / 97 / 97). The gain is the dorsal phase (eval 167 -> 182 of 245); flukes 114 ->
115 of 168. Blending the fluke part too: val 0.824, eval 0.740 (fluke phase 110 of 168): no gain, so the fluke part
stays the student's (WHALE_VIEW_MATCH_FLUKE). On the train days the blend is worse than the in-sample student alone
(0.798 against 0.822): view_match.npz carries student_in_sample and the page leaves those days as they were.
Mechanics: DAY/safe/view_match.npz (cluster/view_match.py embed) -> backend match.bin = [fluke, dorsal] codes on the
scale of the student cosine (the two calibrated logits averaged where both exist; the new one's calibration: slope
from 24 days out of sample, intercept shifted to the val base rate, the scale the page's own calibration was fitted
on) -> lab_core.js rank() reads S.F1 / S.D1 when the day has them. pairs.bin is unchanged byte for byte.
Merge hints (Review): not changed. With the new similarity the hint score ranks fixed split cases better (AUC eval
0.879 / 0.898 / 0.533 / 0.922 / 0.907 -> 0.877 / 0.899 / 0.561 / 0.926 / 0.921 for the five kinds, val up on all five),
but the three real split whales of the 2024-12-23 test run fall from 0.34 / 0.32 / 0.28 (all hinted) to 0.28 / 0.19 /
0.19 (one hinted), and with sim2's researcher (splits where the page's similarity is low) the split whales without a
hint go 14.6 -> 18.4 per eval run when page and hints share the blend, 15.6 with the blend on the page and the
student in the hints. A hint is worth most when it does not come from the score that produced the split.
"No view in common" as a hint class: such pairs of pieces hold 23-34 % of the simulated true split pairs and the rule
shows almost none; a class for them (each the other's closest such piece, similarity >= 0.15) adds 0.7-0.8 hints a
day, 50-61 % right, reaching 3-5 % more split whales. Too small; not built.
Tests: tests/labeling_e2e.py 78 checks pass on the four days against a private copy of frontend/ and, after the
patch, three times against the live files; bursts_e2e passes. Some runs in between timed out on a save while other
sessions' servers and tests were running (ports 8801-8805); re-runs on other ports pass every time.
Eval days: read three times for the similarity (both-parts blend with two calibrations, then the shipped setting);
the fin-only choice was made knowing the first read; val alone points the same way.
Open: flukes. Pair AUC says the fluke-specific similarity is better, the fluke phase's first choice does not move
(68 % on eval): a fluke matcher that aligns the blades (local features, or a fine-tuned embedding on the 4 058
confident fluke crops) is the next lever. Hints from both scores (more reach, more wrong hints) is a threshold choice
for the researcher.

2026-11-12, later: contact sheets out of the commits (user's request). The three commits of the day (7528b05, a39a49a,
29e9549) were rebuilt without the 109 images of experiments/{view_class,stage2_labels,stage2_linker}/sheets (new
79f8b44, 889e296, f4ec975; same authors, dates and messages; nothing else differs); local main points at f4ec975, the
images stay on disk, untracked and ignored (.gitignore in each sheets folder). The push to GitHub needs the user's
credentials (`git push --force-with-lease origin main`); pull.ff=only is set in this repository so that a pull or
Sync cannot merge the old commits back before that. The seven sheets of experiments/labeling_runs/2024-12-23 (whales,
a drone, shoreline; no people) and docs/outline_pairs were left in.
