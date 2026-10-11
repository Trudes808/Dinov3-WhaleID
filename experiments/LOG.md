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
