# Dinov3-WhaleID

Fine-tuning DINOv3 on synthetic humpback renders from
[Whale-Image-Data-Factory](../Whale-Image-Data-Factory) for two tasks:

1. **Segmentation** — which pixels are visible whale (above water, not behind spray).
2. **Body coordinates + 3D pose** — for those pixels, *where on the whale* they are
   (fraction along the body from fluke to snout, and angle around the body axis), and from
   that the rigid rotation and image scale that align the canonical 3D whale mesh to the
   photo (via PnP on the dense predictions, plus a direct rotation head as a second opinion).

Both are DINOv3 ViT-B/16 backbones with the last blocks unfrozen under layer-wise LR decay.
The body-coordinate task uses the same backbone type on purpose: dense correspondence to a
canonical surface is what DINOv3 patch features are best at, and the 3D fit is done by
classical geometry on top of them, not by a separate pose network.

## Layout

```text
model_fine_tuning/
  common/        backbone loading & unfreezing, dataset/crops/augmentation, geometry (body
                 table, PnP, rotations), metrics, splits, visualisation
  segmentation/  model.py, train.py, evaluate.py, predict.py
  bodycoords/    model.py, train.py, evaluate.py, predict.py
  assets/        humpback_body_table.npz ((u, theta) -> 3D mesh point), landmarks
  splits/        run01.json  train/val/test ids (70/15/15, stratified by pose category)
test_scripts/
  python/        unittest modules (geometry conventions, data invariants, model shapes)
  bash/          run_unit_tests.sh, smoke_train.sh, train_*.sh, evaluate_test_split.sh,
                 predict_test_data.sh
experiments/     one folder per run (config, logs, metrics; checkpoints gitignored), LOG.md
```

## Setup

```bash
python3 -m venv --without-pip .venv && curl -sS https://bootstrap.pypa.io/get-pip.py | .venv/bin/python
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install -r requirements-training.txt
```

Defaults (override with environment variables, see `model_fine_tuning/common/paths.py`):
`DINOV3_REPO=~/dinov3` (the facebookresearch checkout), `DINOV3_WEIGHTS` (ViT-B/16
LVD-1689M checkpoint in that repo's `weights/`), `WHALE_RUN_DIR` (factory run with
`dataset/`), `WHALE_TEST_DATA=~/NOAA_whales/test_data_do_not_use`.

One-time per run of the factory:

```bash
.venv/bin/python -m model_fine_tuning.common.splits make --run-dir RUN --out model_fine_tuning/splits/run01.json
.venv/bin/python -m model_fine_tuning.common.build_body_table      # from the humpback OBJ
test_scripts/bash/run_unit_tests.sh
```

## Train, evaluate, predict

```bash
test_scripts/bash/train_segmentation.sh seg_vitb16_ft4 --unfreeze-blocks 4 --epochs 30
test_scripts/bash/train_bodycoords.sh   body_vitb16_ft4 --unfreeze-blocks 4 --epochs 40
test_scripts/bash/evaluate_test_split.sh seg_vitb16_ft4 body_vitb16_ft4          # held-out test split
test_scripts/bash/predict_test_data.sh   seg_vitb16_ft4 body_vitb16_ft4          # photos in test_data_do_not_use
```

`predict_test_data.sh` runs the labelled evaluations if the folder is a factory run, and the
photo pipeline (segment -> crop -> body coordinates -> PnP pose) otherwise. For real photos
pass `--focal-mm` and `--sensor-width-mm` (or `--hfov-deg`) so the translation and scale are
in metres; the rotation does not need them.

## Data the models learn from

Per rendered frame (`RUN/dataset/`): RGB, visible-whale mask, body-coordinate image over
the whole whale (R = fraction along body, G = angle/360, B = defined), the 6-DoF pose in
the camera frame, intrinsics, pixels-per-metre at the whale and projected landmarks. The
segmentation model trains on random 512-px crops at near-native scale (all 1400 training
frames, negatives included) and is validated on full 1280x720 frames. The body model
trains on whale-centred crops (1112 frames with >= 300 visible pixels) and is evaluated
with oracle crops (GT mask box) and end-to-end crops from the segmentation model.

Frames and conventions are documented in `model_fine_tuning/common/geometry.py`; note the
Unreal frames are left-handed, which the PnP step has to account for (the unit tests check
that PnP on the ground-truth coordinates recovers the ground-truth pose).

## Earlier work

The classical-vision isolation and clustering code from the first iteration lives in git
history (`helpers/`, `notebooks/`, `initial_work_old/` at commit a372690).
