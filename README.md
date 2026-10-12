# Dinov3-WhaleID

Fine-tuning DINOv3 on synthetic humpback renders from
[Whale-Image-Data-Factory](../Whale-Image-Data-Factory) for two tasks:

1. **Segmentation** — which pixels are visible whale (above water, not behind spray).
2. **Body coordinates + 3D pose** — for those pixels, *where on the whale* they are
   (fraction along the body from fluke to snout, and angle around the body axis), and from
   that the rigid rotation and image scale that align the canonical 3D whale mesh to the
   photo (via PnP on the dense predictions, plus a direct rotation head as a second opinion).

3. **Canonical surface points** (`model_fine_tuning/canon/`, the successor of 2 for
   point-to-point matching) — for every whale pixel, the 3D point on the undeformed whale mesh
   it shows, predicted *folded* (x, |y|, z): which flank is left to a single per-whale decision
   afterwards (`geometry.resolve_side`: two global hypotheses, alternating PnP + surface-normal
   facing vote, a prior from the map's own head/dorsal directions or the rotation head), so the
   network never guesses left/right pixel by pixel. Non-degenerate on the flukes and fins,
   unlike the angle in 2. Points of interest are read off the map. Every crop is also run
   mirrored and the disagreement is reported as a confidence. (`train.py --real-images` adds the
   same mirror consistency as a label-free loss on real photos; on its own it collapses to
   constant predictions on real inputs, see LOG.md 2026-10-12, so do not use it without a
   stop-gradient teacher or labels.)

4. **Clustering a day's photos by individual** (`model_fine_tuning/cluster/`): every whale
   pixel is mapped to one of 512 surface cells (`cells.py`); per photo the DINOv3 patch
   features and colour of each visible cell are stored, signed by the resolved flank
   (`atlas.py`); pairs of photos are scored by how alike they look *at the same body
   locations* (per-cell z-scored similarity, under both side hypotheses) plus a capture-time
   term (`match.py`); average-linkage clustering with a threshold self-calibrated from bursts
   and far-apart pairs, scored against the Lightroom labels which are never used otherwise
   (`cluster.py`); heatmaps of why two photos were linked (`explain.py`). Generic DINOv3
   features carry almost no identity (pair AUC ~0.6); `identity.py` learns a projection from
   the day's timestamps alone (same spot seconds apart = same whale, hours apart = different),
   which is what makes within-group separation possible. `local.py` (patch-level aligned
   matching, SIFT) and `fin.py` (dorsal-fin crops) are kept as negative results.

All are DINOv3 ViT-B/16 backbones with the last blocks unfrozen under layer-wise LR decay.
The body-coordinate task uses the same backbone type on purpose: dense correspondence to a
canonical surface is what DINOv3 patch features are best at, and the 3D fit is done by
classical geometry on top of them, not by a separate pose network.

## Layout

```text
model_fine_tuning/
  common/        backbone loading & unfreezing, dataset/crops/augmentation, geometry (body
                 table, PnP, rotations), metrics, splits, visualisation
  segmentation/  model.py, train.py, evaluate.py, predict.py
  bodycoords/    model.py, train.py, evaluate.py, predict.py   (u, angle) head, kept for comparison
  canon/         model.py, train.py, evaluate.py, predict.py   canonical surface points (current)
  cluster/       cells, atlas, match, identity, cluster, explain (+ local, fin)   day clustering
  assets/        humpback_body_table.npz ((u, theta) -> 3D mesh point), humpback_folded_table.npz
                 (folded mesh vertices + normals, `canon.train --build-table`), landmarks
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
test_scripts/bash/canon_pipeline.sh canon_r02_ft4        # canonical surface model: train, evaluate, real photos
python -m model_fine_tuning.canon.predict --seg-checkpoint SEG --checkpoint CANON --images DIR --out OUT
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

The canonical-surface model needs `canon/<id>.npz` in the run (factory
`postprocess --canon-only --workers N` adds it to an existing run). Real photos in camera RAW are
decoded with `test_scripts/python/convert_raw.py` (rawpy).

Frames and conventions are documented in `model_fine_tuning/common/geometry.py`; note the
Unreal frames are left-handed, which the PnP step has to account for (the unit tests check
that PnP on the ground-truth coordinates recovers the ground-truth pose).

## Earlier work

The classical-vision isolation and clustering code from the first iteration lives in git
history (`helpers/`, `notebooks/`, `initial_work_old/` at commit a372690).
