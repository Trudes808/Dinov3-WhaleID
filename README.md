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
scripts/run04_train.sh    RUN/dataset   # run04: contour-weighted (seg_r04_cw, canon_r04_cw) then plain variants, each evaluated
scripts/run04_evaluate.sh RUN/dataset   # run04: test split, real photos, run03 models on the same split, experiments/run04_report.md
```

Run04 training (2026-10-14) re-weights the losses with the factory's contour label (`contours/<id>.png`: 1 = the
whale's own silhouette, 2 = waterline / foam seam, 3 = calf edge; `common/contours.py`): within 3 px of the
silhouette the segmentation / visibility BCE counts x4 and the canonical-coordinate L1 x3, within 3 px of a seam
the BCE counts x0.3 (`--contour-w-true`, `--contour-w-seam`, `--contour-band`, canon `--contour-w-xyz`; unit
weights = off, the default). The boundary F-score is reported along the whale's own silhouette (`boundary_f1_open`:
the gt boundary away from the seams against the predicted boundary near it, what a mesh fit should match) and at
the seams (`boundary_f1_seam`), plus the recall of the silhouette pixels
(`true_contour_recall`); the `_pooled` versions (match counts summed over the split) are the ones to compare, since
many surfacing whales have almost no open-air contour (an elevated camera sees the far waterline, so the back is
all seam) and their per-image scores are noise. `run04_train.sh` trains the plain recipe on the same data as the ablation.

`predict_test_data.sh` runs the labelled evaluations if the folder is a factory run, and the
photo pipeline (segment -> crop -> body coordinates -> PnP pose) otherwise. For real photos
pass `--focal-mm` and `--sensor-width-mm` (or `--hfov-deg`) so the translation and scale are
in metres; the rotation does not need them.

## Mesh fit (bendable) and the real-vs-synthetic grid

`python -m model_fine_tuning.canon.fit_mesh` fits the pose *and* the body bend of the canonical mesh to a
canonical-surface prediction instead of a rigid PnP: rotation from the direct head, a PnP-independent linear
translation start, robust least squares on the dense correspondences over pose + arch / tail curl / head pitch /
lateral sweep (the factory's `deform.py`, imported from the sibling repo) with a soft prior, then a bounded Powell
refinement on the rendered above-water silhouette (per-column top edge against the mask's top edge, mesh above or
beyond the whale, excess above water; the waterline is never matched). `scripts/fit_real_photos.sh NAME...` runs it
on the real photos (`FIT_MESH` for the 4k canonical mesh, `assets/humpback4k_folded_table.npz` shares the K9239
normalisation), `scripts/fit_synthetic.sh NAME` on run04 test crops scored against the true mask and rotation, and
`canon/fit_grid.py` lays real photos beside synthetic shots of the same pose class (fluke / back / head from the
predicted body fraction) with the rigid PnP (orange), bend (green) and refined (white) outlines. Per fit the json
holds the above-water IoU, coverage, excess, top-edge error and the bend angles for each variant.

Training on several runs at once: `--extra-run-dir RUN --extra-split SPLIT` (ids prefixed by the run index,
`common.data.CombinedRun`). Real-photo mirror consistency: `--consistency-teacher ema` predicts the mirrored branch
with an EMA copy under no_grad (the plain two-branch loss collapsed to a constant map).

## Data the models learn from

Per rendered frame (`RUN/dataset/`): RGB, visible-whale mask, body-coordinate image over
the whole whale (R = fraction along body, G = angle/360, B = defined), the 6-DoF pose in
the camera frame, intrinsics, pixels-per-metre at the whale and projected landmarks. The
segmentation model trains on random 512-px crops at near-native scale (all 1400 training
frames, negatives included) and is validated on full 1280x720 frames. The body model
trains on whale-centred crops (1112 frames with >= 300 visible pixels) and is evaluated
with oracle crops (GT mask box) and end-to-end crops from the segmentation model.

The canonical-surface model needs `canon/<id>.npz` in the run (factory
`postprocess --canon-only --workers N` adds it to an existing run). Run04 adds `contours/<id>.png` (boundary
classes, above) and `instances/<id>.png` (1 = the shot's whale, 2 = the calf; crops are centred on the shot's
own whale, `WhaleRun.primary`). Real photos in camera RAW are
decoded with `test_scripts/python/convert_raw.py` (rawpy).

Frames and conventions are documented in `model_fine_tuning/common/geometry.py`; note the
Unreal frames are left-handed, which the PnP step has to account for (the unit tests check
that PnP on the ground-truth coordinates recovers the ground-truth pose).

## Day clustering, stage one: safe clusters (2026-11-03)

`model_fine_tuning/cluster/safe_clusters.py` + `safe_tracklets.py`: the photographer's runs of shots of one whale,
linked with high precision from the capture sequence and the frames, before any identity matching. Stage one =
tracklets (`cluster/tracklets.py`: bursts of one camera with gaps <= 1.5 s, instances of consecutive photos assigned
on position, size and the student body-embedding cosine, max cost 1.5). Stage two = a gradient-boosted classifier
on tracklet pairs of one camera (at most 3 tracklets apart, gap <= 300 s, no shared photo): the gap, the continuity
between the last and first frames (position, size, focal length, metered exposure, range from focal length over
pixels-per-metre, the canon model's rotation, colour of whale and water), the appearance over all instance pairs
(max / mean cosine per part) and the tracklet sizes; links are chained best first by union-find, one instance per
photo per cluster. Production = the strict tier (`--strict --threshold 0.995`): P >= 0.995 and no link whose end
frame holds several whale instances, which made zero wrong automatic links on all 7 labelled days; weaker candidates
and the closest-looking clusters of one interaction are written as `suggestions` for the reviewer (Whale-Review-Web
/bursts, level "safe clusters"). Fitted on the train days, threshold and settings chosen on the val days.

    python -m model_fine_tuning.cluster.safe_clusters features --day-dir experiments/days_all/DAY     # EXIF, pose, colour per instance -> DAY/safe/photos.json
    python -m model_fine_tuning.cluster.safe_tracklets cluster --day-dir experiments/days_all/DAY --model experiments/safe_clusters/safe_model.pkl \
        --threshold 0.995 --strict --cost 1.5 --skip 3 [--labels model_fine_tuning/splits/labels_all_days.json]   # -> DAY/safe/clusters.json (+ suggestions, audit)
    python -m model_fine_tuning.cluster.safe_tracklets report|sweep --root experiments/days_all --days model_fine_tuning/splits/days.json \
        --labels ... --model ... [--threshold 0.98] [--baselines]                                            # per-day table / threshold sweep

Needs `DAY/cluster_v3/embeddings.npz` (student embeddings: `cluster_day.py channels --sources id:experiments/pair_student_g/checkpoints/best.pt`).
Result of the strict tier (photos in impure clusters, by the catalog's one-label-per-photo): val 4 / 878, eval 4 / 2146,
all four stage-one assignments inside multi-whale bursts, at 8.5 / 9.2 clusters per animal (P 0.98 without the rule:
8 and 22 impure photos at 6.9 / 6.7; 1.5-s bursts: 18 and 79 at 5.7 / 5.5). Multi-whale audit (bursts holding two or
more catalog animals): val 7 of 10 split into pure clusters, eval 16 of 17. Nearly every remaining wrong link is a frame with two whales whose catalog label flips; nearly no
link is made across gaps over 1.5 s at this threshold (same-animal share there is 30-60 % and the appearance AUC
0.75-0.88), so joining the clusters into animals is the job of the part-based stage. See experiments/LOG.md.

## Day clustering: the instance filter and the link scores for the reviewer (2026-11-11)

`model_fine_tuning/cluster/link_scores.py`, after the first day of researcher links (experiments/LOG.md 2026-11-11,
"later" and "night"). Nothing in it is fine-tuned: frozen DINOv3 ViT-B/16 features with small models on top.

- **Not a whale** (`instances`, `not-whale`; model `experiments/safe_clusters/not_whale.pkl`): on wide-angle frames
  the crew on the bow, the deck, gear in the foreground, poles and drones get a whale mask. A logistic regression on
  the frozen features of one square crop per instance, fitted on 150 by-eye labels of 14 train days, lists them in
  `DAY/safe/not_whale.json`; `safe_tracklets --filter` leaves them out of the clustering (and gives the photo's
  catalog label to the largest instance left). Every flag on the val days (34) and on the eval days (11, looked at
  once) is a real non-whale by eye; 210 of 11 810 instances are left out; the strict tier still makes no wrong
  automatic link. Production clusters are made with `--filter` since 2026-11-11 (earlier files: `clusters_v1.json`).
- **Window scores** (`windows`, `look`, `scores`): a researcher ties two clusters through one small area (a fin, a
  scar, barnacles on a fluke tip). Every instance is covered with square windows at seven sizes; each window gets a
  frozen CLS feature whitened to 128 d; a where-to-look model (logistic regression fitted on the 576 boxes the
  researcher drew on 2025-01-02) keeps the ten windows per instance a researcher would use; `D_sal[i, j]` is the best
  cosine between the kept windows of two instances (`DAY/safe/window_scores.npz`). A crop at a box cannot tell two
  whales apart on its own (the best-looking patch of another whale scores as high). As a ranking signal next to the
  student embedding it helped on the val days and not on the eval days (below): unproven, with boxes from one day.
- **Candidate links, scored** (`pairs` -> `DAY/safe/link_scores.json`, read by Whale-Review-Web /bursts): every pair
  of clusters inside one burst; each cluster's best partners up to a minute away by the **short-gap linker**
  (`experiments/safe_clusters/link_gbt.pkl`: gradient-boosted trees on the cluster-pair table, time, frame
  continuity, student cosines and window scores, fitted on the catalog labels of the train days by
  `experiments/stage2_linker/linker.py`); and its best partners up to 15 min away by the **rank mean** (the mean
  within-day rank of the student's fin and body cosines and `D_sal`). The linker holds everywhere it was checked: in
  its band (1.5 to 60 s) AUC 0.82 on the val days, 0.82 on the eval days (read once) and 0.84 on the train days with
  each day held out, against 0.67 to 0.69 for the student's best part (higher on 19 of 19 held-out days). The rank
  mean with the window score does not: on the val days it finds 277 / 480 / 787 catalog-same pairs at precision 0.95
  / 0.90 / 0.80 against 224 / 328 / 640 for the student alone, on the eval days 203 / 332 / 857 against 247 / 419 /
  845. A train day is scored by the linker fitted without it (`experiments/safe_clusters/link_gbt_loo/DAY.pkl`,
  `linker.py --loo`). No score is precise enough to link automatically across a gap: these order the reviewer's
  queue, and each verdict saved there keeps the pair's scores so the order can be judged on the researcher's labels.

    python -m model_fine_tuning.cluster.link_scores instances --day-dir experiments/days_all/DAY    # crop features of every instance (quick)
    python -m model_fine_tuning.cluster.link_scores not-whale --day-dir experiments/days_all/DAY    # -> DAY/safe/not_whale.json
    python -m model_fine_tuning.cluster.safe_tracklets cluster --day-dir experiments/days_all/DAY --model experiments/safe_clusters/safe_model.pkl \
        --threshold 0.995 --strict --filter [--labels model_fine_tuning/splits/labels_all_days.json]     # clusters without the non-whales
    python -m model_fine_tuning.cluster.link_scores windows --day-dir experiments/days_all/DAY      # GPU: ~250 windows per instance, ~560 crops/s
    python -m model_fine_tuning.cluster.link_scores scores --day-dir experiments/days_all/DAY       # -> DAY/safe/window_scores.npz
    python -m model_fine_tuning.cluster.link_scores pairs --day-dir experiments/days_all/DAY        # -> DAY/safe/link_scores.json (the review queue)

The window extraction is bound by the GPU; several workers side by side gain little and each queued batch of crops
sits in /dev/shm, so keep the loader's small batches. `look` and `--fit-whitening` refit the where-to-look model and
the whitening (done once, on 2025-01-02); refit `look` when another day has region boxes.

## View class: fluke / dorsal fin / other (2026-11-12)

`model_fine_tuning/cluster/view_class.py`: does a whale instance show its **fluke**, its **dorsal fin**, or neither?
The researchers tell animals by the fluke first and the fin second, and the Photo Labeling page works through a day in
that order; this says which stack is which, straight from the pixels (no mesh, no canonical map, no keypoints). Two
small classifiers (fluke yes / no, fin yes / no) on frozen DINOv3 ViT-B/16 features of two crops of the instance box
(a 448-px square and a shape-keeping crop, long side up to 672 px): PCA 128 + RBF SVM, a probability per answer.
Trained on 1 360 labelled instances of the train + val days (by-eye labels of random and hard instances, the
researcher's keypoint clicks, the catalogue's fluke collection; `experiments/view_class`, README there).

Three-way accuracy on a random sample of the four eval days, labelled blind: **0.970** (128 of 132; fluke precision
0.971 / recall 0.985, dorsal fin 0.965 / 0.982; at P >= 0.9 no false call in 113), against 0.841 for the fin / fluke
flags of the part charts (`cluster/parts.py`, from the synthetic-trained canonical map), which call a fin on almost
any back and miss a fifth of the flukes. "Other" is the weak class (8 of 10): pectoral fins and tail stocks.
Synthetic shots as extra training data were tested and are not used: they help with 50-100 real labels (+2-3 points)
and not with the labels there are now.

    python -m model_fine_tuning.cluster.view_class features --day-dir experiments/days_all/DAY    # GPU, 10-20 s -> DAY/safe/view_feat.npz
    python -m model_fine_tuning.cluster.view_class predict  --day-dir experiments/days_all/DAY    # -> DAY/safe/view_class.json {instance key: [P fluke, P dorsal fin]}
    python -m model_fine_tuning.cluster.view_class fit                                           # refit on experiments/view_class/targets.json (train + val days)
    python -m model_fine_tuning.cluster.view_class report --split val|eval                       # the saved model against the labels of a split

`model_fine_tuning/cluster/view_match.py` (the same evening): for the instances that show a part, the classifier's
frozen features become a 32-dimensional match embedding per part (centred on the day, whitened, scaled down along the
directions in which one animal varies, within-day LDA; fitted on the catalogue animals of the train days by
`experiments/view_match/vm_fit.py`, a train day embedded by the projection fitted without it). On days neither model
saw, the dorsal-fin embedding tells one whale from two better than the student's fin cosine (stack pairs inside
encounters, AUC 0.81 against 0.78); the Labeling Assistant blends the two for its ranking (first choice right on the
eval days 71.7 % -> 75.1 %). For flukes it made no difference on the page and is not used.

    python -m model_fine_tuning.cluster.view_match embed --day-dir experiments/days_all/DAY       # -> DAY/safe/view_match.npz (after view_class predict)

A new day needs `view_class features`, `view_class predict` and `view_match embed` after `safe_clusters features` (they read `DAY/safe/photos.json`);
Whale-Review-Web /labeling reads `view_class.json` when the day has one and falls back to the chart flags when not.

## Earlier work

The classical-vision isolation and clustering code from the first iteration lives in git
history (`helpers/`, `notebooks/`, `initial_work_old/` at commit a372690).
