# run04 fine-tunes vs run03 (2026-10-16T08:02Z)

Synthetic metrics are on the **run04 test split** for the three model columns (the run03 models are evaluated on it too, so
the comparison is like for like: run04's split has the rigged whale, calves, close-ups and long-lens framing, a harder set
than run03's); run03's own test split is the reference column. 'open air' / 'at seams' boundary scores use the run04 contour
label: the whale's own silhouette against sky or water vs the waterline / foam / calf edges. Real-photo numbers are label-free.

## Training

- seg_r04_cw: finished in 1.9 h, epochs 30, train images 6676, best val {'dataset_iou': 0.9601, 'boundary_f1_open': 0.8949}, contour weights {'contour_w_true': 4.0, 'contour_w_seam': 0.3}
- canon_r04_cw: finished in 1.9 h, epochs 40, train images 6296, best val {'surface_err_m_mean': 0.4707, 'vis_bf1_open_mean': 0.808, 'score': 0.7076}, contour weights {'contour_w_true': 4.0, 'contour_w_seam': 0.3}
- seg_r04_ft4: finished in 1.9 h, epochs 30, train images 6676, best val {'dataset_iou': 0.9601, 'boundary_f1_open': 0.8824}
- canon_r04_ft4: finished in 1.9 h, epochs 40, train images 6296, best val {'surface_err_m_mean': 0.4633, 'vis_bf1_open_mean': 0.797, 'score': 0.7122}
- seg_r03_ft4: finished in 1.6 h, epochs 30, train images 5753, best val {'dataset_iou': 0.9631}
- canon_r03_ft4: finished in 1.6 h, epochs 40, train images 5345, best val {'surface_err_m_mean': 0.2481, 'score': 0.8458}

## Segmentation (test split)

| segmentation | run04 contour-weighted | run04 plain | run03 models | run03 (own test split) | best |
|---|---|---|---|---|---|
| dataset IoU | 0.961 | 0.960 | 0.935 | 0.963 | run04 contour-weighted |
| mean image IoU | 0.928 | 0.926 | 0.883 | 0.949 | run04 contour-weighted |
| median image IoU | 0.951 | 0.949 | 0.916 | 0.991 | run04 contour-weighted |
| precision | 0.974 | 0.975 | 0.960 | 0.967 | run04 plain |
| recall | 0.986 | 0.984 | 0.972 | 0.996 | run04 contour-weighted |
| IoU >= 0.5 | 0.997 | 0.996 | 0.986 | 0.980 | run04 contour-weighted |
| IoU >= 0.75 | 0.966 | 0.964 | 0.896 | 0.949 | run04 contour-weighted |
| boundary F1 (all, image mean) | 0.888 | 0.886 | 0.802 | - | run04 contour-weighted |
| boundary F1, open air (pooled) | 0.857 | 0.851 | 0.790 | - | run04 contour-weighted |
| boundary precision / recall, open air (pooled) | 0.888 / 0.829 | 0.898 / 0.808 | 0.830 / 0.754 | - |  |
| boundary F1, at seams (pooled) | 0.889 | 0.883 | 0.809 | - | run04 contour-weighted |
| boundary F1, open air (image mean) | 0.873 | 0.857 | 0.794 | - | run04 contour-weighted |
| true-contour recall (pooled) | 0.723 | 0.709 | 0.541 | - | run04 contour-weighted |
| negatives clean | 0.966 | 0.966 | 0.966 | 1.000 | run04 contour-weighted |
| images | 1429 | 1429 | 1429 | 1232 |  |

## Canonical-surface model (test split)

Oracle crops (ground-truth box):

| oracle crops | run04 contour-weighted | run04 plain | run03 models | run03 (own test split) | best |
|---|---|---|---|---|---|
| surface error mean (m) | 0.432 | 0.441 | 2.366 | 0.251 | run04 contour-weighted |
| surface error median (m) | 0.265 | 0.275 | 1.625 | 0.178 | run04 contour-weighted |
| torso / fluke / fin error (m) | 0.436 / 0.568 / 1.250 | 0.448 / 0.584 / 1.320 | 2.201 / 4.134 / 2.677 | 0.260 / 0.366 / 0.453 |  |
| visibility IoU | 0.922 | 0.922 | 0.857 | 0.947 | run04 plain |
| visibility boundary F1, open air (pooled) | 0.803 | 0.813 | 0.727 | - | run04 plain |
| visibility boundary F1, at seams (pooled) | 0.916 | 0.911 | 0.850 | - | run04 contour-weighted |
| visibility boundary F1, open air (image mean) | 0.816 | 0.821 | 0.726 | - | run04 plain |
| true-contour recall (pooled) | 0.717 | 0.698 | 0.590 | - | run04 contour-weighted |
| mirror disagreement (m) | 0.154 | 0.150 | 0.445 | 0.109 | run04 plain |
| side accuracy | 0.868 | 0.881 | 0.662 | 0.961 | run04 plain |
| side flip agreement | 0.868 | 0.874 | 0.791 | 0.943 | run04 plain |
| unstable crops (agreement < 0.8) | 0.152 | 0.135 | 0.275 | 0.048 | run04 plain |
| rotation direct median (deg) | 10.650 | 10.536 | 85.191 | 9.352 | run04 plain |
| rotation direct < 15 deg | 0.679 | 0.677 | 0.016 | 0.741 | run04 contour-weighted |
| rotation PnP median (deg) | 30.881 | 30.859 | 94.958 | 12.655 | run04 plain |
| rotation PnP < 15 deg | 0.099 | 0.108 | 0.015 | 0.567 | run04 plain |
| log-scale PnP error | 0.146 | 0.135 | 0.234 | 0.059 | run04 plain |
| PnP inlier fraction | 0.466 | 0.468 | 0.338 | 0.640 | run04 plain |
| landmark PCK@0.1 (map) | 0.903 | 0.909 | 0.511 | 0.967 | run04 plain |
| landmark PCK@0.1 (PnP) | 0.876 | 0.886 | 0.618 | 0.967 | run04 plain |
| images evaluated | 1342 | 1342 | 1342 | 1153 |  |

Crops from the column's own segmentation model:

| predicted crops | run04 contour-weighted | run04 plain | run03 models | run03 (own test split) | best |
|---|---|---|---|---|---|
| surface error mean (m) | 0.447 | 0.448 | 2.419 | 0.249 | run04 contour-weighted |
| surface error median (m) | 0.275 | 0.283 | 1.670 | 0.179 | run04 contour-weighted |
| torso / fluke / fin error (m) | 0.450 / 0.570 / 1.286 | 0.452 / 0.589 / 1.342 | 2.246 / 4.193 / 2.713 | 0.262 / 0.356 / 0.467 |  |
| visibility IoU | 0.917 | 0.917 | 0.849 | 0.940 | run04 contour-weighted |
| visibility boundary F1, open air (pooled) | 0.828 | 0.829 | 0.747 | - | run04 plain |
| visibility boundary F1, at seams (pooled) | 0.918 | 0.912 | 0.854 | - | run04 contour-weighted |
| visibility boundary F1, open air (image mean) | 0.823 | 0.825 | 0.731 | - | run04 plain |
| true-contour recall (pooled) | 0.723 | 0.707 | 0.595 | - | run04 contour-weighted |
| mirror disagreement (m) | 0.162 | 0.158 | 0.464 | 0.108 | run04 plain |
| side accuracy | 0.872 | 0.873 | 0.662 | 0.958 | run04 plain |
| side flip agreement | 0.865 | 0.873 | 0.794 | 0.944 | run04 plain |
| unstable crops (agreement < 0.8) | 0.152 | 0.134 | 0.282 | 0.046 | run04 plain |
| rotation direct median (deg) | 10.982 | 10.756 | 84.553 | 9.265 | run04 plain |
| rotation direct < 15 deg | 0.660 | 0.663 | 0.015 | 0.735 | run04 plain |
| rotation PnP median (deg) | 31.282 | 31.271 | 93.876 | 12.621 | run04 plain |
| rotation PnP < 15 deg | 0.100 | 0.104 | 0.017 | 0.572 | run04 plain |
| log-scale PnP error | 0.139 | 0.140 | 0.234 | 0.059 | run04 contour-weighted |
| PnP inlier fraction | 0.490 | 0.485 | 0.365 | 0.646 | run04 contour-weighted |
| landmark PCK@0.1 (map) | 0.905 | 0.899 | 0.491 | 0.964 | run04 contour-weighted |
| landmark PCK@0.1 (PnP) | 0.879 | 0.880 | 0.606 | 0.967 | run04 plain |
| images evaluated | 1342 | 1342 | 1342 | 1153 |  |

## Real photos (2024-12-18, 334 photos)

Rigid PnP on the dense map (rotation and translation from the map):

| label-free outline metric, PnP pose | run04 contour-weighted | run04 plain | run03 models | best |
|---|---|---|---|---|
| IoU above water, median | 0.197 | 0.232 | 0.308 | run03 models |
| IoU above water, mean | 0.225 | 0.242 | 0.306 | run03 models |
| photos with IoU >= 0.5 | 0.036 | 0.033 | 0.102 |  |
| coverage, median | 0.767 | 0.785 | 0.806 | run03 models |
| excess above water, median | 0.780 | 0.745 | 0.649 | run03 models |
| PnP inlier fraction, median | 0.382 | 0.401 | 0.409 | run03 models |
| median reprojection error (px), median | 11.370 | 10.680 | 10.700 | run04 plain |
| photos scored | 335 | 335 | 333 |  |

Rotation from the model's direct head, translation fitted on the map (a rigid PnP on a bent whale takes the rotation with it):

| label-free outline metric, direct rotation | run04 contour-weighted | run04 plain | run03 models | best |
|---|---|---|---|---|
| IoU above water, median | 0.228 | 0.235 | 0.306 | run03 models |
| IoU above water, mean | 0.209 | 0.214 | 0.271 | run03 models |
| photos with IoU >= 0.5 | 0.006 | 0.012 | 0.054 |  |
| coverage, median | 0.565 | 0.544 | 0.691 | run03 models |
| excess above water, median | 0.615 | 0.604 | 0.571 | run03 models |
| PnP inlier fraction, median | 0.065 | 0.064 | 0.108 | run03 models |
| median reprojection error (px), median | 29.750 | 29.510 | 21.040 | run03 models |
| photos scored | 335 | 335 | 333 |  |

| mirror test | run04 contour-weighted | run04 plain | run03 models | best |
|---|---|---|---|---|
| crops | 336 | 336 | 336 |  |
| crops whose side flips > 20 % under mirroring | 0.179 | 0.176 | 0.171 | run03 models |
| side flip agreement, median | 0.976 | 0.969 | 0.971 | run04 contour-weighted |
| folded disagreement under mirroring (m), median | 0.297 | 0.246 | 0.325 | run04 plain |

<details><summary>side-flip comparison, run04 contour-weighted</summary>

```
== real photos: fraction of whale pixels whose LEFT/RIGHT changes when the crop is mirrored
  before ((u,theta) head, 120 crops): median 0.000 mean 0.162  crops >20%: 21.05%
  after  (canonical surface, 336 crops): median 0.024 mean 0.153  crops >20%: 17.91%
== after: folded-coordinate disagreement under mirroring (metres on the whale), median over crops: 0.297 m
== after: side margin (relative cost gap between the two global hypotheses): median 0.397  crops with margin < 0.1: 2.98%
== after: direct-head rotation change under mirroring, median deg: 19.4   direct vs PnP (resolved side), median deg: 84.7
== after: points of interest found per photo (of 9 landmarks): median 1.0
sheet: experiments/real_2024-12-18/compare_canon_r04_cw/worst_before_as_seen_after.jpg
```
</details>

<details><summary>side-flip comparison, run04 plain</summary>

```
== real photos: fraction of whale pixels whose LEFT/RIGHT changes when the crop is mirrored
  before ((u,theta) head, 120 crops): median 0.000 mean 0.162  crops >20%: 21.05%
  after  (canonical surface, 336 crops): median 0.031 mean 0.160  crops >20%: 17.61%
== after: folded-coordinate disagreement under mirroring (metres on the whale), median over crops: 0.246 m
== after: side margin (relative cost gap between the two global hypotheses): median 0.406  crops with margin < 0.1: 4.17%
== after: direct-head rotation change under mirroring, median deg: 19.6   direct vs PnP (resolved side), median deg: 58.9
== after: points of interest found per photo (of 9 landmarks): median 1.0
sheet: experiments/real_2024-12-18/compare_canon_r04_ft4/worst_before_as_seen_after.jpg
```
</details>

<details><summary>side-flip comparison, run03 models</summary>

```
== real photos: fraction of whale pixels whose LEFT/RIGHT changes when the crop is mirrored
  before ((u,theta) head, 120 crops): median 0.000 mean 0.162  crops >20%: 21.05%
  after  (canonical surface, 335 crops): median 0.029 mean 0.144  crops >20%: 17.12%
== after: folded-coordinate disagreement under mirroring (metres on the whale), median over crops: 0.325 m
== after: side margin (relative cost gap between the two global hypotheses): median 0.333  crops with margin < 0.1: 5.97%
== after: direct-head rotation change under mirroring, median deg: 15.6   direct vs PnP (resolved side), median deg: 44.4
== after: points of interest found per photo (of 9 landmarks): median 1.0
sheet: experiments/real_2024-12-18/compare_canon_r03_ft4/worst_before_as_seen_after.jpg
```
</details>

## Files

- run04 contour-weighted: `experiments/canon_r04_cw/eval_test/metrics.json`, `experiments/seg_r04_cw/eval_test/metrics.json`, checkpoints `experiments/canon_r04_cw/checkpoints/best.pt`, `experiments/seg_r04_cw/checkpoints/best.pt`, real predictions `experiments/real_2024-12-18/canon_canon_r04_cw/`, outline metric `experiments/real_2024-12-18/outline_metric_canon_r04_cw.json`
- run04 plain: `experiments/canon_r04_ft4/eval_test/metrics.json`, `experiments/seg_r04_ft4/eval_test/metrics.json`, checkpoints `experiments/canon_r04_ft4/checkpoints/best.pt`, `experiments/seg_r04_ft4/checkpoints/best.pt`, real predictions `experiments/real_2024-12-18/canon_canon_r04_ft4/`, outline metric `experiments/real_2024-12-18/outline_metric_canon_r04_ft4.json`
- run03 models: `experiments/canon_r03_ft4/eval_run04_test/metrics.json`, `experiments/seg_r03_ft4/eval_run04_test/metrics.json`, checkpoints `experiments/canon_r03_ft4/checkpoints/best.pt`, `experiments/seg_r03_ft4/checkpoints/best.pt`, real predictions `experiments/real_2024-12-18/canon_canon_r03_ft4_v2/`, outline metric `experiments/real_2024-12-18/outline_metric_canon_r03_ft4_v2.json`
