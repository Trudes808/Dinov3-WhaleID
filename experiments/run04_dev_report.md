# run04 fine-tunes vs run03 (2026-10-14T14:45Z)

Synthetic metrics are on the **run04 test split** for the three model columns (the run03 models are evaluated on it too, so
the comparison is like for like: run04's split has the rigged whale, calves, close-ups and long-lens framing, a harder set
than run03's); run03's own test split is the reference column. 'open air' / 'at seams' boundary scores use the run04 contour
label: the whale's own silhouette against sky or water vs the waterline / foam / calf edges. Real-photo numbers are label-free.

## Training

- seg_r04_cw_dev: finished in 0.0 h, epochs 6, train images 498, best val {'dataset_iou': 0.9499, 'boundary_f1_open': 0.3176}, contour weights {'contour_w_true': 4.0, 'contour_w_seam': 0.3}
- canon_r04_cw_dev: finished in 0.0 h, epochs 6, train images 471, best val {'surface_err_m_mean': 5.0439, 'vis_bf1_open_mean': 0.0709, 'score': 0.0}, contour weights {'contour_w_true': 4.0, 'contour_w_seam': 0.3}
- seg_r04_ft4_dev: finished in 0.0 h, epochs 6, train images 498, best val {'dataset_iou': 0.9488, 'boundary_f1_open': 0.3181}
- canon_r04_ft4_dev: finished in 0.0 h, epochs 6, train images 471, best val {'surface_err_m_mean': 5.1398, 'vis_bf1_open_mean': 0.0868, 'score': 0.0}
- seg_r03_ft4: finished in 1.6 h, epochs 30, train images 5753, best val {'dataset_iou': 0.9631}
- canon_r03_ft4: finished in 1.6 h, epochs 40, train images 5345, best val {'surface_err_m_mean': 0.2481, 'score': 0.8458}

## Segmentation (test split)

| segmentation | run04 contour-weighted | run04 plain | run03 models | run03 (own test split) | best |
|---|---|---|---|---|---|
| dataset IoU | 0.960 | 0.959 | 0.946 | 0.963 | run04 contour-weighted |
| mean image IoU | 0.901 | 0.900 | 0.897 | 0.949 | run04 contour-weighted |
| median image IoU | 0.926 | 0.926 | 0.921 | 0.991 | run04 contour-weighted |
| precision | 0.982 | 0.982 | 0.986 | 0.967 | run03 models |
| recall | 0.977 | 0.975 | 0.960 | 0.996 | run04 contour-weighted |
| IoU >= 0.5 | 1.000 | 1.000 | 1.000 | 0.980 | run04 contour-weighted |
| IoU >= 0.75 | 0.906 | 0.906 | 0.938 | 0.949 | run03 models |
| boundary F1 (all, image mean) | 0.815 | 0.813 | 0.798 | - | run04 contour-weighted |
| boundary F1, open air (pooled) | 0.805 | 0.803 | 0.800 | - | run04 contour-weighted |
| boundary precision / recall, open air (pooled) | 0.858 / 0.759 | 0.848 / 0.763 | 0.849 / 0.756 | - |  |
| boundary F1, at seams (pooled) | 0.769 | 0.764 | 0.784 | - | run03 models |
| boundary F1, open air (image mean) | 0.779 | 0.778 | 0.737 | - | run04 contour-weighted |
| true-contour recall (pooled) | 0.511 | 0.504 | 0.530 | - | run03 models |
| negatives clean | 1.000 | 1.000 | 1.000 | 1.000 | run04 contour-weighted |
| images | 106 | 106 | 106 | 1232 |  |

## Canonical-surface model (test split)

Oracle crops (ground-truth box):

| oracle crops | run04 contour-weighted | run04 plain | run03 models | run03 (own test split) | best |
|---|---|---|---|---|---|
| surface error mean (m) | 4.983 | 5.083 | 2.778 | 0.251 | run03 models |
| surface error median (m) | 3.795 | 3.855 | 1.642 | 0.178 | run03 models |
| torso / fluke / fin error (m) | 4.885 / 8.850 / 2.812 | 4.944 / 9.075 / 3.139 | 2.699 / 4.864 / 2.384 | 0.260 / 0.366 / 0.453 |  |
| visibility IoU | 0.537 | 0.578 | 0.848 | 0.947 | run03 models |
| visibility boundary F1, open air (pooled) | 0.102 | 0.166 | 0.753 | - | run03 models |
| visibility boundary F1, at seams (pooled) | 0.313 | 0.364 | 0.831 | - | run03 models |
| visibility boundary F1, open air (image mean) | 0.060 | 0.130 | 0.748 | - | run03 models |
| true-contour recall (pooled) | 0.162 | 0.186 | 0.555 | - | run03 models |
| mirror disagreement (m) | 0.125 | 0.128 | 0.492 | 0.109 | run04 contour-weighted |
| side accuracy | 0.561 | 0.544 | 0.652 | 0.961 | run03 models |
| side flip agreement | 0.600 | 0.623 | 0.833 | 0.943 | run03 models |
| unstable crops (agreement < 0.8) | 0.634 | 0.602 | 0.260 | 0.048 | run03 models |
| rotation direct median (deg) | 78.243 | 78.544 | 100.082 | 9.352 | run04 contour-weighted |
| rotation direct < 15 deg | 0.021 | 0.021 | 0.031 | 0.741 | run03 models |
| rotation PnP median (deg) | 119.676 | 120.522 | 102.012 | 12.655 | run03 models |
| rotation PnP < 15 deg | 0.011 | 0.000 | 0.042 | 0.567 | run03 models |
| log-scale PnP error | 0.833 | 0.959 | 0.228 | 0.059 | run03 models |
| PnP inlier fraction | 0.190 | 0.163 | 0.318 | 0.640 | run03 models |
| landmark PCK@0.1 (map) | 0.121 | 0.071 | 0.507 | 0.967 | run03 models |
| landmark PCK@0.1 (PnP) | 0.098 | 0.065 | 0.660 | 0.967 | run03 models |
| images evaluated | 96 | 96 | 96 | 1153 |  |

Crops from the column's own segmentation model:

| predicted crops | run04 contour-weighted | run04 plain | run03 models | run03 (own test split) | best |
|---|---|---|---|---|---|
| surface error mean (m) | 4.984 | 5.085 | 2.838 | 0.249 | run03 models |
| surface error median (m) | 3.855 | 3.765 | 1.667 | 0.179 | run03 models |
| torso / fluke / fin error (m) | 4.887 / 8.842 / 2.812 | 4.945 / 9.066 / 3.151 | 2.734 / 4.918 / 2.516 | 0.262 / 0.356 / 0.467 |  |
| visibility IoU | 0.529 | 0.570 | 0.836 | 0.940 | run03 models |
| visibility boundary F1, open air (pooled) | 0.112 | 0.183 | 0.759 | - | run03 models |
| visibility boundary F1, at seams (pooled) | 0.318 | 0.367 | 0.827 | - | run03 models |
| visibility boundary F1, open air (image mean) | 0.073 | 0.131 | 0.736 | - | run03 models |
| true-contour recall (pooled) | 0.165 | 0.202 | 0.568 | - | run03 models |
| mirror disagreement (m) | 0.135 | 0.131 | 0.523 | 0.108 | run04 plain |
| side accuracy | 0.556 | 0.535 | 0.631 | 0.958 | run03 models |
| side flip agreement | 0.602 | 0.622 | 0.814 | 0.944 | run03 models |
| unstable crops (agreement < 0.8) | 0.587 | 0.565 | 0.260 | 0.046 | run03 models |
| rotation direct median (deg) | 77.157 | 77.163 | 95.140 | 9.265 | run04 contour-weighted |
| rotation direct < 15 deg | 0.010 | 0.010 | 0.021 | 0.735 | run03 models |
| rotation PnP median (deg) | 114.807 | 105.328 | 97.152 | 12.621 | run03 models |
| rotation PnP < 15 deg | 0.000 | 0.000 | 0.010 | 0.572 | run03 models |
| log-scale PnP error | 0.762 | 0.931 | 0.295 | 0.059 | run03 models |
| PnP inlier fraction | 0.195 | 0.165 | 0.325 | 0.646 | run03 models |
| landmark PCK@0.1 (map) | 0.116 | 0.071 | 0.475 | 0.964 | run03 models |
| landmark PCK@0.1 (PnP) | 0.104 | 0.095 | 0.627 | 0.967 | run03 models |
| images evaluated | 96 | 96 | 96 | 1153 |  |

## Real photos (2024-12-18, 334 photos)

| label-free outline metric | run04 contour-weighted | run04 plain | run03 models | best |
|---|---|---|---|---|
| IoU above water, median | 0.064 | 0.052 | - | run04 contour-weighted |
| IoU above water, mean | 0.069 | 0.065 | - | run04 contour-weighted |
| photos with IoU >= 0.5 | 0.000 | 0.000 | - |  |
| coverage, median | 0.629 | 0.579 | - | run04 contour-weighted |
| excess above water, median | 0.935 | 0.946 | - | run04 contour-weighted |
| PnP inlier fraction, median | 0.206 | 0.182 | - | run04 contour-weighted |
| median reprojection error (px), median | 31.040 | 30.410 | - | run04 plain |
| photos scored | 33 | 36 | - |  |

| mirror test | run04 contour-weighted | run04 plain | run03 models | best |
|---|---|---|---|---|
| crops | 40 | 40 | - |  |
| crops whose side flips > 20 % under mirroring | 0.515 | 0.250 | - | run04 plain |
| side flip agreement, median | 0.701 | 1.000 | - | run04 plain |
| folded disagreement under mirroring (m), median | 0.095 | 0.073 | - | run04 plain |

<details><summary>side-flip comparison, run04 contour-weighted</summary>

```
== real photos: fraction of whale pixels whose LEFT/RIGHT changes when the crop is mirrored
  before ((u,theta) head, 120 crops): median 0.000 mean 0.162  crops >20%: 21.05%
  after  (canonical surface, 40 crops): median 0.299 mean 0.350  crops >20%: 51.52%
== after: folded-coordinate disagreement under mirroring (metres on the whale), median over crops: 0.095 m
== after: side margin (relative cost gap between the two global hypotheses): median 0.237  crops with margin < 0.1: 27.50%
== after: direct-head rotation change under mirroring, median deg: 15.1   direct vs PnP (resolved side), median deg: 138.0
== after: points of interest found per photo (of 9 landmarks): median 0.0
```
</details>

<details><summary>side-flip comparison, run04 plain</summary>

```
== real photos: fraction of whale pixels whose LEFT/RIGHT changes when the crop is mirrored
  before ((u,theta) head, 120 crops): median 0.000 mean 0.162  crops >20%: 21.05%
  after  (canonical surface, 40 crops): median 0.000 mean 0.225  crops >20%: 25.00%
== after: folded-coordinate disagreement under mirroring (metres on the whale), median over crops: 0.073 m
== after: side margin (relative cost gap between the two global hypotheses): median 0.483  crops with margin < 0.1: 17.50%
== after: direct-head rotation change under mirroring, median deg: 15.1   direct vs PnP (resolved side), median deg: 160.4
== after: points of interest found per photo (of 9 landmarks): median 0.0
```
</details>

## Files

- run04 contour-weighted: `experiments/canon_r04_cw_dev/eval_test/metrics.json`, `experiments/seg_r04_cw_dev/eval_test/metrics.json`, checkpoints `experiments/canon_r04_cw_dev/checkpoints/best.pt`, `experiments/seg_r04_cw_dev/checkpoints/best.pt`, real predictions `/tmp/claude-1072903897/-home-sat3737-NOAA-whales/4fbba7ff-3ba5-4b50-9cb7-970c81250409/scratchpad/real_dev/canon_canon_r04_cw_dev/`, outline metric `/tmp/claude-1072903897/-home-sat3737-NOAA-whales/4fbba7ff-3ba5-4b50-9cb7-970c81250409/scratchpad/real_dev/outline_metric_canon_r04_cw_dev.json`
- run04 plain: `experiments/canon_r04_ft4_dev/eval_test/metrics.json`, `experiments/seg_r04_ft4_dev/eval_test/metrics.json`, checkpoints `experiments/canon_r04_ft4_dev/checkpoints/best.pt`, `experiments/seg_r04_ft4_dev/checkpoints/best.pt`, real predictions `/tmp/claude-1072903897/-home-sat3737-NOAA-whales/4fbba7ff-3ba5-4b50-9cb7-970c81250409/scratchpad/real_dev/canon_canon_r04_ft4_dev/`, outline metric `/tmp/claude-1072903897/-home-sat3737-NOAA-whales/4fbba7ff-3ba5-4b50-9cb7-970c81250409/scratchpad/real_dev/outline_metric_canon_r04_ft4_dev.json`
- run03 models: `experiments/canon_r03_ft4/eval_run04_dev_test/metrics.json`, `experiments/seg_r03_ft4/eval_run04_dev_test/metrics.json`, checkpoints `experiments/canon_r03_ft4/checkpoints/best.pt`, `experiments/seg_r03_ft4/checkpoints/best.pt`, real predictions `/tmp/claude-1072903897/-home-sat3737-NOAA-whales/4fbba7ff-3ba5-4b50-9cb7-970c81250409/scratchpad/real_dev/canon_canon_r03_ft4_v2/`, outline metric `/tmp/claude-1072903897/-home-sat3737-NOAA-whales/4fbba7ff-3ba5-4b50-9cb7-970c81250409/scratchpad/real_dev/outline_metric_canon_r03_ft4_v2.json`
