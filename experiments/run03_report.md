# run03 fine-tunes vs run02 baselines (2026-10-13T18:17Z)

Synthetic metrics are on each run's own test split (run03's has close-ups, rolled whales, calf-to-adult
sizes and long-lens framing, so it is a harder set than run02's; compare like with like on the real photos).

## Training

- seg_r03_ft4: finished in 1.6 h, epochs 30, best {'dataset_iou': 0.9631437235900213}
- canon_r03_ft4: finished in 1.6 h, epochs 40, best {'surface_err_m_mean': 0.24813366363044476, 'score': 0.8458383469164119}

## Synthetic test split: canonical-surface model

| metric (oracle_crops) | run03 | run02 baseline | better |
|---|---|---|---|
| surface error mean (m) | 0.251 | 0.475 | run03 |
| surface error median (m) | 0.178 | 0.248 | run03 |
| torso / fluke / fin error (m) | 0.260 / 0.366 / 0.453 | 0.417 / 0.814 / 1.090 |  |
| visibility IoU | 0.947 | 0.897 | run03 |
| mirror disagreement (m) | 0.109 | 0.167 | run03 |
| side accuracy | 0.961 | 0.931 | run03 |
| side flip agreement | 0.943 | 0.915 | run03 |
| unstable crops (agreement < 0.8) | 0.048 | 0.066 | run03 |
| rotation direct median (deg) | 9.352 | 12.595 | run03 |
| rotation direct < 15 deg | 0.741 | 0.613 | run03 |
| rotation PnP median (deg) | 12.655 | 14.464 | run03 |
| rotation PnP < 15 deg | 0.567 | 0.527 | run03 |
| log-scale PnP error | 0.059 | 0.072 | run03 |
| PnP inlier fraction | 0.640 | 0.658 | run02 |
| landmark PCK@0.1 (map) | 0.967 | 0.659 | run03 |
| landmark PCK@0.1 (PnP) | 0.967 | 0.879 | run03 |
| images evaluated | 1153 | 256 | |

| metric (seg_crops) | run03 | run02 baseline | better |
|---|---|---|---|
| surface error mean (m) | 0.249 | 0.490 | run03 |
| surface error median (m) | 0.179 | 0.253 | run03 |
| torso / fluke / fin error (m) | 0.262 / 0.356 / 0.467 | 0.417 / 0.879 / 1.257 |  |
| visibility IoU | 0.940 | 0.897 | run03 |
| mirror disagreement (m) | 0.108 | 0.169 | run03 |
| side accuracy | 0.958 | 0.927 | run03 |
| side flip agreement | 0.944 | 0.927 | run03 |
| unstable crops (agreement < 0.8) | 0.046 | 0.062 | run03 |
| rotation direct median (deg) | 9.265 | 11.718 | run03 |
| rotation direct < 15 deg | 0.735 | 0.625 | run03 |
| rotation PnP median (deg) | 12.621 | 14.772 | run03 |
| rotation PnP < 15 deg | 0.572 | 0.504 | run03 |
| log-scale PnP error | 0.059 | 0.071 | run03 |
| PnP inlier fraction | 0.646 | 0.661 | run02 |
| landmark PCK@0.1 (map) | 0.964 | 0.670 | run03 |
| landmark PCK@0.1 (PnP) | 0.967 | 0.880 | run03 |
| images evaluated | 1153 | 256 | |

## Segmentation

| segmentation (val) | run03 | run02 baseline |
|---|---|---|
| dataset IoU | 0.963 | 0.953 |
| mean image IoU | 0.949 | 0.932 |
| median image IoU | 0.991 | 0.963 |
| precision | 0.967 | 0.979 |
| recall | 0.996 | 0.974 |
| IoU >= 0.5 | 0.980 | 0.992 |
| IoU >= 0.75 | 0.949 | 0.953 |
| negatives clean | 1.000 | 1.000 |
| images | 1232 | 281 |

## Real photos (2024-12-18, 334 photos)

| real photos, label-free outline metric | run03 | run02 baseline |
|---|---|---|
| IoU above water, median | 0.308 | 0.279 |
| IoU above water, mean | 0.306 | 0.282 |
| photos with IoU >= 0.5 | 0.102 | 0.069 |
| coverage, median | 0.806 | 0.813 |
| excess above water, median | 0.649 | 0.686 |
| PnP inlier fraction, median | 0.409 | 0.467 |
| median reprojection error (px), median | 10.700 | 8.200 |
| photos scored | 333 | 334 |

| real photos, mirror test | run03 | run02 baseline |
|---|---|---|
| crops | 336 | - |
| crops whose side flips > 20 % under mirroring | 0.171 | - |
| side flip agreement, median | 0.971 | - |
| folded disagreement under mirroring (m), median | 0.325 | - |

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

## Files

- synthetic: `experiments/canon_r03_ft4/eval_test/metrics.json`
- real predictions: `experiments/real_2024-12-18/canon_canon_r03_ft4/`
- outline metric: `experiments/real_2024-12-18/outline_metric_canon_r03_ft4.json`
- checkpoints: `experiments/canon_r03_ft4/checkpoints/best.pt`, `experiments/seg_r03_ft4/checkpoints/best.pt`
