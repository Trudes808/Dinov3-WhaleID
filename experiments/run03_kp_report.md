# run03: keypoint heads vs plain retrain (2026-10-13T20:01Z)

Both models: same backbone, split, schedule; `canon_r03_kp` adds 18 keypoint heatmaps, a visibility state per keypoint (visible / under water / hidden / out of frame) and the 7 deformation parameters. Synthetic numbers are on the run03 test split (oracle crops).

## Synthetic test split, overall

| metric | canon_r03_ft4 | canon_r03_kp |
|---|---|---|
| surface error mean (m) | 0.251 | 0.250 |
| surface error median (m) | 0.143 | 0.139 |
| visibility IoU | 0.947 | 0.947 |
| side accuracy | 0.961 | 0.956 |
| rotation direct < 15 deg | 0.741 | 0.735 |
| rotation direct median (deg) | 9.352 | 9.388 |
| rotation PnP (dense map) < 15 deg | 0.567 | 0.572 |
| rotation PnP from detected keypoints < 15 deg | - | 0.675 |
| images with a keypoint PnP | 0 | 255 |
| landmark PCK@0.1 from the map | 0.967 | 0.961 |
| keypoint head PCK@0.05 / @0.1 | - / - | 0.984 / 0.988 |
| keypoint state accuracy | - | 0.918 |
| keypoint visible F1 | - | 0.777 |
| deformation error (deg, mean abs) | - | 5.993 |
| images | 1153 | 1153 |

## By pose category (oracle crops): hard views first

| category | n | surf. median m: canon_r03_ft4 / canon_r03_kp | rot direct <15: canon_r03_ft4 / canon_r03_kp | rot PnP(map) <15: canon_r03_ft4 / canon_r03_kp | rot PnP(keypoints) <15 (canon_r03_kp) | kp PCK@0.1 | state acc |
|---|---|---|---|---|---|---|---|
| fin_tip | 75 | 0.138 / 0.132 | 0.29 / 0.33 | 0.03 / 0.01 | - (n=0) | 1.00 | 1.00 |
| pec_up | 66 | 0.230 / 0.220 | 0.38 / 0.33 | 0.15 / 0.15 | 0.50 (n=4) | 1.00 | 0.94 |
| roll_belly | 44 | 0.148 / 0.166 | 0.05 / 0.05 | 0.11 / 0.05 | - (n=0) | 1.00 | 0.94 |
| fluke_edge | 82 | 0.136 / 0.151 | 0.49 / 0.52 | 0.38 / 0.40 | 1.00 (n=1) | 1.00 | 0.94 |
| blow_sliver | 77 | 0.109 / 0.116 | 0.77 / 0.70 | 0.68 / 0.69 | 0.33 (n=3) | 1.00 | 0.94 |
| awash | 70 | 0.098 / 0.089 | 0.76 / 0.77 | 0.50 / 0.54 | - (n=0) | 1.00 | 0.94 |
| peduncle | 80 | 0.141 / 0.137 | 0.89 / 0.93 | 0.75 / 0.70 | 0.81 (n=37) | 1.00 | 0.94 |
| spyhop_low | 40 | 0.084 / 0.078 | 0.93 / 0.85 | 0.60 / 0.55 | - (n=0) | 1.00 | 1.00 |
| arch | 107 | 0.190 / 0.181 | 0.86 / 0.82 | 0.54 / 0.59 | 0.61 (n=44) | 1.00 | 0.89 |
| blow | 124 | 0.145 / 0.135 | 0.92 / 0.92 | 0.81 / 0.86 | 0.90 (n=21) | 1.00 | 0.94 |
| breach | 66 | 0.201 / 0.202 | 0.91 / 0.88 | 0.83 / 0.79 | 0.56 (n=48) | 0.92 | 0.89 |
| fluke_up | 149 | 0.132 / 0.126 | 0.87 / 0.84 | 0.62 / 0.62 | 0.65 (n=80) | 1.00 | 0.94 |
| logging | 122 | 0.149 / 0.161 | 0.83 / 0.87 | 0.76 / 0.75 | 0.76 (n=17) | 1.00 | 0.94 |
| spyhop | 50 | 0.096 / 0.095 | 0.94 / 0.96 | 0.70 / 0.74 | - (n=0) | 1.00 | 0.94 |
| submerged | 1 | 0.331 / 0.484 | 1.00 / 1.00 | 0.00 / 0.00 | - (n=0) | 1.00 | 0.44 |

## Real photos (2024-12-18)

| outline metric | canon_r02_ft4 | canon_r03_ft4 | canon_r03_kp, dense map | canon_r03_kp, map + keypoints |
|---|---|---|---|---|
| IoU above water, median | 0.279 | 0.308 | 0.292 | 0.296 |
| IoU above water, mean | 0.282 | 0.306 | 0.286 | 0.286 |
| photos with IoU >= 0.5 | 0.069 | 0.102 | 0.078 | 0.075 |
| coverage, median | 0.813 | 0.806 | 0.794 | 0.808 |
| excess above water, median | 0.686 | 0.649 | 0.662 | 0.669 |
| PnP inlier fraction, median | 0.467 | 0.409 | 0.411 | 0.405 |
| median reprojection error (px) | 8.200 | 10.700 | 10.115 | 10.215 |
| photos with keypoints used | - | 0 | 0 | 275 |

Hardest quartile of real photos by the plain model's outline IoU (83 photos, IoU <= 0.202):

| model | IoU above water, median | mean |
|---|---|---|
| canon_r03_ft4 | 0.112 | 0.107 |
| canon_r03_kp, dense map | 0.121 | 0.145 |
| canon_r03_kp, map + keypoints | 0.113 | 0.152 |

Side-flip (mirror) test on the real photos:

```
== real photos: fraction of whale pixels whose LEFT/RIGHT changes when the crop is mirrored
  before ((u,theta) head, 120 crops): median 0.000 mean 0.162  crops >20%: 21.05%
  after  (canonical surface, 335 crops): median 0.028 mean 0.152  crops >20%: 17.07%
== after: folded-coordinate disagreement under mirroring (metres on the whale), median over crops: 0.322 m
== after: side margin (relative cost gap between the two global hypotheses): median 0.381  crops with margin < 0.1: 5.37%
== after: direct-head rotation change under mirroring, median deg: 16.4   direct vs PnP (resolved side), median deg: 50.6
== after: points of interest found per photo (of 9 landmarks): median 1.0
sheet: experiments/real_2024-12-18/compare_canon_r03_kp/worst_before_as_seen_after.jpg
```

## Files

- `experiments/canon_r03_kp/eval_test_kp/metrics.json` (per-category under `by_category`), `experiments/canon_r03_ft4/eval_test_kp/metrics.json`
- `experiments/real_2024-12-18/canon_canon_r03_kp/` (keypoints in `points.jsonl` and the npz), `experiments/real_2024-12-18/outline_metric_canon_r03_kp.json`, `experiments/real_2024-12-18/outline_metric_canon_r03_kp_kp.json`
