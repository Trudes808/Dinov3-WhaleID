# Paired reflection test (renders only): tables

## Renders: each model on both arms' validation renders

| model | IoU m1 | IoU m2 | mirror-side P m1 | m2 | mirror-side share >= 0.5 m1 | m2 | lowest 10 px of whale P m1 | m2 | under-water zone P m1 | m2 |
|---|---|---|---|---|---|---|---|---|---|---|
| r04 | 0.895 | 0.8921 | 0.0627 | 0.0188 | 0.0591 | 0.0165 | 0.719 | 0.6168 | 0.0049 | 0.0021 |
| r05 | 0.8964 | 0.8842 | 0.079 | 0.0233 | 0.0746 | 0.0207 | 0.7392 | 0.6241 | 0.0034 | 0.0009 |
| A_m1_s0 | 0.9004 | 0.8884 | 0.0627 | 0.0258 | 0.0592 | 0.0237 | 0.7185 | 0.6335 | 0.0099 | 0.0078 |
| A_m1_s1 | 0.8969 | 0.8909 | 0.0546 | 0.0236 | 0.0511 | 0.0213 | 0.688 | 0.6004 | 0.0118 | 0.0107 |
| A_m2_s0 | 0.8185 | 0.9257 | 0.2587 | 0.0403 | 0.2535 | 0.0375 | 0.8533 | 0.7921 | 0.0445 | 0.0091 |
| A_m2_s1 | 0.7828 | 0.9301 | 0.2777 | 0.0454 | 0.2741 | 0.0427 | 0.8624 | 0.8111 | 0.0715 | 0.0111 |
| B_m1_s0 | 0.9163 | 0.9013 | 0.0609 | 0.022 | 0.0571 | 0.0197 | 0.7555 | 0.6391 | 0.0075 | 0.0028 |
| B_m1_s1 | 0.9193 | 0.9089 | 0.0663 | 0.0258 | 0.0628 | 0.0237 | 0.763 | 0.6589 | 0.0075 | 0.0035 |
| B_m2_s0 | 0.883 | 0.9356 | 0.2212 | 0.0404 | 0.2169 | 0.0376 | 0.8787 | 0.8239 | 0.0204 | 0.006 |
| B_m2_s1 | 0.8802 | 0.9399 | 0.2278 | 0.0459 | 0.2241 | 0.0431 | 0.8851 | 0.8351 | 0.0223 | 0.0065 |

## Real photos, val days (view-labelled main whales; mean P(whale) unless a share)

| measure | r04 | r05 | A_m1_s0 | A_m1_s1 | A_m2_s0 | A_m2_s1 | B_m1_s0 | B_m1_s1 | B_m2_s0 | B_m2_s1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0-9 px above the lower edge | 0.902 | 0.571 | 0.848 | 0.799 | 0.881 | 0.885 | 0.772 | 0.740 | 0.906 | 0.876 |
| 0-9 px above the lower edge, share P >= 0.5 | 0.991 | 0.571 | 0.855 | 0.803 | 0.888 | 0.891 | 0.780 | 0.749 | 0.916 | 0.886 |
| 10-19 px above the lower edge | 0.978 | 0.774 | 0.934 | 0.929 | 0.943 | 0.945 | 0.901 | 0.902 | 0.961 | 0.955 |
| 10-19 px above the lower edge, share P >= 0.5 | 0.999 | 0.778 | 0.936 | 0.931 | 0.945 | 0.947 | 0.906 | 0.910 | 0.964 | 0.958 |
| 20-39 px above the lower edge | 0.985 | 0.895 | 0.959 | 0.956 | 0.963 | 0.966 | 0.955 | 0.951 | 0.972 | 0.972 |
| 20-39 px above the lower edge, share P >= 0.5 | 0.999 | 0.906 | 0.960 | 0.957 | 0.965 | 0.967 | 0.957 | 0.957 | 0.974 | 0.974 |
| below the lower edge +1..4 px, share P >= 0.5 | 0.004 | 0.174 | 0.487 | 0.406 | 0.594 | 0.622 | 0.373 | 0.315 | 0.581 | 0.490 |
| below the lower edge +5..12 px, share P >= 0.5 | 0.003 | 0.055 | 0.265 | 0.197 | 0.405 | 0.390 | 0.186 | 0.154 | 0.357 | 0.270 |
| height fifth 1 (5 = bottom) | 0.981 | 0.950 | 0.918 | 0.923 | 0.924 | 0.932 | 0.944 | 0.947 | 0.945 | 0.946 |
| height fifth 2 (5 = bottom) | 0.995 | 0.971 | 0.978 | 0.978 | 0.978 | 0.980 | 0.976 | 0.977 | 0.983 | 0.983 |
| height fifth 3 (5 = bottom) | 0.994 | 0.963 | 0.982 | 0.979 | 0.984 | 0.981 | 0.978 | 0.977 | 0.988 | 0.985 |
| height fifth 4 (5 = bottom) | 0.991 | 0.931 | 0.975 | 0.970 | 0.982 | 0.978 | 0.970 | 0.963 | 0.985 | 0.982 |
| height fifth 5 (5 = bottom) | 0.943 | 0.693 | 0.919 | 0.897 | 0.936 | 0.945 | 0.867 | 0.856 | 0.950 | 0.938 |
| kept share of the r04 blob (mean over whales) | 0.997 | 0.911 | 0.939 | 0.930 | 0.942 | 0.950 | 0.941 | 0.930 | 0.964 | 0.959 |

## Real photos, train days (view-labelled main whales; mean P(whale) unless a share)

| measure | r04 | r05 | A_m1_s0 | A_m1_s1 | A_m2_s0 | A_m2_s1 | B_m1_s0 | B_m1_s1 | B_m2_s0 | B_m2_s1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0-9 px above the lower edge | 0.917 | 0.659 | 0.853 | 0.843 | 0.891 | 0.923 | 0.813 | 0.809 | 0.903 | 0.900 |
| 0-9 px above the lower edge, share P >= 0.5 | 0.995 | 0.662 | 0.857 | 0.847 | 0.896 | 0.928 | 0.821 | 0.818 | 0.910 | 0.907 |
| 10-19 px above the lower edge | 0.981 | 0.807 | 0.933 | 0.943 | 0.943 | 0.962 | 0.911 | 0.911 | 0.948 | 0.950 |
| 10-19 px above the lower edge, share P >= 0.5 | 0.999 | 0.811 | 0.934 | 0.945 | 0.945 | 0.964 | 0.915 | 0.916 | 0.950 | 0.952 |
| 20-39 px above the lower edge | 0.988 | 0.883 | 0.952 | 0.963 | 0.959 | 0.974 | 0.945 | 0.942 | 0.962 | 0.962 |
| 20-39 px above the lower edge, share P >= 0.5 | 0.999 | 0.888 | 0.953 | 0.964 | 0.960 | 0.975 | 0.948 | 0.947 | 0.964 | 0.963 |
| below the lower edge +1..4 px, share P >= 0.5 | 0.005 | 0.207 | 0.450 | 0.399 | 0.595 | 0.644 | 0.403 | 0.353 | 0.589 | 0.534 |
| below the lower edge +5..12 px, share P >= 0.5 | 0.002 | 0.052 | 0.217 | 0.185 | 0.391 | 0.401 | 0.178 | 0.139 | 0.352 | 0.272 |
| height fifth 1 (5 = bottom) | 0.984 | 0.961 | 0.945 | 0.953 | 0.947 | 0.959 | 0.958 | 0.963 | 0.958 | 0.960 |
| height fifth 2 (5 = bottom) | 0.996 | 0.976 | 0.986 | 0.989 | 0.985 | 0.990 | 0.986 | 0.986 | 0.987 | 0.989 |
| height fifth 3 (5 = bottom) | 0.996 | 0.954 | 0.979 | 0.986 | 0.981 | 0.988 | 0.980 | 0.977 | 0.984 | 0.986 |
| height fifth 4 (5 = bottom) | 0.992 | 0.893 | 0.962 | 0.971 | 0.969 | 0.981 | 0.955 | 0.943 | 0.968 | 0.970 |
| height fifth 5 (5 = bottom) | 0.946 | 0.678 | 0.882 | 0.888 | 0.911 | 0.943 | 0.842 | 0.833 | 0.912 | 0.905 |
| kept share of the r04 blob (mean over whales) | 0.998 | 0.929 | 0.954 | 0.955 | 0.959 | 0.972 | 0.954 | 0.954 | 0.969 | 0.971 |

## Mirror-band nearest neighbours (real whale patches each model drops; pool = m1 validation renders)

```
{
 "render_pool (m1 val)": {
  "band below waterline (mirror side)": 1110,
  "open water": 33555,
  "under-water zone": 9352,
  "whale skin": 7450
 },
 "all whale patches": {
  "patches": 5729,
  "nearest_10_shares": {
   "band below waterline (mirror side)": 0.124,
   "open water": 0.131,
   "under-water zone": 0.083,
   "whale skin": 0.662
  }
 },
 "r05": {
  "patches": 1131,
  "nearest_10_shares": {
   "band below waterline (mirror side)": 0.234,
   "open water": 0.338,
   "under-water zone": 0.254,
   "whale skin": 0.174
  }
 },
 "A_m1_s0": {
  "patches": 650,
  "nearest_10_shares": {
   "band below waterline (mirror side)": 0.209,
   "open water": 0.503,
   "under-water zone": 0.258,
   "whale skin": 0.03
  }
 },
 "A_m1_s1": {
  "patches": 555,
  "nearest_10_shares": {
   "band below waterline (mirror side)": 0.217,
   "open water": 0.496,
   "under-water zone": 0.265,
   "whale skin": 0.022
  }
 },
 "A_m2_s0": {
  "patches": 577,
  "nearest_10_shares": {
   "band below waterline (mirror side)": 0.226,
   "open water": 0.538,
   "under-water zone": 0.205,
   "whale skin": 0.03
  }
 },
 "A_m2_s1": {
  "patches": 361,
  "nearest_10_shares": {
   "band below waterline (mirror side)": 0.25,
   "open water": 0.569,
   "under-water zone": 0.165,
   "whale skin": 0.016
  }
 },
 "B_m1_s0": {
  "patches": 800,
  "nearest_10_shares": {
   "band below waterline (mirror side)": 0.216,
   "open water": 0.414,
   "under-water zone": 0.293,
   "whale skin": 0.077
  }
 },
 "B_m1_s1": {
  "patches": 694,
  "nearest_10_shares": {
   "band below waterline (mirror side)": 0.21,
   "open water": 0.454,
   "under-water zone": 0.295,
   "whale skin": 0.041
  }
 },
 "B_m2_s0": {
  "patches": 583,
  "nearest_10_shares": {
   "band below waterline (mirror side)": 0.184,
   "open water": 0.503,
   "under-water zone": 0.294,
   "whale skin": 0.019
  }
 },
 "B_m2_s1": {
  "patches": 540,
  "nearest_10_shares": {
   "band below waterline (mirror side)": 0.177,
   "open water": 0.511,
   "under-water zone": 0.298,
   "whale skin": 0.014
  }
 }
}
```
