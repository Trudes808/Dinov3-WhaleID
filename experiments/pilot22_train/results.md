| measure | n | W3e s0 | W3e s1 | P_add s0 | P_add s1 | P_only s0 | P_only s1 | P_add: both / sep | P_only: both / sep |
|---|---|---|---|---|---|---|---|---|---|
| VAL total cost (weighted error) | 202 | 0.164 | 0.155 | 0.151 | 0.134 | 0.166 | 0.158 | better / n.s. | - / n.s. |
| train-day total cost (ranking set, reference) | 395 | 0.160 | 0.179 | 0.158 | 0.151 | 0.156 | 0.182 | better / n.s. | - / n.s. |
| VAL F_low error | 22 | 0.159 | 0.159 | 0.184 | 0.184 | 0.184 | 0.210 | worse / higher, sep | worse / n.s. |
| VAL F_edge error | 16 | 0.138 | 0.138 | 0.063 | 0.000 | 0.000 | 0.000 | better / lower, sep | better / lower, sep |
| VAL F_up error | 12 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | - / n.s. | - / n.s. |
| VAL D_small error | 55 | 0.065 | 0.072 | 0.076 | 0.068 | 0.068 | 0.068 | - / n.s. | - / n.s. |
| VAL D_tip error | 7 | 0.239 | 0.239 | 0.239 | 0.000 | 0.561 | 0.400 | - / n.s. | worse / n.s. |
| VAL D_tall error | 23 | 0.012 | 0.012 | 0.012 | 0.012 | 0.012 | 0.012 | - / n.s. | - / n.s. |
| VAL B_back error | 30 | 0.657 | 0.650 | 0.608 | 0.593 | 0.638 | 0.629 | better / lower, sep | better / lower, sep |
| VAL P_pec error | 29 | 0.193 | 0.091 | 0.121 | 0.134 | 0.164 | 0.143 | - / n.s. | - / n.s. |
| VAL H_head error | 7 | 0.000 | 0.000 | 0.000 | 0.000 | 0.064 | 0.000 | - / n.s. | - / n.s. |
| VAL F_low cost | 22 | 0.015 | 0.015 | 0.017 | 0.017 | 0.017 | 0.020 | worse / higher, sep | worse / n.s. |
| VAL F_edge cost | 16 | 0.010 | 0.010 | 0.004 | 0.000 | 0.000 | 0.000 | better / lower, sep | better / lower, sep |
| VAL F_up cost | 12 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | - / n.s. | - / n.s. |
| VAL D_small cost | 55 | 0.024 | 0.026 | 0.028 | 0.025 | 0.025 | 0.025 | - / n.s. | - / n.s. |
| VAL D_tip cost | 7 | 0.009 | 0.009 | 0.009 | 0.000 | 0.022 | 0.015 | - / n.s. | worse / n.s. |
| VAL D_tall cost | 23 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | - / n.s. | - / n.s. |
| VAL B_back cost | 30 | 0.082 | 0.082 | 0.076 | 0.074 | 0.080 | 0.079 | better / lower, sep | better / lower, sep |
| VAL P_pec cost | 29 | 0.020 | 0.009 | 0.013 | 0.014 | 0.017 | 0.015 | - / n.s. | - / n.s. |
| VAL H_head cost | 7 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 | 0.000 | - / n.s. | - / n.s. |
| IoU with the expanded-triangle truth (clicks 03-18) | 99 | 0.636 | 0.662 | 0.697 | 0.691 | 0.700 | 0.690 | better / higher, sep | better / higher, sep |
| fin tip inside the largest fin region (hr, clicks 03-18) | 119 | 0.956 | 0.939 | 0.991 | 0.955 | 0.964 | 0.965 | - / n.s. | better / n.s. |
| finless backs called dorsal, by eye | 33 | 0.182 | 0.242 | 0.091 | 0.121 | 0.091 | 0.121 | better / lower, sep | better / lower, sep |
| finless backs called dorsal, classifier | 151 | 0.139 | 0.152 | 0.086 | 0.086 | 0.099 | 0.093 | better / lower, sep | better / lower, sep |
| three-way call by eye, crops | 79 | 0.886 | 0.861 | 0.924 | 0.924 | 0.937 | 0.899 | better / higher, sep | better / n.s. |
| three-way call by eye, whole photo | 79 | 0.911 | 0.848 | 0.873 | 0.911 | 0.886 | 0.848 | - / n.s. | - / n.s. |
| three-way call by classifier, whole photo | 828 | 0.949 | 0.930 | 0.948 | 0.963 | 0.951 | 0.942 | - / n.s. | - / n.s. |
| fluke notch inside the fluke (hr, clicks 03-18) | 54 | 0.812 | 0.818 | 0.812 | 0.809 | 0.766 | 0.809 | - / n.s. | worse / n.s. |
| pilot val dorsal IoU (rule A, 1280) | 495 | 0.308 | 0.239 | 0.736 | 0.737 | 0.789 | 0.794 | better / higher, sep | better / higher, sep |
| pilot val fluke IoU (1280) | 495 | 0.532 | 0.575 | 0.912 | 0.913 | 0.925 | 0.926 | better / higher, sep | better / higher, sep |
| run05 val dorsal IoU (rule-A label) | 1379 | 0.666 | 0.653 | 0.651 | 0.637 | 0.122 | 0.102 | worse / n.s. | worse / lower, sep |
| run05 val fluke IoU | 1379 | 0.938 | 0.941 | 0.945 | 0.946 | 0.884 | 0.884 | better / higher, sep | worse / lower, sep |

**P_add: HELPS** (val total: better; standing measures worse for both seeds: none; better for both: ['finless backs called dorsal, by eye', 'finless backs called dorsal, classifier', 'three-way call by eye, crops']; categories moved for both seeds: {'VAL F_low error': 'worse', 'VAL F_edge error': 'better', 'VAL B_back error': 'better'})

**P_only: not separable at this size** (val total: -; standing measures worse for both seeds: ['fluke notch inside the fluke (hr, clicks 03-18)']; better for both: ['fin tip inside the largest fin region (hr, clicks 03-18)', 'finless backs called dorsal, by eye', 'finless backs called dorsal, classifier', 'three-way call by eye, crops']; categories moved for both seeds: {'VAL F_low error': 'worse', 'VAL F_edge error': 'better', 'VAL D_tip error': 'worse', 'VAL B_back error': 'better'})

Inputs (all made by unchanged scripts, see run_p22.sh / run_p22_compare.sh): part-gap reports (moonshot_cache/pilot22/d/part_gaps/report_<model>_hr.json), new_measures_p22.json (pred_fill.py score), compare_p22.json (compare.py), finless_p22.json (finless.py), synth_val/<model>.json (w3e_scores.py synth), pilot_val/<model>.json (score_pilot.py). Rule (plan.md): an arm "helps" when both its seeds have a lower validation-day total part-gap cost than both W3e seeds and no standing measure is worse for both seeds (both beyond the worse W3e seed on the bad side); "hurts" the mirror; else "not separable at this size". "both" column: "better" / "worse" when both arm seeds lie beyond both W3e seeds, else "-"; "sep" column: W3c's separability rule (gap larger than the larger seed distance) for reference.
