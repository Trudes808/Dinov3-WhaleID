# Tape colour, round 2 (colour2), 2026-11-23

The marker reader (`model_fine_tuning/seg_multi/markers.py`) read every green tape as blue (0 of 5 over the two unculled
days) and missed 2 pink ones. Both unculled days (2026-04-07 dev, 2026-03-30 the held-out day scored at 05:02) have now been
looked at for this rule: **neither is a test any more; a fresh unculled day is needed to call the new rule tested.**

Code: `colour2_measure.py` (band measurements), `colour2_rules.py` (candidate rules, cross-day scores, renders),
`colour2_sheet.py` (sheet; written outside the repo). Outputs (untracked): `moonshot_cache/sam3_labeller/v3/day_runs_colour2/`
(new records for both days, `bands_real.json`, `bands_renders.json`, `rules_scores.json`, `markers_py_before_colour2.py`),
sheet `review_sheets/labeling_markers/colour2_bands_sheet.jpg`.

## Cause (measured inside the scene model's tape component, on the 2560-px day JPEG)
- Dark green tape images as dark teal: HSV hue 170-187, V 0.23-0.29. The frozen C3 ranges put green at 70-170, so every
  green falls into blue (170-260). Real blue: hue 211-219, S 0.57-0.94. In HSV the two sit only ~25 degrees apart.
- Band choice: the reader reads the largest tape component. The stanchions carry pink + yellow on one side and blue +
  green on the other, so 10 of 37 photos show two bands of near-equal size (ratio 0.56-0.99). In 216 and 238 the largest
  is the off-centre blue band; in 193 and 203 the largest is the off-centre green band and the reader was right only
  because green read as blue (two faults cancelling).
- 027 (pink, half hidden): HSV hue 307 under a bluish light (tube RGB 141/151/157) falls in "other" (260-320).
- 068 (pink, out of focus): band saturation mean 0.15; only 318 of 23,924 band pixels pass S >= 0.25 and those are bluish
  fringe pixels (hue 227) -> blue. Its mean colour is pink (CIE a* +6.8, b* -1.3).
- White balance is not the cause: the white tube beside the band is near neutral in most photos, and dividing by it
  (labwb_nn) does not separate the classes better than plain Lab.

## Rule (markers.py `--colour-rule lab_centre`, the default; `old` = frozen C3 rule)
Band = among tape components with >= 25 % of the largest's area, the one whose centroid is nearest the frame centre (the
truth's definition: the photographer focuses on the band meant). Colour = hue angle of the band's mean CIE a*, b*
(eroded as before, >= 200 px else none), nearest of red 13, yellow 83, green 198, blue 284 degrees (midpoint
boundaries 48 / 141 / 241 / 329). In Lab the real greens sit at 184-211 and blues at 279-287: ~70 degrees apart.
Prototypes are the circular means of the labelled bands of both days (centre bands + the off-centre bands the 03-30
truth notes name), rounded. No "other" class any more.

## Validation (fit prototypes on one day, score the other; n = rail markers)
| fit -> scored | old reader | largest band + Lab | centre band + HSV-nn | centre band + Lab (chosen) |
|---|---|---|---|---|
| 2026-04-07 -> 2026-03-30 (21) | 17 | 18 | 21 | 21 |
| 2026-03-30 -> 2026-04-07 (13) | 10 | 13 | 13 | 13 |
Crew photos 137 / 140 / 141 (bands, not markers): 3 of 3 for every rule. The HSV-nn, Lab, white-balanced Lab and a*b*
nearest-centroid variants tie on counts; Lab hue was chosen on its margins (worst margin 15.5 / 40 degrees against
11.4 / 21 for HSV), a choice that saw both days. Band choice could not be tested on 2026-04-07 (one band per marker
there); it rests on the truth's definition and 2026-03-30 (10 two-band photos).

Renders (r06r_boat20b_look, r06r_boat20_look, r06r_boat22_markers rail shots; the colour rule never saw them):
held-out 28 with the frozen model's maps old 26 / new 28; all 170 with exact tape labels old 159 / new 170. Render greens
are yellow-green (Lab hue 142-200), not real teal: the lowest sits 1.5 degrees from the yellow / green boundary, and with
prototypes from 2026-04-07 alone 2 of 150 render greens read yellow.

## Records with the new rule (cached scene maps, no GPU)
`--colour-rule old` reproduces the scored records byte for byte (both days). With the default: kinds, refers_to and every
other field unchanged; colour of the rail markers 2026-04-07 10 -> 13 of 13 (027 other -> red, 068 blue -> red, 078 blue
-> green), 2026-03-30 17 -> 21 of 21 (174, 216, 238, 267 blue -> green). tape_colour_any on other photos (informational, tiny
bands of about 0.01 % of the frame): 2026-04-07 16 non-markers (10 blue -> green, 4 red -> yellow, 2 none -> yellow) and
2 bollard records (136 none -> blue, 172 none -> green); 2026-03-30 1 (221 red -> yellow); none of these checked by eye.
New fields: colour_rule, tape_hue_ab, tape_band_dist.

Note: run_new_day.sh checks markers.py against `v3/FREEZE.json`; with this change it stops with FREEZE MISMATCH until a
new freeze is written.
