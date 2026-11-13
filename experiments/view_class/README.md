# View class: fluke / dorsal fin / other (2026-11-12)

Question (user): instead of mapping every photo onto a 3D mesh, is it enough to find confidently which photos show a
**fluke**, which a **dorsal fin**, and call the rest **other**, so that the Photo Labeling page can put the stacks with
a fluke or a dorsal fin first?

Answer: yes. Two small classifiers on frozen DINOv3 features of the whale instance do it at 95-97 % three-way accuracy
on days they never saw, against 84-88 % for the flags the page used until now (the synthetic-trained canonical map of
`cluster/parts.py`). The production code is `model_fine_tuning/cluster/view_class.py`; this folder is the experiment:
labels, scripts, logs.

## What is classified

One whale **instance** (a mask component of a photo, key `<stem>~k`, the same unit the safe clusters are made of), from
the pixels of its box in the day's 2560-px image. Two yes / no answers, so a frame can show both:

| answer | yes when | not when |
|---|---|---|
| fluke | the tail fluke is out of the water: both blades, one blade with its tip, the underside or the top, from behind or side-on | a tail stock arching with the blades still under water, a pectoral fin, a fluke-shaped splash |
| dorsal fin | the fin is visible above the water as a fin (any side, also from behind) | a back without the fin, a bump at the size of a few pixels, the knuckles of the tail stock |

"other" = neither: head, blow, a back without the fin, pectoral fin, tail stock, breach, belly, a whale under water.
The three classes of the page: fluke if the fluke says yes, else dorsal if the fin says yes, else other.

## Labels (all in this folder)

| source | what | count |
|---|---|---|
| `view_labels.json`, round `r0` | random instances of the 24 train + val days, labelled by eye from contact sheets, no model output shown | 432 |
| round `ev` | random instances of the 4 eval days, labelled blind before any model was scored on them | 144 |
| rounds `al1`, `al2` | hard cases on train + val days: where the first classifier was unsure or said "other"; where classifier and chart flags disagree | 216 |
| rounds `aud`, `aud2` | audits of stacks whose phase changes (train + val for the first classifier; eval days for the final one) | 151 |
| `other_labels.json` researcher | keypoints the researcher clicked on the Pairs / Bursts pages (2024-12-18, 2025-01-02, 2025-03-18): a clicked dorsal / fluke point = the part is there; clicks are attributed to the instance whose box holds them | 627 instances |
| `other_labels.json` catalog | photos filed in the Lightroom collection "2025 Flukes - ALL" (and the per-photographer fluke collections): fluke positives chosen by hand | 296 photos |

Codes of the by-eye labels: `F` / `D` clear, `f` / `d` marginal (tiny, mostly under water, in spray, cut by the frame;
no loss in training, kept out of the accuracy figures), `O` other, `X` not a whale. `vc_common.all_targets()` merges
the sources into `targets.json` (1 783 instances; 1 360 on train + val days, of which fluke 574 yes / 653 no, dorsal fin
437 / 614). **By eye = labelled by Claude from contact sheets, not by the researcher**, and every accuracy figure on
a random sample below is measured against those labels. How far the two agree, on the 39 instances that carry both
(mostly from the hard-case rounds, so a harsh sample): where the researcher clicked only the other part, the by-eye
label says the part is absent 36 times of 36; of the 36 parts the researcher clicked, the by-eye label says yes 28
times, marginal 4, no 4 (3 dorsal fins, 1 fluke); the 11 catalogue fluke photos with a by-eye label are all `F`. So
the by-eye labels miss some faint parts the researcher sees; where the two differ the researcher's click wins in
`targets.json`. Against the researcher's clicks and the catalogue flukes of the eval days (never fitted on) the final
classifier finds 93 of 97 flukes and 113 of 117 dorsal fins with one false call.

Two things learned while labelling: a single frame of a blade from the side cannot always be told from a pectoral fin,
but the burst can (`vc_context.py` shows the two frames before and after; 15 of 25 unsure cases were flukes); and the
"no keypoints" tick of the Pairs page often means "a fluke with no clickable tip", so it counts as a marginal fluke.

## Model

Frozen DINOv3 ViT-B/16 (last block, final norm), cls token + mean patch token of two crops of the box (x 1.15):
a 448-px square and a crop that keeps the box's shape (1:1 to 3:1, long side up to 672 px) = 4 x 768 numbers, each
block at unit length. Standardise, PCA 128, RBF SVM (C 4, balanced), one for the fluke and one for the fin; the
decision value becomes a probability through a logistic fitted on out-of-fold scores of the random sample `r0`.
Fitted on the train + val labels only: `experiments/safe_clusters/view_class.pkl` (`view_class_v2`).

How it was chosen (day-grouped 8-fold cross-validation on train + val, `vc_compare.py`, `vc_compare2.py`; three-way
accuracy on all labels, where hard cases are over-represented, and on the random sample):

| features, head | all labels | random sample |
|---|---|---|
| 224-px square crop (the cached `inst_feat.npz`), logistic regression = the first version | 0.890 | 0.954 |
| the same + the chart flags as two more inputs | 0.894 | 0.957 |
| 448-px square crop, logistic regression | 0.903 | 0.962 |
| shape-keeping crop, logistic regression | 0.903 | 0.957 |
| 448-px square + shape-keeping crop, logistic regression | 0.907 | 0.957 |
| 448-px square + shape-keeping crop, RBF SVM = the final model | 0.927 | 0.962 |

More resolution and a non-linear head help the hard cases (0.89 -> 0.93); on a random photo the first version was
already at 95 %.

## Results

Three-way accuracy on the instances with both parts clearly labelled; rows of the matrices = true fluke / dorsal /
other, columns = predicted.

| set | classifier | chart flags (the page until now) |
|---|---|---|
| train + val days, random sample, day-grouped cross-validation (368) | **0.967** `[[147,0,5],[0,182,2],[4,1,27]]` | 0.875 `[[122,22,8],[0,184,0],[6,10,16]]` |
| eval days, random sample labelled blind (132) | **0.970** `[[65,0,1],[0,55,1],[1,1,8]]` | 0.841 `[[53,12,1],[0,56,0],[2,6,2]]` |

Per answer on the eval random sample: fluke precision 0.971 (66 / 68), recall 0.985 (66 / 67); dorsal fin precision
0.965 (55 / 57), recall 0.982 (55 / 56). At P >= 0.9 both are 100 % precise there (59 flukes, 54 fins) at recall 0.88 /
0.96. Per eval day: 35 / 35, 32 / 34, 31 / 32, 30 / 31. Chart flags on the same sample: fluke precision 0.964, recall
0.806 (13 of 67 flukes missed); dorsal fin precision 0.718 (22 of 78 "fin" flags are no fin), recall 1.0.

What the chart flags get wrong: they call a fin on almost any back (precision 0.72-0.79), they miss a fifth of the
flukes, and they have nearly no "other" (3 % of instances; the classifier says 14 %).

All 11 810 instances of the 28 days: fluke 37.6 % (34.4 % with P >= 0.9), dorsal fin only 48.4 % (46.0 % sure), other
13.9 %. The unsure band (P 0.5-0.9) is 3.3 % of instances for the fluke and 2.4 % for the fin.

Eval days and their use: read twice for this classifier, once for the first version (three-way 0.939 on the same
random sample, `fit_final.log`) and once for the final one (`eval_v2.log`); the final recipe was chosen on the train +
val cross-validation, not on the eval read. No eval label was ever fitted on.

### Where it is weak

- "other" is the hard class: 8 of 10 right on the eval sample, 27 of 32 in cross-validation, 131 of 160 on the
  hard-case-heavy set. Typical misses: a pectoral fin or a tail stock called a fluke, a rounded hump called a fin.
- Audit of the stacks whose phase changes on the eval days (146 of 967; 54 looked at blind, 9 per kind of change,
  `audit2_meta.json`): on the disputed part the classifier was right 39 times, the chart flag 9, 6 marginal.

  | change (chart -> classifier) | stacks on eval days | classifier right | chart right | marginal |
  |---|---|---|---|---|
  | fluke -> fin | 28 | 8 | 0 | 1 |
  | fluke -> rest | 19 | 8 | 0 | 1 |
  | fin -> fluke | 27 | 7 | 1 | 1 |
  | fin -> rest | 43 | 8 | 1 | 0 |
  | rest -> fin | 11 | 4 | 3 | 2 |
  | rest -> fluke | 18 | 4 | 4 | 1 |

  Where the classifier finds a part on a stack the charts had as neither (29 of the 146 changes) it is a coin flip:
  that is where pectoral fins and tail stocks sit. Everywhere else it is right about 31 times in 33.
- Marginal cases (a tenth of the labels) are not in the accuracy figures; the classifier calls about half of them yes.

## Synthetic data: tested, not used in the shipped model

`vc_synth.py` gives the 16 917 whale shots of the factory's run04 + run05 exact labels from the render metadata
(keypoint states, part pixels) and the same frozen features; `vc_synth_test.py` and `vc_final_cv.py` add them to the
training side of the same cross-validation, real photos only on the test side (three-way accuracy, all labels / random
sample):

| training data | first version (224 px, logistic) | final recipe |
|---|---|---|
| real photos only | 0.890 / 0.954 | 0.927 / 0.962 |
| synthetic only | 0.624 / 0.704 (0.720 / 0.821 with each domain centred on its own mean) | - |
| real + synthetic (total synthetic weight 0.1 / 0.3 / 1.0 x real) | 0.890 / 0.957, 0.888 / 0.957, 0.889 / 0.965 | 0.920 / 0.954, 0.918 / 0.951, 0.916 / 0.948 |
| real + synthetic 0.3, domains centred | 0.893 / 0.965 | - |

Learning curve, first version, N real labels per classifier (mean of 3 draws; all labels / random sample):

| N real | real only | + synthetic | + synthetic, centred |
|---|---|---|---|
| 50 | 0.806 / 0.897 | 0.830 / 0.918 | 0.839 / 0.928 |
| 100 | 0.835 / 0.924 | 0.846 / 0.934 | 0.855 / 0.936 |
| 200 | 0.867 / 0.939 | 0.866 / 0.948 | 0.875 / 0.949 |
| 400 | 0.872 / 0.950 | 0.878 / 0.951 | 0.882 / 0.956 |

Reading: synthetic shots alone do not transfer (the renders sit in another corner of the feature space: cosine of
the two domain means 0.80), they are worth about a doubling of the real labels when there are 50-100 of them, and
nothing measurable with the 1 360 there are now. So the shipped model is fitted on real photos. Synthetic data is
still in the pipeline one step earlier: the whale masks, and so the instance boxes this classifier looks at, come
from the synthetic-trained segmentation / canonical models.

## Effect on the Photo Labeling page

`Whale-Review-Web/backend/labeling.py` reads `DAY/safe/view_class.json` (when the day has one) and takes the fluke /
dorsal-fin bit of every instance from it instead of from the chart flags; the fin side and the fluke face still come
from the charts. 17 % of the stacks of the 28 days change phase (chart: fluke 2 092 / fin 2 581 / rest 367; classifier:
1 988 / 2 349 / 703). On the single-photo stacks of the blind eval sample the phase is right for 44 of 45, against 39.

The page's own catalogue replay (`Whale-Review-Web/tests/labeling_replay.py`; the catalogue plays the researcher):

| days | labels | decisions | first choice right | top three | looks |
|---|---|---|---|---|---|
| val (3) | chart flags | 315 | 0.775 | 0.967 | 487 |
| | classifier | 312 | 0.795 | 0.974 | 474 |
| | classifier + gated similarities (shipped) | 311 | 0.805 | 0.982 | 461 |
| eval (4) | chart flags | 542 | 0.696 | 0.954 | 929 |
| | classifier | 541 | 0.696 | 0.953 | 925 |
| | classifier + gated similarities (shipped) | 541 | 0.717 | 0.958 | 902 |
| train (21) | chart flags | 2 017 | 0.813 | 0.957 | 3 294 |
| | classifier | 2 030 | 0.814 | 0.954 | 3 324 |
| | classifier + gated similarities (shipped) | 2 030 | 0.822 | 0.957 | 3 291 |

"Gated similarities" (`WHALE_VIEW_GATE=blend`, the default): the fluke and dorsal-fin similarities of the ranking
compare two stacks only through instances the classifier says show that part (a "fluke" chart cut from a back no
longer counts as a fluke match). The link model keeps the inputs it was fitted on. `WHALE_VIEW_GATE=all` also gates
the link model's inputs: 3-5 % fewer decisions (eval 518, val 298, train 1 948) through longer runs, at run precision
0.907 against 0.922 on eval; not shipped, it needs the link model refitted on gated inputs first.

Reading: the labels are right far more often, the count of decisions barely moves (the order of work costs nothing
either way, as measured on 2026-11-12 when the order was introduced), and the first choice is right a little more often
(+1 to +3 points). What the researcher gets is a fluke phase that holds flukes and a "rest" pile that holds what cannot
be told by a fluke or a fin, twice as large as before. The gating variant was picked with the eval replay in view; val
and train alone point the same way.

The merge hints of the Review step (thresholds chosen the same day on the chart-flag inputs,
`experiments/labeling_runs/merge_hints/sim2.py`) behave the same on the new inputs: split whales reached eval 0.60 ->
0.60, train 0.69 -> 0.71, val 0.65 -> 0.58 (3 days, noisy: the unchanged link rule moves as much there); wrong hints
per encounter 1.18 -> 1.21, 0.85 -> 0.87, 0.71 -> 0.78.

## How to

New day (after `safe_clusters features`, which writes `DAY/safe/photos.json`):

    python -m model_fine_tuning.cluster.view_class features --day-dir experiments/days_all/DAY     # GPU, 10-20 s
    python -m model_fine_tuning.cluster.view_class predict  --day-dir experiments/days_all/DAY     # -> DAY/safe/view_class.json

then restart the review server (it reads a day once). A day without `view_class.json` falls back to the chart flags,
and so does a single instance whose box has changed since its features were cut (the file records the boxes): a day
whose instances were rebuilt needs both commands again.
`WHALE_VIEW_CLASS=0` on the server = chart flags everywhere.

More labels, refit:

    cd experiments/view_class
    python vc_sample.py NAME PER_DAY SEED train,val && python vc_sheet.py NAME sheets/NAME_keys.json   # sheets/NAME_000.jpg ...
    python vc_label.py NAME 000 "F D O d ..."                # one code per tile
    python -c "import json; from vc_common import *; json.dump(all_targets(), open('targets.json', 'w'), indent=0, sort_keys=True)"
    cd ../.. && python -m model_fine_tuning.cluster.view_class fit && experiments/view_class/predict_all.sh
    python -m model_fine_tuning.cluster.view_class report --split val       # eval: sparingly

## Files

`vc_common.py` loading and the merge of the label sources; `vc_sample.py`, `vc_sheet.py`, `vc_label.py`,
`vc_context.py`, `vc_relabel.py` the labelling loop; `vc_sources.py` researcher clicks + catalogue flukes;
`vc_fit.py` the first version (logistic on `inst_feat.npz`; `view_class_v1_linear.pkl`, `fit_final.log`);
`vc_feats.py`, `vc_compare.py`, `vc_compare2.py` feature / head comparison; `vc_synth.py`, `vc_synth_test.py`,
`vc_final_cv.py` the synthetic test; `features_all.sh`, `predict_all.sh` all days; `*.log` the outputs quoted above.
`sheets/` holds the contact sheets (crops of the research photos, a few with crew on the bow: not for a public
repository; ignored by git from now on, rebuilt from the `*_keys.json` lists) and the tile -> key indexes.
`feats/` (350 MB, ignored) are the experiment's feature caches; the production features are `DAY/safe/view_feat.npz`.
