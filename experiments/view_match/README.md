# View-aware matching: what the fluke / dorsal / other label does for the match score (2026-11-12)

Question (user, after the view classifier of `../view_class`): can the label improve the match score and the match
suggestions of the Animals step, and the merge suggestions of the Review step?

Short answer.
- **Animals step: yes, three ways, all live.** (1) the phases and the frame a stack opens on use the classifier's
  labels; (2) a fluke / fin similarity only counts between instances that really show the part; (3) for dorsal fins
  the ranking blends in a fin-specific similarity built from the classifier's own features. First choice right on the
  held-out days: eval 69.6 % -> 71.7 % -> **75.1 %**, val 77.5 % -> 80.5 % -> **83.1 %**; every one of the 7 days up.
  The number of decisions does not change (541 against 542).
- **Review step: the merge hints are left on the similarity they were tuned on.** The better similarity ranks
  clean split cases better, but it scores the split pairs of the one real test run lower (they were split because
  they look different), so fewer of them would pass the thresholds. A "no view in common" hint class was measured
  and is too small to be worth building.
- **Flukes: no gain.** The same recipe for flukes scores pairs better (AUC) and changes nothing in the fluke phase's
  first choice, so the fluke part of the ranking is unchanged. Fluke matching remains the weak spot.

## 1. Which comparisons carry information (`vm_exp1.py`)

Stack pairs inside catalogue encounters, same animal against different animals, AUC of the student embedding's part
cosine (what the page had):

| | every pair with two part charts | both stacks really show the part (classifier) | chart, but the classifier says no |
|---|---|---|---|
| dorsal fin, val | 0.724 | **0.779** | 0.658 |
| dorsal fin, train (embedding in-sample) | 0.779 | 0.829 | 0.694 |
| fluke, train | 0.717 | 0.756 | 0.652 |
| fluke, val (283 pairs) | 0.696 | 0.657 | 0.668 |

A third of the "fin" comparisons were between a fin and a back with no fin; removing them is worth 5 points of AUC for
the fin similarity. Coverage: 11-13 % of the stacks that show a fluke have no fluke chart at all (2 % for fins), so the
student has nothing to compare them by.

## 2. A view-specific similarity from the classifier's features (`vm_exp3.py`, `vm_exp3b.py`, `vm_fit.py`)

For the instances that show a part: the frozen features the classifier already computes, centred on the day,
whitened (PCA 128), scaled down along the directions in which one animal varies (within-animal whitening, shrink
0.3), reduced to the 32 directions that separate the animals of a day (LDA with the between-animal scatter taken inside
each day). Fitted on the catalogue's animals of the 21 train days (fluke 2 937 instances of 233 animals, dorsal fin
4 133 of 269); every train day also gets a projection fitted without it. Production: `cluster/view_match.py embed`
-> `DAY/safe/view_match.npz`, model `experiments/safe_clusters/view_match.pkl`.

Stack pairs inside encounters, out of fold (AUC):

| | new | student | both (rank mean) |
|---|---|---|---|
| dorsal fin, val (3 268 pairs; neither model saw these days) | **0.811** | 0.777 | 0.819 |
| dorsal fin, val, pairs more than 10 min apart | 0.823 | 0.770 | 0.822 |
| dorsal fin, train (student in-sample, new out of fold) | 0.744 | 0.836 | 0.817 |
| fluke, val (283 pairs) | 0.799 | 0.657 | 0.752 |
| fluke, train (student in-sample) | 0.744 | 0.757 | 0.775 |

The time gap alone gives 0.49-0.56 beyond 10 minutes, so this is identity, not "same moment". On days neither model
saw, the new similarity is ahead; on the train days the student is ahead because it was trained on those animals.

## 3. On the page: the catalogue replay (`Whale-Review-Web/tests/labeling_replay.py`, logs in `logs/`)

First choice right / looks (animals held against a stack before the answer); decisions are 541-542 (eval), 311-315
(val), 2 017-2 030 (train) throughout.

| | val (3 days) | eval (4 days) | train (21 days) |
|---|---|---|---|
| chart flags (the page until 2026-11-12) | 0.775 / 487 | 0.696 / 929 | 0.813 / 3 294 |
| classifier labels | 0.795 / 474 | 0.696 / 925 | 0.814 / 3 324 |
| + similarities gated by the labels | 0.805 / 461 | 0.717 / 902 | 0.822 / 3 291 |
| + view-specific similarity, fluke and fin | 0.824 / 459 | 0.740 / 876 | 0.798 / 3 363 (blend forced on) |
| + view-specific similarity, **fin only (shipped)** | **0.831 / 455** | **0.751 / 873** | 0.822 (blend off on these days) |

Per day, chart -> gated -> shipped: 47/63, 46/61, 49/61 · 68/92, 70/92, 72/92 · 99/121, 103/119, 105/119 (val);
56/69, 56/71, 58/71 · 133/208, 141/208, 149/208 · 76/100, 75/97, 78/97 · 65/97, 67/97, 70/97 (eval).
By phase on eval, gated -> shipped: flukes 114 -> 115 of 168, dorsal fins 167 -> **182** of 245, the rest 58 of 60.

Choices and how they were made:
- *Blend, not replace*: the two calibrated logits are averaged where both exist (val: blend 0.824, replace 0.827; train
  with the in-sample student: 0.801 against 0.774). The result is written on the scale of the student cosine, so the
  page and `labeling_model.json` are untouched (backend `_match_code`).
- *Not on days the student was trained on* (`student_in_sample` in the npz): there the student's cosine alone is
  better (0.822 against 0.798) because it has seen those animals. A new day is never such a day.
- *Fin only*: on val the fluke part made no difference (fluke-phase first choice 36 / 35 / 34 of 42 for off / fill /
  blend); the eval read of the both-parts blend had shown 110 against 114 of 168. `WHALE_VIEW_MATCH_FLUKE=mean|fill`
  turns it on.
- Eval days were read three times for this section (both-parts blend with two calibrations, then the shipped
  setting); the fluke choice was made knowing the first read. Val alone points the same way for every choice.

## 4. Merge hints (Review): measured, not changed

The hints compare whole animals by the stack average of the same blend (`animalSim`), with thresholds tuned on the
student-based similarity (2026-11-12, `../labeling_runs/merge_hints`). Feeding them the new similarity:

- **Fixed split cases** (`sim.py`: one animal left in two pieces by rule; AUC of the hint score against pairs of
  different animals; fluke stacks apart / one fin side apart / longest gap / one stack / 1-3 stacks):
  eval 0.879 / 0.898 / 0.533 / 0.922 / 0.907 -> 0.877 / 0.899 / 0.561 / 0.926 / 0.921; val 0.832 / 0.876 / 0.672 /
  0.916 / 0.878 -> 0.853 / 0.896 / 0.721 / 0.927 / 0.893. Better or equal.
- **The real splits of the 2024-12-23 test run** (`user_day.py`): the three split whales score 0.34 / 0.32 / 0.28
  with the student-based similarity (best wrong pair 0.20; all three hinted) and 0.28 / 0.19 / 0.19 with the blend
  (best wrong 0.18; one hinted). A train day, so the student is in-sample; three pairs.
- **A researcher who splits where the page's similarity is low** (`sim2.py`, `sim2_counts.py`, `sim2_cross.py`; eval,
  per run of 4 days; blend = the fluke-and-fin blend): split whales without a hint 14.6 (student for both) -> 18.4
  (blend for both); with the blend on
  the page and the student in the hints 15.6; hints from either score 11.6-14.0 at 35 wrong hints instead of 27.
  A hint works best when it does not come from the score that produced the split.

So the ranking of step 2 reads the blend (`match.bin`) and the hints keep `pairs.bin`, byte for byte what they were
checked on. Showing hints from both scores is the one variant that reaches more split whales (about 3 more of 35 for
8 more wrong hints per 4 days): a threshold trade, not built.

**"No view in common"** (`sim2_shared.py`, `sim2_shared_runs.py`, `sim2_noview.py`): pairs of pieces that share neither
a fluke nor a fin hold 23-34 % of the simulated true split pairs (one such pair in four or five is one whale; the
other kinds of pairs: 7-26 %), and the rule shows almost none of them. A hint class for them (each the other's closest such piece,
similarity >= 0.15) would add 0.7-0.8 hints per day, 50-61 % right, and reach 3-5 % more of the split whales, because
most of those whales already get a hint through another pair. Not built.

## Files

`vm_common.py` (the page's inputs per day through `../labeling_runs/merge_hints/sim.py`), `vm_exp1.py`, `vm_exp3.py`,
`vm_exp3b.py`, `vm_fit.py` (projections, leave-one-day-out, calibration; `fit.log`), `vm_calibrate.py`,
`embed_all.sh`; `fetch_inputs.py` (page inputs from a test server -> `inputs_*/cache`, ignored by git),
`sim2_*.py` (run from a folder with `cache/`, `PYTHONPATH=../labeling_runs/merge_hints`), `logs/` (replay outputs).
The calibration of the new similarity is on the page's scale: slope from the 24 train + val days out of sample,
intercept shifted to the val days' base rate, where the page's own calibration of the student cosine was fitted.
