# pilot22d — does the pilot r06_pilot22 improve the part model on REAL photos? (plan, written 2026-11-23 before any training)

Written before any training of a new arm. Baseline (W3e) numbers below were read before writing (they are the
reference, not a result of this test). Not to be changed after results are seen.

## Arms
- Baseline: W3e seeds 0 / 1, the existing checkpoints
  `/var/tmp/sat3737-parts-cache/parts_weak/runs/parts_seg_r05_w3e{,s1}/checkpoints/best.pt` (not retrained).
- P_add (seeds 0, 1): W3e's command unchanged (`experiments/parts_tri_expand/run_w3e.sh`: parts_weak.train --n-real 2
  --w-need 0.5 --w-whale 1.0 --maps-dir maps_tri3 -- --cache run05/labels_triA --init-weights phase1_last.pt
  --epochs 10 --eval-every 5 --warmup-steps 100 --batch-size 8 --crop 512 --seed S --workers 5) run through
  `model_fine_tuning/pilot22_train/train.py --pilot-share 0.3333333`: the training render stream draws one third of
  its samples from the pilot, two thirds from run05. Same steps per epoch (805 batches of 8 render + 2 real crops),
  same schedule, same real weak crops, same class weights (run05), same validation (run05 val, rule-A labels) and
  checkpoint choice (best of epochs 5 / 10 by run05 val fluke + dorsal IoU).
  Names `parts_seg_p22add`, `parts_seg_p22adds1`.
- P_only (seeds 0, 1; only if GPU time is left after P_add's two seeds are trained AND scored, within 8 GPU hours):
  the same with `--pilot-share 1` (no run05 shot in the training stream; validation / checkpoint choice / class
  weights still run05's, as the recipe). Names `parts_seg_p22only`, `parts_seg_p22onlys1`.

Why one third (fixed, not tuned): by count the pilot is 1191 / (1191 + 6440) = 16 % of the shots; it was rendered for
the views where the model fails on real photos, so it is weighted up, but kept the minority so run05 (the source of
every earlier result) still dominates; one third = each pilot shot about 18 times over training, each run05 shot
about 6.7 times (W3e: 10). Equal steps rather than extra steps: so the arm differs from W3e in data composition only,
not in compute or in the amount of real weak supervision. Consequence to keep in mind: run05 is seen 2/3 as often.

## Pilot data (prep.py, cache `/home/sat3737/NOAA_whales/moonshot_cache/pilot22/d/cache`)
- train: pilot train split 1345 - 61 shots with individual i12 - 0 rejected (rejected shots are not in the splits)
  - 16 run05 defect flags - 77 under run05's min-visible filter (800 px at 1280) = 1191 shots (23 negatives).
- val (render scoring only, never used for any choice): pilot val 507 - 12 i12 = 495 (0 rejected in the split).
- 2560 x 1707 -> 1280 x 854 (images INTER_AREA, labels nearest): run05's pixel scale (median sqrt(visible px):
  run05 91 at 1280, pilot 101 at 1280).
- label = factory `parts_triA` (rule-A dorsal, awash variant as rendered; 255 undefined), through-water zone
  (masks == 2 on background) -> ignore, as run05's cache. Checked by eye on 8 crops (sheet in scratch).

## Measures (all existing; no new thresholds)
1. Part-gap instrument (`model_fine_tuning/part_gaps/report.py`, unchanged, `--out` into
   `moonshot_cache/pilot22/d/part_gaps/`, variant hr): rule chosen on the training days per model (as always);
   REPORTED: validation days (2025-01-15, 2025-02-27, 2025-03-14) total cost = weighted error (`overall_val`), n = 195
   labelled instances, and the per-category rows F_low, F_edge, F_up, D_small, D_tip, D_tall, B_back, P_pec, H_head
   (error, k / n, cost). Training-day total listed for reference.
2. IoU with the expanded-triangle truth, held-out clicks 2025-03-18 (`experiments/parts_tri_expand/pred_fill.py score`,
   unchanged), n = 99 fins.
3. Standing measures of `two_seeds_w3e.md`, same scripts (`experiments/parts_weak/compare.py`, `finless.py`):
   fin tip inside the largest fin region (hr, clicks 03-18); finless backs called dorsal by eye (n 33) and by
   classifier (n 151); three-way call by eye crops (n 79), by eye whole photo (n 79), by classifier whole photo;
   fluke notch inside the fluke (hr, clicks 03-18).
4. Render side: pilot val (495 shots, full frame at 1280, pooled pixel IoU against the prep labels; new script
   `pilot22_train/score_pilot.py`) dorsal and fluke IoU; run05 val dorsal (rule-A label) and fluke IoU
   (`experiments/parts_tri_expand/w3e_scores.py synth`, unchanged).
Real-photo outputs via `experiments/parts_weak/eval_w.py` (unchanged; predict train + val days into the parts_weak
shadow root). Evaluation days 2024-12-18, 2024-12-26, 2025-02-19, 2025-03-21 never read.

## Decision rule (fixed)
- "The pilot helps" (per arm): BOTH seeds of the arm have a LOWER validation-day total part-gap cost than BOTH W3e
  seeds (i.e. arm max < min(0.1640, 0.1547) = 0.1547; numbers of the amendment below), AND no measure of (3) is worse for both seeds (worse for both =
  both arm seeds beyond the worse W3e seed on the bad side; bad side = lower for fin tip / three-way / notch, higher for
  finless backs called dorsal).
- "Hurts": both seeds higher than both W3e seeds on the validation-day total (arm min > 0.1640) AND no measure of (3)
  better for both seeds.
- Anything else: "not separable at this size".
- Also reported whatever the total says: the categories whose validation-day error moved the same way for both seeds
  past both W3e seeds (both lower than the lower W3e seed, or both higher than the higher one), with k / n.
- Measures (2) and (4) are reported with the same both-seeds reading, outside the rule.

## What this test can and cannot see
- Baseline seeds: validation-day total cost W3e s0 0.1616 (37 of 195 wrong), s1 0.1500 (33 of 195): seed spread 0.012,
  i.e. about 2-4 instances. Training-day totals 0.157 / 0.176 (n 386). B_back alone is 0.09 of the 0.16 (16 / 14 of 24
  wrong); then D_small 0.024 / 0.026, P_pec 0.020 / 0.009, F_low 0.015 / 0.015 (4 of 22), F_edge 0.010.
- One instance is about 0.005 of the weighted total. A paired difference on 195 instances with ~15 discordant has a
  standard error of about 0.02. So the rule can only call a change of roughly 0.03-0.04 or more in the total
  (about a fifth of the cost, 6-8 instances) with any reliability; a real gain of 0.01-0.02 will most likely read
  "not separable". "Not separable" is therefore not evidence that the pilot does nothing.
- Per category the validation counts are small (F_low 22, F_edge 16, F_up 12, D_small 55, D_tip 7, D_tall 22, B_back
  24, P_pec 29, H_head 7): only moves of several instances in F_low / B_back / P_pec / D_small can show; D_tip and
  H_head (n 7) cannot show anything.
- The pilot is 1191 training shots (about 2,150 samples per epoch at the one-third share) against 6440 run05 shots;
  it is a pilot, so the test measures whether this small dose moves real photos at all, not what a full render would
  do. A full render could help where this cannot show a gain (dose too small); a clear harm here would be informative.
- Two seeds only: the "both seeds" rule guards against seed noise but not against the shared real-photo sample.

## Amendment 19:30 (before any new arm was trained; the GPU chain opened 19:25 with the W3e pilot-val scoring)
- Found while testing the scorers: experiments/days_all has pointed to the NEW mask root (switch_root/days) since
  2026-11-22 22:16; W3e's class maps were made on the OLD root (/var/tmp/sat3737-whale-days), and the parts_weak shadow's
  links now resolve into the new root, so scoring there mixes the two (my first W3e part-gap numbers above, 0.1616 /
  0.1500 on n 195, came from that mix; finless n 31 instead of 33).
- Fix (no change to any scorer): `model_fine_tuning/pilot22_train/oldroot.py` builds
  /var/tmp/sat3737-parts-cache/pilot22/days_shadow = links to the OLD root + W3e's class maps; new arms are predicted
  there (predict.py / evaluate_real.py unchanged, roots set before main()); compare.py, finless.py, pred_fill.py score
  and part_gaps/report.py run unchanged with their root constants set to it (report's rule cache bypassed, it holds
  live-root rules). Check: W3e's standing measures on this root reproduce two_seeds_w3e.md exactly (fin tip 0.956 /
  0.939, finless by eye 0.182 / 0.242 n 33, classifier 0.139 / 0.152 n 151, three-way 0.886 / 0.861, 0.911 / 0.848,
  0.949 / 0.930, notch 0.812 / 0.818, IoU expanded 0.636 / 0.662).
- Baseline part-gap numbers on the old root (these replace the ones above for the rule): validation-day total cost
  W3e s0 0.1640 (40 of 202 wrong), s1 0.1547 (36 of 202); training days 0.1603 / 0.1791 (n 395). Validation n per
  category: F_low 22, F_edge 16, F_up 12, D_small 55, D_tip 7, D_tall 23, B_back 30 (19 / 17 wrong), P_pec 29 (7 / 4),
  H_head 7. Seed spread of the total 0.009 (about 2-4 instances); the sensitivity reading above is unchanged.
