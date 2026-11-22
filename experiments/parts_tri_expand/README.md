# Expanded triangle: the dorsal part (fin + hump + a strip of back) from the three fin clicks and the whale mask

User's request (2026-11-20): keep the triangle's coverage and direction, but fill out to the full dorsal fin. User's
choice (2026-11-21): **variant A** ("I prefer variant A which has the extra back pixels as well") is the definition of
the dorsal part from now on. Variant B is kept only as a side column where it was already computed.

## The rule (A), in words

Inputs: the three fin points (tip T, front base F, back base B) and a whale mask. In the fin's own frame, "along" runs
on the base line from B (0) to F (1), and "up" is the height above the base line. h is the tip's height above the
base line, and h_eff = max(h, base / 3), which keeps a usable zone when the three points lie close to a line.

    zone     = along in [-0.05, 1.05]  and  distance to the triangle <= 0.3 h_eff   (and up >= -0.45 h_eff, not binding)
    expanded = (triangle AND mask)  OR  the pieces of (mask AND zone) that touch (triangle AND mask)

So the dorsal part is every whale-mask pixel within 0.3 h_eff of the clicked triangle, between the two bases plus 5 %
of the base, connected to the triangle. It takes in:
- the curved leading edge,
- the hooked or rounded tip,
- the trailing edge and the hump's arc,
- a strip of back about 0.3 fin heights deep under the base line.

The mask stops it in water and sky. The distance bound stops it running along another whale, the own fluke or the
flank.

| parameter | A (the definition) | B (side column) |
|---|---|---|
| m_along (beyond the bases, fraction of base) | 0.05 | 0.0 |
| m_below (under the base line, fraction of h_eff) | 0.45 (not binding: d_near bounds the depth) | 0.0 |
| d_near (outside the straight sides, fraction of h_eff) | 0.3 | 0.3 |
| fixed on | r06r_check20, 127 fins with all 3 fin points visible, exact whale mask, mean IoU with the run06 fin + hump label | same fins, fin-only (blade) label |

- Code: `model_fine_tuning/parts_weak/tri_expand.py`; parameters are in `rule.json`.
- Grid: `grid_r06r_check20.json`. It covers m_along 0-0.3, m_below 0-0.45 (extended to 1.0), d_near 0.1-9.
- The objective is flat within 0.005 for m_along 0.05-0.3 and m_below >= 0.3. d_near peaks at 0.3: 0.2 gives -0.035, 0.5 gives -0.018, unbounded (9) gives -0.039.
- B was added after the first look at the real sheet, before any check21 or real score of B was read.
- Masks: S6h is the candidate mask, `experiments/seg_splash_s6/checkpoints/last.pt` with hysteresis 0.6 / 0.1, at 1280 px. "Live" is seg_r04_ft4 at P >= 0.5.

## 1. Renders (exact truth): report set r06r_check21 (rule fixed on r06r_check20)

147 fins with the tip visible and the bases visible or under water; the clicked-style triangle is taken from the true 3D fin points. IoU is a per-fin mean with a 95 % bootstrap interval; R / P are mean recall and precision.

Against the run06 fin + hump label (the dorsal part as the label pass defines it):

| group | n | triangle | A, exact mask | A, S6h on the render | A, live | B, exact mask |
|---|---|---|---|---|---|---|
| all | 147 | 0.520 [0.50, 0.54], R .57 P .89 | **0.764 [0.74, 0.79]**, R .85 P .91 | 0.675 [0.64, 0.71] | 0.683 [0.65, 0.71] | 0.618 |
| side-on (fin normal < 30 deg from the line of sight) | 76 | 0.536 | 0.784 | 0.695 | 0.706 | 0.637 |
| oblique (30-60) | 40 | 0.547 | 0.798 | 0.713 | 0.729 | 0.644 |
| end-on (> 60) | 31 | 0.448 | 0.670 [0.60, 0.74] | 0.578 | 0.566 | 0.539 |
| one whale | 56 | 0.551 | 0.816 | 0.749 | 0.763 | 0.651 |
| several whales | 91 | 0.501 | 0.731 | 0.630 | 0.633 | 0.598 |
| blade 50-90 % visible (rest under water) | 5 | 0.256 | 0.857 | 0.326 | 0.333 | |
| points near a line (base / h > 6) | 1 | 0.122 | 0.456 | 0.000 | 0.150 | |

Against the fin-only (blade) label:

| | triangle | A, exact mask | A, S6h | B, exact mask |
|---|---|---|---|---|
| IoU | 0.342 | 0.308 | 0.262 | 0.435 |
| recall | 0.83 | **0.99** | 0.94 | 0.98 |

- The fit set check20 (141 fins) reads the same: triangle 0.503, A exact 0.789, S6h 0.700, live 0.696.
- All bins, both sets: `tables_renders.md`.
- Pooled over both sets there are few partly submerged fins (11) and few flat triangles (3), so those rows are not readings.

What the numbers say:
- A takes in the whole blade. Blade recall goes from 0.83 for the triangle to 0.99 with the exact mask.
- A lifts IoU with the fin + hump label from 0.52 to 0.76.
- With a predicted whale mask, A reaches 0.68 (S6h or live). That is still 0.16 above the bare triangle.
- On renders the mask costs about 0.09 IoU; the rule's own error accounts for the rest of the gap to 1.
- Fin-only IoU stays low by design. The blade is 27 % of the run06 dorsal label (median).

Where A fails on renders (`triexp_failures.jpg`, top half):
- Far or low fins that S6h does not find at all: IoU 0.
- End-on fins. The triangle is a sliver, and the run06 label there is a broad blob.
- A fin in front of a second whale: the fill runs into the other whale's back, within d_near.
- Splash on the fin.

## 2. Real photos: the researcher's clicked fins (2025-01-02 and 2025-03-18; 197 fins, read only)

Measured (`real_stats.json`), medians with quartiles:

| | A, S6h | A, live | B, S6h |
|---|---|---|---|
| expanded area over the triangle | 1.90 (1.77-2.05) | 1.77 (1.67-1.94) | 1.20 (1.11-1.33) |
| area added over the triangle | 0.90 | 0.80 | 0.23 |
| triangle share outside the whale mask | 0.012 (0-0.03) | 0.034 (0.01-0.06) | |
| fins with over 25 % of the triangle off the mask | 1 (whale under water, S6h empty) | 1 | |

By-eye tally, judged by me on 60 fins at random (seed 1). Sheets are `triexp_tally_{1,2,3}.jpg`.
- Question: does A hold the full fin and the hump top?
- "Too much" = a block of back clearly thicker than a strip under the base line, or skin well beyond the fin.

| | yes | too much | too little | wrong because the whale mask is wrong |
|---|---|---|---|---|
| A with S6h | 53 | 5 (tall fins seen side-on, one blurred and one odd view) | 0 | 2 (splash in the mask) |
| A with the live mask | 53 | 5 | 0 | 2 |

- The two masks gave the same call on all 60 at this tile size.
- Every expanded mask carries the strip of back under the base line. In "yes" it is a strip; in "too much" a block.
- The largest growth on real photos (A up to 3x the triangle) comes from low, long fins whose three points are nearly on a line. There h_eff = base / 3 sets the strip's depth.

## 3. In the model

Two routes, W3c's measures with the same scripts and days, two seeds each. The rule for "separable" is W3c's own; see `two_seeds.py`.
- **(a) at prediction time.** W3c's predicted dorsal region gives the smallest triangle around it. Its base is the side with the most whale mask just outside it, and the tip is the opposite corner. The result is then filled with S6h by rule A. Code: `pred_fill.py shadow`.
- **(b) as the training label (W3d).** `model_fine_tuning/parts_weak/build_tri3.py`: the clicked triangle of 2025-01-02 becomes the expanded triangle (A, S6h). W3c's "certainly not dorsal" zones are kept except within 4 px of the expanded part. The rest is W3c's recipe exactly (`run_w3d.sh`).

Full table with verdicts: `two_seeds.md`. Paired per-fin differences with 95 % bootstrap: `paired_new_measures.txt`.

**Shape on the held-out clicks of 2025-03-18** (99 fins; no model trained on these clicks):

| | W3c s0 / s1 | (a) W3c filled s0 / s1 | (b) W3d s0 / s1 | the truth (A on S6h) |
|---|---|---|---|---|
| IoU with the expanded-triangle truth, mean [95 %] | 0.593 [.57, .62] / 0.589 | 0.639 [.61, .67] / 0.641 | **0.668 [.65, .69] / 0.652** | 1 |
| lowest painted point, fin heights (median) | -0.04 / -0.02 | -0.48 / -0.47 | -0.31 / -0.33 | -0.26 |
| region over triangle area (median) | 1.47 / 1.51 | 2.88 / 2.87 | 2.33 / 2.53 | 1.93 |
| clicked triangle covered | 0.97 / 0.96 | 1.00 / 1.00 | 1.00 / 1.00 | 0.995 |

Paired IoU differences on the same 99 fins, against the expanded-triangle truth:

| | s0 | s1 |
|---|---|---|
| W3d - W3c | +0.075 [+0.046, +0.102] (67 up / 32 down) | +0.063 [+0.033, +0.090] |
| filled - W3c | +0.046 [-0.003, +0.091] | +0.052 [+0.004, +0.096] |
| W3d - filled | +0.029 [+0.004, +0.057] | +0.012 [-0.012, +0.036] |

Training-day clicks 2025-01-02 (W3c and W3d trained on these, reference only), IoU with the truth: W3c 0.624 / 0.634, filled 0.544 / 0.555, W3d 0.653 / 0.634.

Standing measures (validation days, W3c's scripts and days, unchanged):
- Nothing else separates from W3c in either arm, on two seeds: fin region found, fin tip inside, finless backs called dorsal by eye and by classifier, three-way call by eye (crops, whole photo) and by classifier, fluke quarter, fluke notch, tail stock, synthetic pectoral and old-label dorsal IoU (W3d only).
- Values to watch (not separable):
  - finless backs called dorsal by eye: W3d 6 / 33 on both seeds, against W3c 2 and 5;
  - by-eye three-way call on crops: W3d .899 / .899, against W3c .937 / .911.
- Dorsal share of the whale in dorsal views (val, median): W3c .07, W3d .11, filled .16. This follows from the larger part and is separable for both arms.

Renders r06r_check21 (131 fins with all 3 points visible), IoU of each arm's region near the fin, full-frame class map at 1280:

| | W3c s0 / s1 | filled | W3d | W3d filled |
|---|---|---|---|---|
| fin + hump label | 0.215 / 0.205 | 0.347 / 0.322 | 0.245 / 0.224 | 0.369 / 0.347 |
| fin-only label | 0.397 / 0.385 | 0.318 / 0.322 | 0.388 / 0.378 | 0.293 / 0.302 |

- On renders every part model is poor: fins there are about 17 px high in the 1280 frame. The models were never trained on run06.
- The fill (a) helps there (+0.12 to +0.13 paired, separable) because it uses the render's own mask.
- W3d over W3c is +0.03 / +0.02 paired, inside one seed distance by the rule.

Reading:
- (b) W3d gives the A-shaped part from the photo alone. It is closest to the truth on depth, area and IoU, and does not cost any standing measure.
- (a) the fill gets about two thirds of the IoU gain with no training. But it overshoots on real photos, at 2.9x the triangle and 0.48 fin heights down, because it expands a region that is already 1.5x the triangle.
- (b) over (a) on the held-out clicks: +0.029 [+0.004, +0.057] on seed 0, +0.012 [-0.012, +0.036] on seed 1. Not separable by the two-seed rule. 99 fins on one day is few; this is a lean, not a result.
- (b) beats (a) separably on depth and area: it is closer to the truth's -0.26 and 1.9x.

## 4. Downstream, if A becomes the dorsal part

- **Label builder.**
  - `build_tri3.py` is the A label for the clicked training fins: the expanded triangle on S6h, W3c's "not dorsal" zones minus the expanded part.
  - The W3c not-dorsal margin (0.2 fin heights under the base line) is now inside the A part, so build_tri3 cuts it out locally. A clean version would move the margin to about 0.35 h_eff in `build_tri2` / `build_tri3`.
  - The label uses S6h. If the mask switch does not happen, the live mask gives an A part about 7 % smaller (1.77x against 1.90x the triangle on real photos).
- **Render label pass (run06).** It is NOT the same as A, and it is not the triangle either.
  - With the true fin points and the exact mask, A matches the run06 fin + hump label at IoU 0.76-0.79 (recall .85-.87, precision .91).
  - The run06 label reaches further down the flank: 38 % of its pixels lie under the base line, the 5th percentile at -0.33 h_eff, and in some views it is a round blob down to -0.7.
  - The bare triangle covers only 57 % of it.
  - Two ways to make real and render agree:
    1. Re-make the render dorsal label by rule A from the render's own fin keypoints and instance mask. This is offline, about seconds per shot, with no new render; pixels of the old label outside A become body.
    2. Leave the render label as it is and accept a definition gap of about 0.2 IoU between the two sources.
  - I recommend (1) before any run06 part model is trained, so the model sees one definition. Not done; Whale-Image-Data-Factory was read-only for this job.
- **Cut-outs.** A is larger than W3c's region (1.9x against 1.5x the triangle) and its centroid sits lower, toward the back line.
  - The 2026-11-18 test (`cutouts_w3c`) found that tighter fin cut-outs lose first matches. The loss follows zoom; the centre alone did not lose.
  - So an A-based cut-out sized on A's own extent would be looser than "W3c tight": in the direction that lost less. A-based centring would shift the fin square down by roughly 0.1-0.15 fin heights against W3c's centre.
  - The safe use is the one already tested: centre from the dorsal part, zoom from today's parts_up. Not re-run here, as instructed.
- **Fin drawing (Labeling Assistant, `cluster/fin_sketch.py`).** It does not use the part model.
  - It takes the point model's three fin points and the whale mask, then draws the mask's top and trailing edge in the keypoint frame. That is already the "outline from the mask" half of A.
  - A would only give it a defined lower boundary for where marks are searched: fin + hump + the 0.3 h_eff strip, instead of the whole tile.
  - That is a small change and has no effect on ranking or links (the sketch feeds nothing).
- **Nothing live uses any of this.**

## Commands

```
# renders: windows, masks, grid, scores, tables, sheets
.venv/bin/python experiments/parts_tri_expand/render_fins.py collect
flock /var/tmp/sat3737-parts-cache/gpu_train.lock env PYTHONPATH=. .venv/bin/python experiments/parts_tri_expand/render_masks.py
.venv/bin/python experiments/parts_tri_expand/render_fins.py grid --fit r06r_check20
.venv/bin/python experiments/parts_tri_expand/render_fins.py score && .venv/bin/python experiments/parts_tri_expand/tables.py
.venv/bin/python experiments/parts_tri_expand/render_sheets.py
# real clicked fins
.venv/bin/python experiments/parts_tri_expand/real_fins.py stats|sheet|tally-sheet|failures
# (b) W3d label and training (two seeds, W3c recipe), then (a) fill at prediction time
PYTHONPATH=. .venv/bin/python -m model_fine_tuning.parts_weak.build_tri3
setsid nohup bash experiments/parts_tri_expand/run_w3d.sh > /var/tmp/sat3737-parts-cache/parts_weak/run_w3d.log 2>&1 &
flock /var/tmp/sat3737-parts-cache/gpu_train.lock env PYTHONPATH=. .venv/bin/python experiments/parts_tri_expand/s6h_days.py
PYTHONPATH=. .venv/bin/python experiments/parts_tri_expand/pred_fill.py shadow --model parts_seg_r05_w3c   (and _w3cs1)
# scoring (W3c's scripts, unchanged) and the two new measures
M="parts_seg_r05_w3c parts_seg_r05_w3cs1 parts_seg_r05_w3c_fill parts_seg_r05_w3cs1_fill parts_seg_r05_w3d parts_seg_r05_w3ds1"
PYTHONPATH=. .venv/bin/python experiments/parts_weak/compare.py --arms $M --out experiments/parts_tri_expand/compare_triexp.json
PYTHONPATH=. .venv/bin/python experiments/parts_weak/fin_definition.py --models $M --out experiments/parts_tri_expand/fin_definition_triexp.json --sheet-name triexp_fin_definition.jpg
PYTHONPATH=. .venv/bin/python experiments/parts_weak/finless.py --models $M --out experiments/parts_tri_expand/finless_triexp.json
PYTHONPATH=. .venv/bin/python experiments/parts_weak/shares.py --models $M --out experiments/parts_tri_expand/shares_triexp.json
flock ... pred_fill.py render-pred --model NAME --checkpoint CKPT --sets r06r_check20 r06r_check21   (per model)
PYTHONPATH=. .venv/bin/python experiments/parts_tri_expand/pred_fill.py score --models parts_seg_r05_w3c parts_seg_r05_w3cs1 parts_seg_r05_w3d parts_seg_r05_w3ds1
PYTHONPATH=. .venv/bin/python experiments/parts_tri_expand/two_seeds.py
PYTHONPATH=. .venv/bin/python experiments/parts_tri_expand/pred_fill.py sheet
```

## Where things are
- **Repo, code and tables only:** this folder, plus `model_fine_tuning/parts_weak/{tri_expand,build_tri3}.py`.
- **Checkpoints:** `/var/tmp/sat3737-parts-cache/parts_weak/runs/parts_seg_r05_w3d{,s1}/checkpoints/best.pt`, beside W3c's.
- **Label maps:** `/var/tmp/sat3737-parts-cache/parts_weak/maps_tri3/`.
- **Class maps:** `/var/tmp/sat3737-parts-cache/parts_weak/days_shadow/<day>/<arm>/`.
- **Render windows, render masks and part maps, S6h masks for the fill:** `/home/sat3737/NOAA_whales/moonshot_cache/parts_tri_expand/`.
- **Sheets:** the scratchpad `triexp/sheets/` only.
