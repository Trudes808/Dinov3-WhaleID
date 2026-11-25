# Scene model v3 and the marker reader: rules

Part S written 2026-11-22 20:45 -0600, before any v3 label, training or development score existed, and before v2 was run on
the development sets. sha256 per part in `moonshot_cache/sam3_labeller/v3/rules.sha256`.

## S0. Data roles and what traces back to the test day
- Test day 2026-04-07: nothing fitted, chosen or looked at; one scored run at the end (S7), the third scored reading of
  this day for this line of work (v1 / v2 run, the post-hoc S6h gate, now v3). The two faults fixed here (whale on the boat,
  weak tape / deck) were found on it; so were the marker definition and the colour rule (used unchanged, not re-tuned).
- Eval days 2024-12-18, 2024-12-26, 2025-02-19, 2025-03-21: never read. Val days 2025-01-15, 2025-02-27, 2025-03-14:
  checks only, never trained on.
- Mask2Former ADE20K as a boat labeller: known from the test-day comparison (boat + vessel groups IoU 0.85 on our boat).
  Used here as a labeller on training days only; its numbers on the test day are not re-tuned.

## S1. Development measure "whale on boat" (no test day)
1. Real: `v3/dev_real_boat.json`, 31 photos (2025-03-14 val 7, 2025-01-17 train 14, 2025-03-18 train 10) picked by eye
   on plain thumbnails from the SAM 3 pool (rail + tape + deck + any-boat >= 0.3 % of the frame): photos that show our
   boat's rail tubes / stanchions or bow deck (gear poles, crossbow, person only, life ring left out). Fixed 20:38 before
   any score. These 31 photos are removed from every v3 training set (pseudo and real_s6mc).
   Per photo, at 1280: boat region = (ADE boat group | SAM 3 rail, tape, "boat deck", "boat") minus S6h dilated 8 px;
   measures: whale px on the boat region, whale px outside S6h dilated 8 px; photos with >= 50 px of each.
2. Synthetic: 20 % of the boat renders held out by crc32("run/id") % 5 == 0 (no look at content); exact labels mapped
   to ours (rail 3, tape 4, deck + cleat 5, land 6, pontoon = not whale, whale through water = ignore). Measures: per-class
   IoU; whale px on rail / tape / deck / pontoon; held-out marker shots with >= 50 whale px.

## S2. Label changes (v3 label maps from v2's, `v3_tools.py labels`)
a. NOT WHALE (code 254): pixels that v2 left ignore, where a boat labeller is sure (ADE20K boat group: boat, ship,
   railing, fence, bannister, floor, pole, step, stairs, stairway, column, base; SAM 3 "boat" @ 0.5; SAM 3 rail sure),
   outside the whale's reach (S6 P >= 0.1, S6h dilated 8 px, SAM 3 "humpback whale" @ 0.5). Loss there: -log(1 - P(whale))
   (the pixel may be any class but whale), counted with the cross-entropy pixels; in the Dice term only the whale channel
   sees them (as background). Why: v2's labels left the boat ignore, so nothing taught "deck is not whale"; "not whale"
   is all the labellers can vouch for on pontoons, hulls, other vessels and gear.
b. Deck (5) on real photos: ADE boat / ship connected components touching the frame edge (our own boat always enters
   from the edge; other vessels float inside the frame), minus SAM 3 rail / tape / person / land / gear (sure or unsure),
   minus the whale's reach, eroded 2 px with a 2-px ignore band to water. Pontoon / hull pixels inside it become deck
   (no labeller separates them); stated as a label error.
c. Kept from v2: thin ignore bands (1 px rail / tape, 3 px others), class-balanced sampling (now over person, rail, tape,
   deck, land, not-whale), B5's score >= 0.5 rule inside S6h.
d. real_s6mc: S2a and S2b applied the same way (its SAM 3 regions are ignore there).

## S3. Synthetic boat renders
Sources: `r06r_boat20b_look` (28), `r06r_boat20_look` (25), `r06r_boat22_markers` (new, about 300 marker shots of
round 20h's boat content: 150 rail + tape in the four colours, 90 deck / bitt, 60 land; config copy of the live config +
keys_boat20b.json + force_plan; live config not written). Labels mapped as S1.2. Known look faults: sea behind the rail
near black, deck clean white. 80 % train, 20 % held out (S1.2). If the new run is not done when training starts, the
variants are trained on what exists and that is said.

## S4. Training (recipe of v2: seg_multi.train, same backbone, lr, epochs, init; seed 0)
- v3a: v3 labels (S2), no renders: real_s6mc_v3 p 0.2, pseudo_v3 p 0.3 (balanced), run05 renders otherwise.
- v3b: as v3a + boat renders (train part) with p 0.1 taken from run05's share (balanced over rail, tape, deck, land,
  not whale with p 0.6, else anywhere).
A second seed of the chosen variant if time allows (said if not).

## S5. Choice of the variant (on S1 and the val days only)
1. Eligible: val whale against S6h (883 catalogue photos, score_val.py): pooled IoU >= 0.945 (v2 0.955 - 0.01) and at most
   2 photos without a whale mask where S6h has one.
2. Primary: fewer real dev photos (S1.1) with >= 50 whale px on the boat region. Within 2 photos: fewer held-out render
   shots with >= 50 whale px on the boat; then the higher mean IoU against the labeller's val maps (pseudo_val) over
   person, rail, tape, deck, land; within 0.01: v3a (no renders, fewer assumptions).
3. v2 is the reference; if neither variant beats v2 on S5.2's primary, that is the result and v3 = the better variant anyway.

## S6. Marker reader (`model_fine_tuning/seg_multi/markers.py`), frozen before S7
Per photo of a day folder (time order from timestamps.json), from the scene map at 1280 (A7 rule):
- whale present: by the gate chosen in S6.1; 50 px threshold (as C3).
- classes present with frame shares; marker kind by C3 of rules_labeller.md unchanged (rail + tape >= 4 %, widest tube
  >= 5.5 % of the width, largest tape component >= 0.12 %; else land >= 25 %; else deck >= 10 % with person <= 10 %).
- tape colour: C3's HSV rule unchanged (red / yellow / green / blue / other / none) inside the largest tape component.
- refers_to: walking back from a marker in time order, the photos with a whale (gate) until the previous marker or a gap
  of more than 60 s between consecutive photos of the day (the gap ends the series; photos before it are not included).
S6.1 Gate, chosen on non-test data: S6h alone, the scene model's own whale alone, or both (whale present only when both
paint >= 50 px). Score = false marker calls on the culled days (3 val + 5 train days: 2025-01-17, 2025-02-12,
2025-03-18, 2025-01-02, 2025-03-06; every call there is false) + missed held-out render markers (S1.2 marker shots with no
marker call or the wrong kind). Lowest wins; ties: S6h alone (the whale everyone uses today).
Every marker call on the culled days is looked at by eye on a sheet and reported.

## S7. The test day, once
Frozen v3 (chosen variant; and the other variant reported beside it) and the marker reader: pixel scores
(`pixtruth_score.py`, same root), whale on the 44 no-whale photos, markers per kind against markers.tsv, false markers,
tape colour, refers_to against the truth's (exact set, and overlap). Beside v2, SAM 3 and Mask2Former rows of the earlier
tables. Nothing is changed after it.

## S6 addendum (2026-11-22 21:05 -0600, before any v3 training or marker output)
- refers_to also walks back past the marker's own run of consecutive markers (each step <= 60 s) before collecting whale
  photos, as the truth table's definition states (pixtruth_markers.py docstring; a definition, not data). Two rail
  shots in a row both refer to the series before the first.
- Each record carries `meaning` (user / researcher, 2026-11-22, final): red = female in a group or the mother with a calf;
  green = male in a competitive group or the escort; blue = calf; yellow = challenger or joiner to the group; bollard =
  the animal of the series just before is a passer-by / opportunistic animal, not part of that group (catalogue
  "<day>_Onn"); land = the boat moves to a new group. markers.tsv's meaning column was rewritten with the same text
  by pixtruth_markers.py (8 cells: 1 yellow, 7 bollard; nothing else changed).
- A one-command runner for a new raw day (decode, S6h, scene model, records, sheet) is added; its first run on
  2026-04-07 is the S7 run itself.

# Part R (record, 2026-11-23 03:40 -0600; test day still untouched by v3)
- Renders: r06r_boat22_markers built (300 marker shots: rail 150 = red 38 / green 38 / blue 38 / yellow 36, deck 90,
  land 60; 2 h 58 min of render lock). Render sets (S1.2 hash split): train 291, held out 62 (58 marker shots: rail 28,
  deck 15, land 15; 4 whale shots). An extra early variant v3b0 (= v3b with only the 43 look renders in train) was
  trained while the new run rendered; reported, not a candidate.
- S5 choice: both v3a and v3b (seed 0) eligible (val pooled IoU with S6h 0.949 / 0.951, no photo emptied). Primary
  (real dev photos with >= 50 whale px on the boat): v3a 12, v3b 6 -> v3b. v2 also has 6: on the real dev photos neither
  variant beats v2 (S5.3); the gain is on the held-out renders (v2 46 of 58 marker shots with whale, v3b 0).
- S6.1 gate (v3b seed 0): score = culled-day calls + held-out render markers missed / wrong: s6h 1 + 5 = 6, own 1 + 1 = 2,
  both 1 + 1 = 2. The written tie rule only names S6h; between own and both the script's order (written before any
  output: s6h, own, both) gives OWN. Gate = own (the scene model's whale). S6h misses 4 render bollards because it
  paints the white render deck (render look), which the real culled photos cannot confirm or deny.
- Frozen for S7: seg_scene_v3b_s0 (seed 0 is the primary; seed 1 trained after, reported beside it), gate own,
  model_fine_tuning/seg_multi/markers.py + run_new_day.sh (with the removal step), hashes in v3/FREEZE.json.
