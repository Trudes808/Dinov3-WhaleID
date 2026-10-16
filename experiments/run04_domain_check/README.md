# Is run04 representative of the real photos? (2026-10-16)

Real reference: the 2024-12-18 day (335 photos with a whale, 2560 px, EXIF focal lengths), masks and body
fractions from the run03 model (no labels exist). Synthetic: a 2 500-shot sample of run04 (ground truth).
Figure: `domain_comparison.png`; examples: `body_fraction_synth_vs_real.jpg` (blue = tail, red = snout),
`real_mask_disagreement.jpg` (photos where the run03 and run04 masks differ most).

| measure | real 2024-12-18 | synthetic run04 |
|---|---|---|
| whale area, % of frame: p10 / median / p90 | 0.4 / 1.7 / 5.1 | 0.2 / 2.5 / 42 |
| whale > 5 % of frame / > 20 % | 11 % / 0 % | 40 % / 20 % |
| whale box width / frame width, median | 0.38 | 0.32 |
| box width > 0.9 of the frame | 1 % | 15 % |
| mask touches the frame edge | 12 % | 26 % |
| body length / frame width, median; longer than the frame | 0.77; 30 % | 0.93; 48 % |
| focal length p10 / median / p90 (mm) | 180 / 250 / 455 | 50 / 229 / 1500 |
| visible whale pixels: flukes+peduncle / peduncle→dorsal / mid body / blowhole region / rostrum | 32 / 35 / 25 / 7 / 1 % | 28 / 10 / 9 / 14 / 39 % |

Findings.
1. The medians of framing agree, but the synthetic has a heavy close-up tail that the real day never shows:
   a fifth of the shots have the whale over 20 % of the frame (real: none), 15 % have the box wider than
   0.9 of the frame (real: 1 %), and the lens goes to 1500 mm (real: 455 mm). About 20-25 % of the render
   budget goes to framings that do not occur. (Config: `camera.extent_target` up to 1.6, distance down to
   40 m, focal up to 1500 mm.)
2. The bigger mismatch is which part of the whale is visible. Real surfacing photos show the back from the
   peduncle to the blowhole (60 % of whale pixels between the peduncle and mid body); the rostrum is 1 %.
   The synthetic surface poses expose the head: in `blow` the head part is 60 % visible and the dorsal back
   4 %, `logging` 62 % vs 13 %, `awash` 28 % vs 1 %, `blow_sliver` 38 % vs 2 %; 39 % of all synthetic
   whale pixels are rostrum. Only `back_arch` (dorsal back 53 %, peduncle 37 %) looks like a real surfacing.
   These keyframes come from the start of the asset's dive animation, where the head is raised.
3. The real-photo deficit of the run04 models sits in exactly that regime: on mid-size whales (3k-20k whale
   pixels, 72 % of the day) the outline IoU is 0.17-0.26 vs run03's 0.30-0.33, while large whales are equal.
4. Both models have false positives on this day (a crew member and a coastline for run03, a fishing rod and
   a wake for run04); the label-free metric inherits them. A small set of real masks (a few dozen photos)
   would make the real-photo numbers trustworthy.
