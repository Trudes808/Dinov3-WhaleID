"""Training test of the run06 pilot r06_pilot22 (2026-11-23): does adding the pilot's renders to W3e's synthetic data
improve the part model (fluke / dorsal / other) on real photos?

  prep.py        the pilot at run05's pixel scale (2560 -> 1280 frames, images INTER_AREA, labels nearest) as a label
                 cache in W3e's palette (parts_triA = rule-A dorsal; through-water zone and undefined dorsal = ignore);
                 id lists (train: pilot train split minus individual i12, minus gate-rejected, minus run05's defect
                 flags and min-visible filter; val: pilot val minus i12, minus rejected)
  train.py       W3e's command (parts_weak.train, parts_seg.train unchanged) with the TRAINING render stream replaced by
                 a mixture: --pilot-share of the render samples from the pilot, the rest from run05 (P_add), or the
                 pilot only (--pilot-share 1, P_only); same steps, schedule, validation and checkpoint choice
  score_pilot.py pilot val IoU per class (full frame at 1280, against the prep label cache)

Nothing here changes existing code; the factory run folder is read only. Design and decision rule: the scratchpad
plan.md of the pilot22d helper (copied to experiments/pilot22_train/plan.md).
"""
