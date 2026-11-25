# Fused instances: candidates, measures and the choice rule (written before any candidate was built or scored)

Written 2026-11-22 after step 1 (sizing on train + validation days of the shadow root, size_rows.jsonl, joins.jsonl and
two by-eye sheets look1 / look2 of flagged instances). No candidate output exists at this time. Evaluation days have
not been opened. The strict-tier model (safe_model.pkl) and its threshold 0.995 are not refitted.

## Signals (both computable at inference)

- flag_G (geometry, no extra model): the "<stem>~k" instance is a union of >= 2 mask components (the components
  parts.split_instances considers: >= 400 crop px and >= 0.1 of the photo's largest component) and holds a pair of
  components (c_f, c_o) with c_f fluke-like (median body fraction u < 0.15), c_o not (median u >= 0.30), and a gap
  (minimum pixel distance) between them >= 0.20 x the larger of the two components' extents (max of bbox width /
  height, crop px). Threshold set by eye on look2 (train + val days): below 0.2 the joins are mostly one whale (peduncle
  + fluke, head + back); above, fluke + back pairs are mostly two animals (exceptions seen: pectoral slaps).
- flag_S (SAM 3): >= 2 SAM 3 "humpback whale" instances (score >= 0.5, the sam3_compare setting; de-overlapped by score;
  sampled on the CSE crop grid) each holding >= 0.15 of our instance's pixels and >= 400 crop px.

## Candidates (all against the plain shadow chain P)

- A_G  SPLIT by geometry: the fluke-like components behind flag_G leave the instance and form one new instance.
- A_S  SPLIT by SAM 3: where flag_S, the instance's pixels go to the SAM 3 instance they fall in (pixels in none: the
  nearest of those instances); pieces < 400 crop px go back to the nearest piece; no split if one piece remains.
- In both splits the largest piece keeps the instance's key "<stem>~k", new pieces get the next free k of the photo
  (other keys of the photo do not move). The chain after parts_hr is re-run unchanged (cluster_v3 student, safe
  features, link_scores instances, not-whale, safe_tracklets cluster --strict --threshold 0.995 --filter).
- B_G  GUARD flag_G: no strict link whose end instances (A's last, B's first) include a flagged instance.
- B_Gc GUARD flag_G + cut: as B_G, and stage-one tracklets are cut where a tracklet passes from a flagged to an
  unflagged instance or back.
- B_GS GUARD flag_G or flag_S (strict links only).
- C_SG BOTH: A_S split, then B_G guard on its instances.
- C_GS BOTH: A_G split, then guard flag_S on its instances.

## Days (fixed in plan_days.md before this file)

Validation 2025-01-15, 2025-02-27, 2025-03-14; click days 2025-01-02, 2025-03-18; train 2025-02-03, 2025-01-23,
2025-03-26 (8 days). Guard-only arms are also run on all 24 train + val days (cheap) for the wrong-link check.

## Measures (per day and pooled over the 8 days, n given)

wrong automatic links; strict links made; clusters per animal (mean of days); impure photos; catalogue photos without a
kept instance; click days: flukes with both tips in one instance (n 83), main whales with all clicks in one instance
(n 365), other whales own / fused / none (n 17); multi-whale bursts split pure; saved-key attachment (Labeling
Assistant labeling.json keys and pair_marks.json keys on these days: same box / changed box (IoU < 0.5) / gone);
cost per photo. Labeling Assistant replay: only if cheap (not planned).

## Choice rule

1. Wrong automatic links = 0 on every day run (8 days; 24 for guard-only arms). Else out.
2. Impure photos pooled <= P.
3. Not worse than P beyond noise: clusters per animal (mean of 8 days) <= P + 0.15; catalogue photos without a kept
   instance <= P + 2; flukes both tips >= P - 2; main whales all clicks >= P - 3; multi-whale bursts split pure
   >= P - 1; saved keys not on the same box <= P's + 1 % of the saved keys.
4. Among the arms left: fewest fused "other whales"; ties -> fewest strict links lost against P; then lower cost per
   photo (no SAM 3 at inference first); then the smaller code change (guard before split).
5. If no arm passes 1-3, the choice is "none" (P stays) and the evaluation-day read is made with B_G for information
   only (reported as such).
