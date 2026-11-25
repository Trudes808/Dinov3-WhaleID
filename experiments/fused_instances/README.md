# Two whales in one instance: size, candidates, choice, one evaluation-day read (2026-11-22)

Text only here (this repo is public). Rows, flags, SAM 3 outputs, arm day folders: untracked
`moonshot_cache/fused_instances/` (FI). Sheets: session scratch `fused/` and `FI/sheets/` (they may show crew).
Shadow root (S6h + rule F, read-only): `moonshot_cache/switch_root/days` (SR). Nothing live was changed.

Rules: `rules_fused.md` (candidates, measures, choice rule), written 2026-11-22 21:03 after the sizing step and before
any candidate was built or scored; sha256 in `FI/rules_fused.sha256`, frozen copy `FI/rules_fused_frozen.md`.
Day sets: `plan_days.md` (fixed before the rules). Evaluation days were opened once, after the freeze (step 4).

## Commands

    FI=/home/sat3737/NOAA_whales/moonshot_cache/fused_instances
    experiments/fused_instances/run_sam3_days.sh $FI/sam3 DAY ...            # SAM 3 "humpback whale", one day per GPU-lock hold
    PYTHONPATH=. .venv/bin/python experiments/fused_instances/size.py --out $FI/size_rows.jsonl          # step 1 rows (24 train + val days)
    PYTHONPATH=. .venv/bin/python experiments/fused_instances/tiles.py --keys LIST.tsv --out SHEET.jpg   # by-eye sheets (scratch only)
    PYTHONPATH=. .venv/bin/python experiments/fused_instances/flags.py --root DAYS --day DAY --out F.json # flag_G / flag_S per instance
    PYTHONPATH=. .venv/bin/python experiments/fused_instances/split_arm.py --mode G|S --dst $FI/days/A_G|A_S --day DAY
    experiments/fused_instances/arm_chain.sh $FI/days/A_G -- DAY ...        # features, GPU steps on the split photos only, merge, not-whale
    PYTHONPATH=. .venv/bin/python experiments/fused_instances/score.py --arms P,A_G,... --days ... --out $FI/score_*.json
    experiments/fused_instances/arms_day.sh DAY $FI/days/A_G $FI/days/A_S   # same, both arms, mini steps on the CPU (cpu_steps.py)
    PYTHONPATH=. .venv/bin/python experiments/fused_instances/examples.py --keys LIST.tsv --mode G --flags P --guard --out SHEET.jpg
    FUSED_EVAL_OK=1 PYTHONPATH=. .venv/bin/python experiments/fused_instances/score.py --arms L,P,B_G \
        --days 2024-12-18,2024-12-26,2025-02-19,2025-03-21 --no-clicks --no-attach --out $FI/score_eval_BG.json   # the one eval read

Code: `fcommon.py` (paths, eval-day guard, SAM 3 readers, geometry), `flags.py` (the two signals and the two splits),
`split_arm.py` (re-cuts only the split photos with the unchanged `parts.extract_photo`; its instance list is replaced),
`gpu_mini.py` (the two GPU commands on the split photos only, merged with the shadow's rows), `score.py` (strict tier
exactly as `switch_eval.py`, guards by wrapping `safe_tracklets.link_tracklets` / `make_tracklets`). No pipeline file
was edited; adopting a choice is a flag or patch (section 5).

## Outcome

- Chosen by the rule: **B_G**, a guard. No strict automatic link whose end instance is flagged by flag_G (the instance
  joins a fluke-like mask component and a back / fin component with a gap of >= 0.2 of their size). Instances, keys
  and every file before the clustering stay as they are.
- Train + val: 0 wrong links in every arm (8 days; B_G and B_Gc also on all 24 train + val days). B_G costs 3 of 76
  strict links on the 8 days (10 of 187 on 24 days), all of them correct links; clusters per animal 9.008 -> 9.025.
- It does not reduce fused instances (it is a guard): other whales fused 5 of 16 before and after.
- The splits (A_G, A_S and both C arms) reduce fused other whales (5 -> 4 / 3) and A_S lowers impure photos 14 -> 12,
  but each loses 3 flukes with both tips in one instance (50 -> 47, allowed 48) and 4 main whales with all clicks in
  one instance (315 -> 311, allowed 312). They fail rule 3, narrowly.
- Evaluation days, read once after the freeze: the 2025-02-19 wrong link (154 -> 155) no longer forms; wrong links
  live 0 | shadow 1 | B_G 0. B_G also drops 3 correct links there.

## 1. Size of the problem (train + val days of the shadow root; measured unless "by eye")

24 days, 9 415 instances. Truth sources and signals (n):

| source / signal | n | notes |
|---|---|---|
| click days, "other whale" clicks | 17 (snapshot after go-live: 16) | own 4 / fused 5 / none 8 (7). Fused 5: flag_G catches 1, flag_S 2 (1 shared), none of the signals 3 (the other whale lies mostly outside our instance or SAM 3 also sees one whale) |
| impure safe clusters | 13 clusters, 14 minority photos | 3 contain a flagged instance (1 by G, 2 by S); 10 are stage-one tracklets passing from one whale to another across single-instance photos: no fused instance involved |
| instances made of >= 2 mask components (the u-join of split_instances) | 135 | by eye (look2, 45 sampled): joins with a small gap are one whale (peduncle + fluke, head + back); far fluke + back joins are mostly two animals |
| flag_G | 57 (8 days: 27) | by eye 22 two animals / 10 one / 25 unclear |
| instances with both fin and fluke charts | 415 | 21 flagged by G (by eye 11 two / 2 one / 8 unclear) |
| flag_S (SAM 3 two whales >= 15 % each) | 28 of 1 344 instances where SAM 3 had run at sizing time; 75 of 4 107 on the 8 days | by eye (the 28) 11 two / 9 one (SAM 3 splits one long back or a fluke's two blades) / 8 unclear |
| G and S together | 8 of the sized ones; 22 on the 8 days | |

By eye on all 80 flagged or click-fused instances (sheets label_00..03): 34 two animals, 17 one, 29 unclear (far,
small, splash). Precision of a flag as "two animals" is therefore 0.4-0.7 depending on how the unclear ones fall.
Sheet: `fused/sheets/fused_examples.jpg` (20 two-animal instances: 14 flagged by G, 6 that G misses).

## 3. Candidates against the plain shadow chain P (8 days; label files: snapshot taken after the switch go-live)

| arm | wrong links | strict links | clusters / animal (mean of 8 days) | impure photos | cat. photos without kept inst. | multi-whale bursts split pure | flukes both tips (n 83) | main whales all clicks (n 363) | other whales own / fused / none (n 16) | saved keys same / changed box (n 1 407) | flagged |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P | 0 | 76 | 9.008 | 14 | 33 | 38 of 51 | 50 | 315 | 4 / 5 / 7 | 1 407 / 0 | 0 |
| A_G | 0 | 73 | 9.021 | 14 | 33 | 38 | **47** | **311** | 5 / 4 / 7 | 1 397 / 10 | - |
| A_S | 0 | 67 | 9.101 | 12 | 33 | 39 | **47** | **311** | 6 / 3 / 7 | 1 394 / 13 | - |
| **B_G** | 0 | 73 | 9.025 | 14 | 33 | 38 | 50 | 315 | 4 / 5 / 7 | 1 407 / 0 | 27 |
| B_Gc | 0 | 73 | 9.084 | 14 | 33 | 38 | 50 | 315 | 4 / 5 / 7 | 1 407 / 0 | 27 |
| B_GS | 0 | 67 | 9.065 | 14 | 33 | 38 | 50 | 315 | 4 / 5 / 7 | 1 407 / 0 | 80 |
| C_SG | 0 | 67 | 9.101 | 12 | 33 | 39 | **47** | **311** | 6 / 3 / 7 | 1 394 / 13 | 7 |
| C_GS | 0 | 67 | 9.060 | 14 | 33 | 38 | **47** | **311** | 5 / 4 / 7 | 1 397 / 10 | 54 |

Bold = outside the rule's noise band. Saved keys gone: 0 in every arm. Per day: `FI/score_8days.json`. All 24 days,
guards only (`FI/score_PB_24days.json`): wrong links P 0 / B_G 0 / B_Gc 0; strict links 187 / 177 / 178; clusters per
animal 9.363 / 9.385 / 9.433; impure photos 14 / 14 / 14. B_G drops a link on 8 of 24 days (one each, two on
2025-03-20 and 2025-03-26).

Choice (rules_fused.md): splits out on rule 3; B_G, B_Gc, B_GS tie on fused (5); fewest strict links lost: B_G and
B_Gc (3) before B_GS (9); same cost; B_G is the smaller change. Frozen 2026-11-23 01:06 (`FI/FREEZE.txt`, code hashes).
Only the sheet tools (examples.py, tiles.py captions / tile size) were edited after the freeze.

Cost per photo (measured): flag_G 16 s wall for 6 073 photos on 8 CPU workers (with SAM sampling; G alone is less),
no GPU. SAM 3 ("humpback whale" only) ~0.2-0.8 s per photo on the shared GPU (202 photos in 63 s, 612 in 484 s with
the GPU shared). A split re-cuts only the split photos (27 / 75 photos on the 8 days: seconds) plus the student
embedding / instance crops for them (CPU in this study, ~10-75 s per day; CPU vs GPU rows differ by <= 0.003 in the
unit embeddings and <= 0.013 in the fp16 crop features, about the GPU's own batch noise of 0.0015-0.004).

Not run: the Labeling Assistant replay (B_G changes no instance and no ranking input; it splits 3 clusters on these
days, so stacks change by at most 3); the saved-key check is the box check above, not switch_attach.py on page loads.

## 4. The one evaluation-day read (2026-11-23 01:07, after the freeze; nothing changed afterwards)

| day | wrong links L / P / B_G | strict links | clusters per animal | impure photos | catalogue photos without a kept instance |
|---|---|---|---|---|---|
| 2024-12-18 | 0 / 0 / 0 | 9 / 7 / 6 | 8.71 / 9.06 / 9.12 | 4 / 4 / 4 | 9 / 5 / 5 |
| 2024-12-26 | 0 / 0 / 0 | 34 / 18 / 17 | 11.50 / 11.33 / 11.37 | 0 / 2 / 1 | 4 / 2 / 2 |
| 2025-02-19 | 0 / **1** / 0 | 10 / 11 / 9 | 7.62 / 7.43 / 7.48 | 0 / 1 / 0 | 12 / 3 / 3 |
| 2025-03-21 | 0 / 0 / 0 | 12 / 9 / 9 | 8.58 / 8.12 / 8.12 | 0 / 0 / 0 | 1 / 1 / 1 |
| pooled | **0 / 1 / 0** | 65 / 45 / 41 | 9.10 / 8.99 / 9.02 | 4 / 7 / 5 | 26 / 11 / 11 |

L = the r04 root that was live until the go-live (/var/tmp/sat3737-whale-days), P = shadow (S6h), B_G = guard.
2025-02-19: the link 154~0 -> 155~0 (P 0.998) does not form; 153~0 and 154~0 are flagged (fluke and back in one
instance, far apart), 151 and 152 already hold the two animals as two instances. The guard also drops one correct
link there (133 -> 134), and one each on 2024-12-18 and 2024-12-26. Multi-whale bursts split pure 16 / 14 / 15 of 17.
Sheet: `fused/sheets/fused_eval_link.jpg`.

## 5. What adopting B_G would take (done 2026-11-23 01:55, see below)

ADOPTED 2026-11-23 01:55 (server down 4 s): `safe_tracklets cluster --guard-fused` (default off; the logic lives in the
tracked `model_fine_tuning/cluster/fused_guard.py`, a line-for-line copy of the frozen flag_G here, which stays as the
frozen record). Checks before the swap, in a scratch root: flag off byte-identical to the live clusters.json on all 29
days with safe clusters; flag on reproduces the stored B_G numbers on all 24 train + val days (strict links 177,
clusters 3,523, flagged 57; links instance for instance) and on the 4 evaluation days (reproduction only); flag sets
identical to this study's on all 29 days (incl. the unculled day). Readers re-made: link_scores pairs, log_cards
write, fin_sketch draw. Saved state needs no migration (keys only; the cluster-level records with ids sit on
2025-01-02, whose ids do not change). Backup and roll-back: moonshot_cache/switch_root/backup_clusters_pre_guard_*,
ROLLBACK_guard.sh. New days: run_pipeline.sh and shadow_catalog_fix.sh pass --guard-fused.

What it took (the plan as written before):

- Code: an optional flag, default off, in `cluster/safe_tracklets.py cluster` (e.g. `--guard-fused`): compute flag_G
  per instance from DAY/cse (flags.flag_G + parts.split_instances, CPU, about 3 ms per photo) and AND it into the strict
  allow mask (end instances of a candidate pair). No model, threshold or key changes; no GPU.
- Re-run per day: `safe_tracklets cluster --strict --threshold 0.995 --filter --guard-fused` (and the catalogue re-make
  with `--labels` as shadow_catalog_fix.sh does), then the readers of safe/clusters.json: `link_scores pairs` (review
  queue), `log_cards write`, `fin_sketch draw` on days with drawings; move the per-day cluster caches of
  Whale-Review-Web/data (cluster_views.json) aside. Seconds per day on the CPU; minutes for all 28 days + test day.
- Effect: cluster ids change only on days where a link is dropped (8 of 24 train + val days, 3 of 4 eval days; one
  cluster splits in two, later ids shift by one). Instance keys do not change, so labeling.json / pair_marks.json /
  review_extra.json need no migration (/bursts links follow photo keys). Cluster ids saved anywhere must be mapped by
  instance key, as for the switch.
- Since 2026-11-22 22:16 the shadow root IS the live root (days_all -> switch_root/days); the re-run would write
  safe/clusters.json there, so it needs the user's go and a copy of the old clusters.json files first.

## Not verified / limits

- flag_G misses fused instances whose two animals touch in the mask (one component): 3 of the 5 click-fused other
  whales, and on 2025-02-19 the guard works because 153-154 happen to be two components.
- The geometry threshold (0.20) was set by eye on train + val instances; the by-eye labels are mine, not the
  researcher's, and 29 of 80 are "unclear".
- Splits fail by 1 on the fluke / main-whale counts (n 83 / 363): within a few clicks, but the rule was fixed first.
- SAM 3 covered all photos only on the 8 candidate days; flag_S elsewhere was not measured.
- Click counts moved between my first run (17 other whales, 365 main whales) and the scoring run (16, 363) because the
  label files were migrated at the go-live (22:16); all arms were scored on one snapshot taken after it.
