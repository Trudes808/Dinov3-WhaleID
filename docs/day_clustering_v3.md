# Day clustering v3: instances, tracklets, part-aligned scorers (2026-10-19)

Goal: cluster one day's photos into animals (and groups) from the photos alone, with the Lightroom catalog used only
for scoring. Split discipline: eval days 2024-12-18, 2024-12-26, 2025-02-19, 2025-03-21 are never tuned on; val days
2025-01-15, 2025-02-27, 2025-03-14 select rules and thresholds; 21 train days train the identity models.

## Pipeline per day (`experiments/days_all/<day>/`)

1. `images/` 2560-px jpg from RAW, `canon/` segmentation + crops, `cse/` the CSE body map (`process_day_v2.sh`).
2. `parts_hr/` hi-res instance charts: `cluster/parts.py --instances --size 1536` cuts body / dorsal fin / fluke charts
   from the RAW for every whale instance. Instances are mask components; a component joins an instance whose body
   coverage (canonical body fraction from the CSE map) it does not duplicate, so a back and its fluke in separate blobs
   stay one whale and two backs become two instances (`<stem>~0` the largest, `~1`, ...). The index records each
   instance's area and bounding box. `scripts/parts_hr_all.sh`, `parts_hr_update_multi.sh`.
3. `cluster_v3/channels.npz`: `cluster/cluster_day.py channels --sources ... --local ...`. Every chart is scored
   against the others of its part type by each source, z-scored per part over the day's pairs:
   - `id:<ckpt>` identity model of `cluster/train_id.py` (named by experiment, e.g. `id:id_v3|fin`);
   - `md` MegaDescriptor-L-384; `hr:<tag>` the hi-res DINO head (`cluster/dino_hr.py extract`);
   - local scorers on the top-K candidate pairs: `lightglue` (ALIKED + LightGlue in chart space), `hrlocal` (mutual
     nearest neighbours of the hi-res DINO tokens, aligned in chart coordinates).
   Channels accumulate: re-running with new sources computes only the missing ones (atomic save).
4. `cluster_day.py cluster --use <sources> | --fusion <json>`: instance-pair score = best over parts of the mean z of the
   chosen sources (or `cluster/fuse.py`'s calibrated fusion fitted on the val days); nodes = tracklets
   (`cluster/tracklets.py`: instances linked through a burst by position, size and appearance; `--max-link-cost 1e9`
   = plain bursts); node score = mean of the top-3 instance-pair scores + time term (0 inside a burst); cannot-link
   constraints (`--cannot-link fin`: two dorsal-fin charts in one photo) through `cluster/linkage.py`; threshold
   self-calibrated from burst pairs (<= 2 s, same camera, different photo) vs pairs > 90 min apart, `--thr-offset`;
   groups by interleaving (`--group-window`, `--group-min-pairs`). Photos are scored on their main instance.
5. `cluster/tune_v3.py`: time weight, threshold offset / fixed threshold and group rule on the val days from the saved
   node matrices. `scripts/cluster_v3_all.sh channels|fuse|cluster`, `scripts/cluster_v3_variants.sh` run the seven days.

## What was learned (details and numbers in experiments/LOG.md, 2026-10-19)

- Hard cannot-link constraints from body coverage are wrong too often (a whale's two pectoral fins, a fluke and its
  peduncle); only "two dorsal fins in one photo" is safe, and constraints must sit on pure tracklets.
- The instance / tracklet machinery costs ~0.03 val ARI and nothing on eval with one label per photo; it is kept for
  multi-whale bursts (`--max-link-cost 1.5 --cannot-link fin`), plain bursts remain the best-scoring default.
- The identity model is the lever: id_v3 (group-hard negatives, best epoch 3) beats id_v2 on every day. Learned
  fusions and ensembles raise pair AUC but lower the clustering ARI on the held-out days; keypoint matching
  (LightGlue) adds nothing on this imagery.
- Hi-res DINO head (`cluster/dino_hr.py`, 1024 px, masked attention pooling): val pair AUC 0.856, the best fluke channel,
  clustering on par with the 448-px identity models (eval 0.373); its local token matches (`hrlocal`) are weak
  (AUC ~0.6 within candidate pairs). Final table: `scripts/v3_summary.py` (eval mean ARI 0.298 whole-crop MegaDescriptor
  -> 0.342 id_v2 -> 0.379 id_v3 / 0.373 id_v4 / 0.373 hi-res DINO).

## Signatures: the most distinctive regions per whale (2026-10-19 afternoon)

`cluster/signature.py` (+ `cluster/shape_sig.py`) writes `<day>/signature/{summary.json, heat/}` for the web page
`Whale-Review-Web /signatures`: per whale token of the hi-res charts, distinctiveness = repeatability inside the burst
minus commonness across the day's other whales at the same chart position (per-chart centred tokens, one colour scale
per day); the top regions form the tracklet's signature; links through signature points alone, through the hi-res
global embedding plus the signature (default), or through the fin outline / fluke trailing edge coloured by
uniqueness against the day's population. On the seven days the signature-only link AUC is 0.49-0.66 (global
0.68-0.84), the edge shapes are at chance, and matching tokens by CSE canonical coordinates instead of chart position
is at chance at 0.5 m (the map is not precise enough on real photos yet). Screenshots: `docs/signatures_2025-02-27_*.png`.
