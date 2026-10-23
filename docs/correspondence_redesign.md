# Redesign: from "mesh fit" to dense image-to-template correspondence (plan, 2026-10-16)

## Goal and success criterion

For every whale pixel, say *which point of the body* it shows (template vertex, hence body part, side and
canonical position), so that two photos are compared **portion to portion**: fluke to fluke, dorsal fin to dorsal
fin, a scar on the right flank to the same canonical location on another right flank. The 3D pose is not the
product; it becomes a diagnostic.

Success is measured on real photos, not on renders:
1. **Same-day pair AUC per part type** on the labelled eval days (2024-12-18, 2024-12-26, 2025-02-19, 2025-03-21:
   2 167 photos, 92 identities; `splits/days.json`), against the current pipeline (global canon map -> charts).
2. **Map quality on real photos**: mirror consistency of the vertex map (part and side must survive a flip), part
   agreement with 30-50 photos x ~5 clicked points (to be collected in the review app; the only direct check).
3. On synthetic: geodesic error of the predicted vertex, part accuracy, side accuracy, visible-part coverage error.

## Why the current chain fails on real photos

Folded-xyz regression -> per-whale side resolution -> rigid PnP -> part charts. Measured 2026-10-16:
real-photo above-water IoU 0.22 (run04 model) / 0.31 (run03), side flips under mirroring on ~18 % of crops, PnP
rotation median 31 deg even on synthetic data because every run04 body is bent; the bendable fit (fit_mesh.py)
recovers some of it (real IoU 0.22 -> 0.28, top edge 7 -> 2 px) but is a patch on the wrong formulation. The
identity step never needed the pose; it needs the correspondence.

## Literature (checked 2026-10-16)

- Continuous Surface Embeddings (Neverova et al., NeurIPS 2020): per-pixel embedding matched to learned per-vertex
  embeddings of a template; DensePose for animals with sparse labels. SHIC (2024) learns such maps from foundation
  features with no keypoints. We have exact dense labels from the factory, far more than either had.
- DINOv2/v3 features are the standard backbone for correspondence but are geometrically ambiguous: left/right
  swaps ("Telling Left from Right", CVPR 2024; zero-shot DINOv3 matching, 2026) and a positional bias (INSID3,
  CVPR 2026). Geometry-aware supervision and viewpoint conditioning fix it; a 2025 benchmark finds fine-tuning only
  the last stage of DINOv2 best for correspondence (sim-to-real argues for a mostly frozen backbone here).
- Template-based animal 3D (AniMer CVPR 2025, 4D-Animal WACV 2026, Dessie 2024): DINO features -> template
  parameters, refined by differentiable rendering. The rig + spine labels make this a later, optional layer.
- Animal re-ID (WildFusion 2024/25, AnimalCLEF 2025-26, MegaDescriptor): global descriptor + local keypoint matches
  (LightGlue / LoFTR) with calibrated fusion is the state of the art; part alignment is done by detector crops, not
  by a surface map. The map makes the part definition exact.

## Architecture

### A. Correspondence head (replaces canon regression, side resolution and PnP)

- **Template**: the 4k canonical mesh (K9239 frame), farthest-point downsampled to ~4 000 vertices; per-vertex
  learnable embeddings (32-D); precomputed geodesic distances (for soft targets and the error metric); a vertex ->
  part table (fluke L/R dorsal and ventral, peduncle, dorsal fin, flank L/R, back, head, pectoral L/R, belly) and
  a vertex -> chart UV table (the existing part charts).
- **Head**: DINOv3 ViT-B/16, backbone frozen or last 1-2 blocks tuned (to be tested); the existing DenseDecoder
  (stride-4 output) emitting a 32-D embedding per pixel and a whale/background logit; the existing direct rotation
  head kept as the viewpoint context.
- **Loss**: cross-entropy over the vertex table per whale pixel with geodesic-Gaussian soft targets (sigma ~15 cm),
  visibility BCE (contour-weighted or not: it made no difference), and a whale-level side-consistency term: the
  side implied by each pixel's vertex must agree with the whale's majority side and with the rotation head's
  facing. Mirror augmentation swaps left/right vertices (the template is not folded).
- **Inference**: per pixel the arg-max vertex and its confidence; whale-level side vote (rotation head + pixel
  majority) that flips inconsistent pixels to their mirror vertex; coverage per part = visible vertex set; canonical
  UV per pixel from the vertex table. Bending never enters: the map says which surface point, however the body is
  curved.
- **Data**: run04 (+ run05) dense labels (per-pixel canonical xyz -> nearest template vertex; instances for the
  calf); real photos with the EMA-teacher mirror-consistency loss (now a vertex-swap consistency); clicked points
  when available.
- **Metrics**: synthetic geodesic error (cm), part accuracy, side accuracy, coverage error per part; real: mirror
  consistency of part and side, clicked-point geodesic error; the usual pose-class grid for eyeballing.

### B. Portion-to-portion identity (the deliverable)

1. Coverage per part from the map; a pair of photos is compared on the parts both show with enough pixels.
2. Each shared part rectified into its canonical chart directly through the correspondences (no 3D fit);
   `cluster/parts.py` already builds charts from a canon map and swaps in.
3. Per chart: global embedding (the ArcFace part model in `cluster/train_id.py`, or MegaDescriptor) + local keypoint
   matches (LightGlue / ALIKED) **in chart coordinates**, so a mark only matches a mark near the same canonical
   location; geometric verification = chart-space proximity.
4. Calibrated fusion per part type (WildFusion-style isotonic calibration), then the existing time-aware
   clustering. Report pair AUC per part type and overall on the eval days.

### C. 3D pose and bend (optional, later)

A rig-parameter head trained on run04's spine labels, refined with a differentiable renderer (nvdiffrast /
PyTorch3D) against the mask and the map. For QA, occlusion reasoning and synthetic-vs-real diagnostics only.

## Milestones

| step | work | time | decision at the end |
|---|---|---|---|
| 1 | template + geodesics + vertex labels from run04; CSE head, loss, metrics in the existing trainer; train frozen vs last-2-blocks | 2 days incl. training | backbone choice; vertex count |
| 2 | inference + side vote + coverage; real-photo mirror metrics; pose-class grid | 1 day | is the map good enough on real photos to proceed |
| 3 | chart rectification from the map; LightGlue in chart space; per-part fusion; pair AUC on the eval days vs current | 2-3 days | ship into the day pipeline or iterate on the map |
| 4 | run05 render (surfacing poses fixed, framing capped) and retrain the map | render ~1 day | |
| 5 | pose head + differentiable refinement | later | |

Collect the clicked real points in parallel (step 2 needs them): 30-50 photos, 5 points each, in the review app.

## Risks

- Left/right from a flank view is ambiguous for a symmetric body; the cues are the dorsal fin's asymmetry, the
  head direction and the travel direction. The rotation head + whale-level vote is the mitigation; measure the
  residual flip rate and let the identity step treat "side uncertain" explicitly (compare both hypotheses).
- Sim-to-real: a frozen backbone may transfer better than the 4-block fine-tune; test both; keep the real-photo
  consistency loss; the clicked points decide.
- The chart matching needs the full-resolution photo (marks are small); the day pipeline already reads RAW.
