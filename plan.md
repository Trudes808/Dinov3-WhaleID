# Whale Clustering Pipeline Plan

## Goal

Build a clean, modular pipeline that clusters whale photos by individual within each day of field images. The notebook should stay lean: it should configure a run, call helper functions, and display plots/review artifacts. Core logic should live in Python modules that are easy to test, cache, reuse, and eventually fine-tune.

The first mature baseline should use zero-shot computer vision methods. Once the pipeline is stable and evaluation is reliable, the expert-labeled Lightroom catalog can support supervised fine-tuning for body-part classification, whale re-identification, segmentation, and/or pairwise same-whale scoring.

## Current Data Sources

- Image root: `/home/sat3737/Test/Lightroom images`
- Lightroom catalog: `/home/sat3737/Test/Test.lrcat`
- Old reference notebook: `initial_work_old/dinov3_whale_individuals_experiments.ipynb`

Observed image layout:

```text
/home/sat3737/Test/Lightroom images/
  Topside research/
    2025 Research/
      2025-1-15/
        2025-1-15 C1 Daschbach/
          2025-1-15-Daschbach-NOAA-MMHSRP-24359-226-184A6488.CR2
          2025-1-15-Daschbach-NOAA-MMHSRP-24359-227-184A6489.CR2
        2025-1-15 C3 Lyman/
          ...
      2025-4-1/
        2025-4-1 C1 Lyman/
        2025-4-1 C3 Harvey/
```

The folder and filename conventions are useful, but the ingestion layer should normalize them into metadata instead of depending on path strings throughout the project.

## Proposed Repository Structure

```text
Dinov3-WhaleID/
  plan.md
  notebooks/
    whale_clustering_exp.ipynb
    whale_clustering_helpers.py
    plotting_utils.py
  helpers/
    __init__.py
    lrcat_parser.py
    image_manifest.py
    raw_previews.py
    series_detection.py
    whale_isolation.py
    feature_extraction.py
    clustering.py
    evaluation.py
  runs/
    whale_clustering/
      manifests/
      raw_catalog_exports/
      previews/
      crops/
      masks/
      embeddings/
      series/
      clusters/
      reports/
  initial_work_old/
    ... reference material only ...
```

Expected module responsibilities:

- `notebooks/whale_clustering_exp.ipynb`: compact experiment runner and visual review notebook.
- `notebooks/whale_clustering_helpers.py`: notebook-facing orchestration wrappers and configuration objects.
- `notebooks/plotting_utils.py`: contact sheets, overlays, cluster grids, error galleries, and metric plots.
- `helpers/lrcat_parser.py`: Lightroom catalog schema inspection and extraction of expert labels, collection membership, and color labels.
- `helpers/image_manifest.py`: filesystem scan, metadata parsing, and joining image records to Lightroom-derived labels.
- `helpers/raw_previews.py`: RAW image preview/JPEG cache creation for `.CR2`, `.CR3`, and other camera formats.
- `helpers/series_detection.py`: timestamp/sequence grouping, visual boundary validation, and outlier detection.
- `helpers/whale_isolation.py`: zero-shot whale detection, masks, crop generation, and crop quality scoring.
- `helpers/feature_extraction.py`: DINOv3/global embeddings, crop embeddings, patch embeddings, and optional local descriptors.
- `helpers/clustering.py`: series-to-series match scoring, graph construction, and day-level clustering.
- `helpers/evaluation.py`: metrics against ground truth, threshold sweeps, and failure categorization.

## Ground Truth From Lightroom Catalog

The `.lrcat` file should be treated as the authoritative expert label source. Lightroom catalogs are SQLite databases, but their schema can vary by Lightroom version, so the parser should first inspect available tables and columns before extracting records.

### Label Semantics

- Animal collections use the convention `G02A01`.
- `G02A01` means `Group02 Animal01` for a given day.
- The unique whale label should be:

```text
date-G##A##
```

For example:

```text
2025-01-15-G02A01
```

### Lightroom Catalog Fields To Extract

The parser should attempt to recover:

- original file path or enough folder/file metadata to join to the image manifest
- Lightroom image ID
- collection membership
- collection hierarchy, if needed to infer date/day context
- animal collection names matching `G\d+A\d+`
- derived unique whale ID, `date-G##A##`
- color label / highlight metadata
- green-highlight flag for scarring studies
- yellow-highlight flag for other studies
- rating, pick/reject, or other useful curation flags if present

### Catalog Parser Strategy

1. Open the `.lrcat` read-only with Python's built-in `sqlite3` library.
2. Export a schema summary for debugging:
   - table names
   - relevant columns
   - row counts for candidate tables
3. Identify tables that represent:
   - images/files/folders
   - collections
   - image-to-collection membership
   - labels/color labels
4. Reconstruct image paths from folder + basename fields where possible.
5. Match animal collections with a regex like `G(\d+)A(\d+)`.
6. Infer date from collection hierarchy, image path, filename, or folder name.
7. Emit a normalized ground-truth table:

```text
image_path
image_stem
capture_date
day_label
lightroom_image_id
collection_name
ground_truth_group
ground_truth_animal
ground_truth_whale_id
is_scarring_study_green
is_other_study_yellow
all_collection_names
```

8. Join this table to the filesystem manifest using normalized absolute paths first, then filename/date fallbacks.
9. Report unmatched Lightroom records and unmatched filesystem images.

### Evaluation Uses

The ground truth enables:

- automatic cluster quality scoring by day
- pairwise same-whale precision/recall/F1
- adjusted Rand index and normalized mutual information
- cluster purity
- over-split and over-merge diagnostics
- review galleries of incorrect merges/splits
- stratified evaluation on green-highlight scarring-study images
- optional separate reporting for yellow-highlight study images

## Processing Pipeline

```text
Filesystem Images              Lightroom Catalog
       |                              |
       v                              v
Build Image Manifest          Parse Expert Labels
       |                              |
       +--------------+---------------+
                      v
          Manifest + Ground Truth Join
                      |
                      v
        Create Cached Working Previews
                      |
                      v
       Initial Temporal Series Detection
        day + camera + photographer + sequence/time gaps
                      |
                      v
          Visual Boundary Validation
  adjacent similarity + crop similarity + outlier detection
                      |
                      v
        Whale Detection / Masking / Cropping
                      |
                      v
          Body-Part / View Classification
       left, right, dorsal, fluke, unknown
                      |
                      v
              Feature Extraction
    global crop embeddings + patch/local descriptors
                      |
                      v
          Series-Level Representation
 best images + body-part coverage + aggregate features
                      |
                      v
         Series-to-Series Match Scoring
 compatible views + local evidence + quality weighting
                      |
                      v
          Graph / Hierarchical Clustering
              N whale clusters per day
                      |
                      v
        Evaluation + Plots + Review Exports
```

## Pipeline Components

### 1. Image Manifest

Create one record per image. The manifest should include:

- absolute path
- relative path under image root
- filename and stem
- extension
- day/date parsed from folder or filename
- camera ID, such as `C1` or `C3`, when available
- photographer when available
- sequence number parsed from filename when available
- EXIF timestamp when available
- image dimensions
- RAW/JPEG preview path
- Lightroom join status
- ground-truth whale ID and study flags when available

### 2. RAW Preview Cache

The observed data includes `.CR2` and `.CR3` RAW files. Most CV models expect RGB images, so the pipeline should create cached working images.

Initial options:

- use embedded RAW previews for speed
- use `rawpy` if high-quality conversion is needed
- support existing JPG/TIFF/PNG files directly

The cache should be deterministic and reusable across notebook runs.

### 3. Initial Series Detection

Photos are often taken as short runs where the same whale appears for 2-10 images, then the sequence switches to a different whale or view.

Initial grouping should happen within each `day + camera/photographer` stream using:

- EXIF timestamp gaps
- filename sequence gaps
- folder boundaries
- camera/photographer boundaries

Output:

```text
image_id -> provisional_series_id
```

### 4. Series Boundary Validation

The temporal grouping is only a proposal. The pipeline should validate boundaries visually and flag outliers.

Zero-shot signals:

- DINOv3 similarity between adjacent frames
- DINOv3 similarity between whale crops once crops exist
- perceptual hash or SSIM for cheap continuity checks
- abrupt changes in whale-mask location or crop size
- local feature matching for higher-confidence continuity checks

Expected outputs:

- final series ID
- boundary confidence
- outlier flags
- split/merge suggestions for review

### 5. Whale Isolation

Water and background similarity should not drive whale identity matching.

Initial zero-shot approaches:

- DINO attention/PCA foreground mask, inspired by the old notebook
- SAM/SAM2-style zero-shot segmentation if available
- crop proposals based on foreground masks and saliency

Outputs:

- whale bounding box
- whale mask
- cropped whale image
- crop quality score
- mask/crop review plots

### 6. Body-Part And View Classification

Human matching uses different evidence depending on whether the image shows left side, right side, dorsal, or fluke. The pipeline should model this explicitly.

Initial zero-shot strategy:

- cluster whale-crop embeddings into pose/view groups
- manually inspect and map clusters to view labels
- optionally use CLIP-style text prompts as an auxiliary signal

Target labels:

- left side
- right side
- dorsal
- fluke
- unknown / low-quality

### 7. Feature Extraction

Use multiple feature levels:

- DINOv3 full-image embedding for broad context
- DINOv3 whale-crop embedding for identity-relevant signal
- DINOv3 patch embeddings for local similarity
- local descriptors or feature matches for scars, barnacles, spots, dorsal shape, and fluke shape
- optional shape descriptors from masks/edges

All expensive outputs should be cached by image path, file metadata, model name, and preprocessing settings.

### 8. Series Representation

Each series should become a structured object, not just a list of images.

Suggested fields:

- series ID
- day/date
- camera/photographer
- time or sequence range
- image IDs
- best representative images
- body-part coverage
- per-image quality scores
- aggregate embeddings
- per-view embeddings
- known ground-truth labels for evaluation only

### 9. Series-To-Series Matching

Compare series to other series within the same day.

Scores should be view-aware:

- left-to-left comparisons should be strong evidence
- right-to-right comparisons should be strong evidence
- dorsal-to-dorsal comparisons should be strong evidence
- fluke-to-fluke comparisons should be strong evidence
- incompatible views should either be ignored or down-weighted
- a series containing multiple views can bridge evidence across series

Pair scoring can combine:

- maximum compatible image-pair similarity
- mean top-k compatible similarity
- local feature match quality
- body-part confidence
- crop/mask quality
- time/camera context as a weak prior

### 10. Day-Level Clustering

Represent the problem as a graph:

- nodes: validated series
- edges: likely same-whale pair scores

Initial clustering options:

- connected components after thresholding pair scores
- agglomerative clustering on series distances
- HDBSCAN or related density clustering if the representation supports it

The output should be:

```text
day -> whale_cluster_id -> series -> images
```

### 11. Evaluation And Review

Use the Lightroom-derived ground truth for repeatable scoring.

Core metrics:

- pairwise precision
- pairwise recall
- pairwise F1
- adjusted Rand index
- normalized mutual information
- cluster purity
- number of over-split ground-truth whales
- number of over-merged predicted clusters

Review outputs:

- cluster contact sheets
- series contact sheets
- incorrect merge galleries
- incorrect split galleries
- threshold sweep plots
- green-highlight scarring-study subset reports
- yellow-highlight subset reports

## Zero-Shot First Strategy

The initial pipeline should avoid training and focus on measurable baselines:

- parse Lightroom ground truth for evaluation only
- build a manifest and preview cache
- identify provisional series from timestamps/sequences
- validate series boundaries with zero-shot image similarity
- isolate whales with zero-shot masks/crops
- extract DINOv3 embeddings
- cluster series within a selected day
- evaluate against Lightroom labels
- inspect failure cases visually

This gives a reproducible benchmark before adding supervised learning.

## Future Fine-Tuning Paths

Once the zero-shot pipeline and labels are trustworthy, labeled data can support:

- body-part/view classifier fine-tuning
- whale crop detector or segmentation model fine-tuning
- contrastive whale re-identification embedding fine-tuning
- pairwise same-whale classifier fine-tuning
- active learning, where uncertain cluster pairs are prioritized for expert review

Fine-tuning should be added only after the pipeline can produce stable manifests, reliable train/validation/test splits by day or whale, and clear baseline metrics.

## Initial Implementation Milestones

### Milestone 1: Clean Project Skeleton

- Create `notebooks/` and `helpers/`.
- Create `whale_clustering_exp.ipynb` as a lean runner notebook.
- Create `whale_clustering_helpers.py` and `plotting_utils.py`.
- Create helper modules with clear interfaces.

### Milestone 2: Manifest And Lightroom Ground Truth

- Implement catalog schema inspection.
- Parse collections matching `G##A##`.
- Extract green/yellow color labels.
- Build image manifest from filesystem.
- Join Lightroom labels to filesystem images.
- Produce unmatched-record reports.

### Milestone 3: Single-Day Baseline

- Select one day with good labels.
- Create RAW preview cache.
- Detect provisional series.
- Plot series contact sheets.
- Compute zero-shot embeddings.
- Run simple clustering.
- Evaluate against ground truth.

### Milestone 4: Whale Isolation And Better Matching

- Add foreground masks/crops.
- Add crop quality scoring.
- Add body-part/view grouping.
- Add view-aware series matching.
- Compare metrics against the Milestone 3 baseline.

### Milestone 5: Review And Export Workflow

- Export final clusters by day.
- Export review galleries.
- Add threshold sweeps and diagnostics.
- Make it easy to rerun experiments with different settings.

## Open Questions

- Which date/day should be the first target for a baseline run?
- Does the Lightroom catalog store color labels in standard color-label fields, collection names, smart collections, or another metadata field?
- Are animal collections nested under date collections, or must date be inferred from image path/filename?
- Are there already exported JPG previews, or should we build RAW preview extraction immediately?
- Do ground-truth animal collections ever intentionally include only the best ID images rather than every image of that whale?
- Should green/yellow study flags affect clustering, or should they only be used for stratified evaluation/reporting?

## Notebook Shape

The experiment notebook should stay close to this pattern:

```python
from whale_clustering_helpers import WhaleClusteringConfig
from whale_clustering_helpers import build_manifest, load_ground_truth
from whale_clustering_helpers import detect_series, compute_artifacts
from whale_clustering_helpers import score_series_matches, cluster_day, evaluate_clusters
from plotting_utils import plot_manifest_summary, plot_series_grid, plot_cluster_review

config = WhaleClusteringConfig(
    image_root="/home/sat3737/Test/Lightroom images",
    lrcat_path="/home/sat3737/Test/Test.lrcat",
    run_dir="../runs/whale_clustering/baseline_zero_shot",
    target_day="2025-01-15",
)

ground_truth = load_ground_truth(config)
manifest = build_manifest(config, ground_truth)
series = detect_series(manifest, config)
artifacts = compute_artifacts(series, config)
matches = score_series_matches(artifacts, config)
clusters = cluster_day(matches, config)
metrics = evaluate_clusters(clusters, ground_truth, config)

plot_manifest_summary(manifest)
plot_series_grid(series)
plot_cluster_review(clusters, manifest, ground_truth)
```