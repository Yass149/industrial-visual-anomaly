# File and function map

Use [the visual guide](START_HERE.html) for the big picture and [DESIGN.md](DESIGN.md) for mechanics.

This index is generated from the current source by `make report`. Private helpers appear too.

## [src/anomaly/__init__.py](../src/anomaly/__init__.py)

PatchCore anomaly detection and localisation on MVTec AD.


## [src/anomaly/api.py](../src/anomaly/api.py)

Load the verified model once and serve health, frontend and image predictions.

- `Prediction` (line 30): Prediction
- `create_app` (line 40): Load artifacts once at startup; imports and unit tests do not download a backbone.
- `lifespan` (line 46): Lifespan
- `health` (line 87): Health
- `index` (line 93): Index
- `predict_bytes` (line 96): Predict bytes
- `predict` (line 133): Predict

## [src/anomaly/config.py](../src/anomaly/config.py)

Validate the single YAML file, resolve paths and choose a device.

- `_Base` (line 42): Base
- `DataConfig` (line 46): Dataconfig
- `ModelConfig` (line 89): Modelconfig
- `EvalConfig` (line 125): Evalconfig
- `ThresholdConfig` (line 131): Thresholdconfig
- `ServeConfig` (line 144): Serveconfig
- `MonitoringConfig` (line 152): Monitoringconfig
- `Config` (line 159): Config
- `repo_root` (line 179): Repository root, derived from this file's location rather than the cwd.
- `load_config` (line 184): Load and validate the YAML config, resolving relative paths against the repo root.
- `_absolutise` (line 195): Make every path absolute so behaviour does not depend on the working directory.
- `get_config` (line 207): Cached default config, for callers that do not take a config argument (e.g. the API).
- `resolve_device` (line 212): Turn the config's device spec into a concrete torch device string.
- `_expand_root` (line 57): Expand root
- `_known_categories` (line 64): Known categories
- `_looks_like_sha256` (line 74): Looks like sha256
- `extract_dir` (line 80): Directory holding the per-category trees, e.g. data/mvtec/bottle/train/good.
- `archive_path` (line 85): Archive path
- `_known_layers` (line 101): Known layers
- `_odd_neighbourhood` (line 111): Odd neighbourhood
- `_crop_fits` (line 119): Crop fits
- `_positive_sweep` (line 138): Positive sweep
- `_serve_category_is_built` (line 170): Serve category is built

## [src/anomaly/coreset.py](../src/anomaly/coreset.py)

Choose representative normal descriptors with a seeded greedy coverage algorithm.

- `_random_projection` (line 23): Johnson-Lindenstrauss projection to speed up the greedy selection.
- `greedy_coreset_indices` (line 39): Select ceil(ratio * N) representative rows of `features`; returns their indices.

## [src/anomaly/data.py](../src/anomaly/data.py)

Load normal/test samples, align masks and share deterministic transforms.

- `Sample` (line 29): One image with its label, ground-truth mask and provenance.
- `build_transform` (line 39): The image preprocessing pipeline. Deterministic: no augmentation, anywhere.
- `build_display_transform` (line 55): The geometric half of `build_transform`, without normalisation.
- `build_mask_transform` (line 70): Mask preprocessing: same geometry as the image, nearest-neighbour to stay binary.
- `load_image` (line 81): Open an image as RGB. Several MVTec categories ship single-channel PNGs.
- `MVTecDataset` (line 86): One MVTec AD category and split.
- `collate_samples` (line 152): Stack a list of Samples into batched tensors, keeping paths and defect types as lists.
- `__init__` (line 93): Init
- `__len__` (line 124): Len
- `__getitem__` (line 127): Getitem

## [src/anomaly/evaluation.py](../src/anomaly/evaluation.py)

Create disjoint split indices and calculate honest image-level metrics.

- `split_indices` (line 17): Stratify by defect type, preserving at least one example per side.
- `image_metrics` (line 41): Return raw benchmark PR and a prevalence-adjusted deployment scenario separately.

## [src/anomaly/features.py](../src/anomaly/features.py)

Extract frozen multi-layer image descriptors and flatten spatial patches.

- `PatchFeatureExtractor` (line 24): Extracts a single concatenated patch-feature map from selected backbone layers.
- `flatten_patches` (line 104): (B, D, H, W) -> (B*H*W, D), one row per patch, row-major in (b, h, w).
- `__init__` (line 31): Init
- `_make_hook` (line 64): Make hook
- `feature_dim` (line 71): Descriptor length D, i.e. the summed channel counts of the selected layers.
- `forward` (line 78): Map a batch of images to patch features of shape (B, D, H, W).
- `hook` (line 65): Hook

## [src/anomaly/monitoring.py](../src/anomaly/monitoring.py)

Compare disjoint incoming windows with the training feature reference.

- `DriftMonitor` (line 13): Compare image-level feature projections in disjoint incoming windows to training.
- `__init__` (line 20): Init
- `update` (line 41): Append one image; test only complete nonoverlapping windows and log warnings.

## [src/anomaly/patchcore.py](../src/anomaly/patchcore.py)

Fit, persist, load and score category banks; return maps and monitoring summaries.

- `ScoreResult` (line 32): Per-image anomaly score and the pixel map it was derived from.
- `PatchCore` (line 40): Memory-bank anomaly detector. Fit on normal images, scored by distance to the bank.
- `__init__` (line 43): Init
- `fit` (line 60): Build the memory bank from a loader over normal images only.
- `_nearest_distances` (line 120): For each row of `patches`, distance to and index of its nearest bank entry.
- `_bank_neighbours` (line 134): (K, b) indices of each bank entry's b nearest bank entries, itself first.
- `score` (line 146): Score a batch of preprocessed images.
- `_reweighted_image_scores` (line 187): PatchCore's reweighted image score.
- `save` (line 227): Write the memory bank and the settings that produced it.
- `load` (line 243): Load a memory bank, refusing it if it was built with incompatible settings.
- `bank_size_mb` (line 267): Bank tensor size in decimal megabytes, excluding backbone, metadata and reference.

## [src/anomaly/provenance.py](../src/anomaly/provenance.py)

Fingerprint artifacts and scoring settings so thresholds cannot go stale silently.

- `artifact_digest` (line 12): Hash in chunks so fingerprints do not duplicate an artifact in memory.
- `scoring_signature` (line 21): Thresholds are invalid if preprocessing or scoring settings have changed.

## [src/anomaly/reproducibility.py](../src/anomaly/reproducibility.py)

Set Python/NumPy/torch seeds and deterministic-kernel preferences.

- `set_seed` (line 16): Seed Python, NumPy and torch.
- `enable_determinism` (line 24): Ask torch for deterministic kernels where it has them.

## [src/anomaly/threshold.py](../src/anomaly/threshold.py)

Minimise prevalence-weighted false-alarm and missed-defect costs.

- `OperatingPoint` (line 12): Operatingpoint
- `validate_scores` (line 25): Require both classes so FPR and FNR are defined rather than silently invented.
- `optimise_threshold` (line 35): Minimise C_FN * prevalence * FNR + (1-prevalence) * FPR on calibration data.
- `to_dict` (line 20): JSON-safe record, with costs measured in false-alarm cost units per image.

## [src/anomaly/viz.py](../src/anomaly/viz.py)

Normalise maps, blend aligned heatmaps and encode PNG responses.

- `normalise_map` (line 14): Scale an anomaly map to [0, 1].
- `overlay_heatmap` (line 35): Blend a colourised anomaly map over an image of the same size.
- `to_base64_png` (line 56): Encode an image as a base64 PNG string, for embedding in a JSON response.

## [scripts/benchmark_api.py](../scripts/benchmark_api.py)

Measure warmed sequential CPU requests through the actual API.

- `main` (line 16): Main

## [scripts/download_data.py](../scripts/download_data.py)

Fetch the archive, check its checksum, extract it and verify category structure.

- `sha256_file` (line 38): Stream a file through SHA-256 without loading it into memory.
- `download` (line 52): Download `url` to `dest`, resuming from a partial file if one is present.
- `extract` (line 84): Extract the archive into `target`, stripping its single top-level directory.
- `verify_layout` (line 117): Check every configured category is present and well-formed; return train image counts.
- `already_present` (line 134): True when every configured category is already extracted, so the work can be skipped.
- `main` (line 143): Main

## [scripts/evaluate.py](../scripts/evaluate.py)

Calibrate thresholds and produce holdout metrics, manifests, maps and plots.

- `write_json` (line 31): Reject nonstandard JSON rather than shipping NaN thresholds to the service.
- `plot_errors` (line 37): Show actual errors only, with common colour bounds and ground-truth contours.
- `evaluate_category` (line 89): Create metrics, split manifest, calibrated threshold, and diagnostic figures.
- `main` (line 228): Main

## [scripts/score_image.py](../scripts/score_image.py)

Score one file or example and save a relatively coloured heatmap.

- `main` (line 27): Main

## [scripts/serve.py](../scripts/serve.py)

Start a single service from YAML host, port and category settings.

- `main` (line 13): Main

## [scripts/snapshot_report.py](../scripts/snapshot_report.py)

Refresh committed evidence, guide data and this file/function index.

- `main` (line 36): Main

## [scripts/train.py](../scripts/train.py)

Orchestrate normal-only fitting per category and save a training summary.

- `train_category` (line 26): Fit a memory bank for one category and write it to the artifacts directory.
- `main` (line 67): Main

## [tests/conftest.py](../tests/conftest.py)

Package marker or automated regression checks.

- `TinyExtractor` (line 12): Deterministic spatial features; exercise real scoring without a network download.
- `cfg` (line 20): Cfg
- `tiny_model` (line 41): Tiny model
- `drift_reference` (line 52): Drift reference
- `forward` (line 15): Forward

## [tests/test_api.py](../tests/test_api.py)

Package marker or automated regression checks.

- `app` (line 14): App
- `image_bytes` (line 31): Image bytes
- `test_api_schema_overlay_and_bad_uploads` (line 37): Test api schema overlay and bad uploads
- `test_stale_threshold_refused` (line 60): Test stale threshold refused

## [tests/test_core.py](../tests/test_core.py)

Package marker or automated regression checks.

- `test_preprocessing_is_identical_for_same_pixels` (line 11): Test preprocessing is identical for same pixels
- `test_local_defect_scores_higher_and_localises` (line 17): Test local defect scores higher and localises
- `test_coreset_has_unique_indices_even_for_identical_points` (line 29): Test coreset has unique indices even for identical points
- `test_coreset_covers_rare_normal_tail` (line 36): Test coreset covers rare normal tail
- `test_coreset_reproducible_projection` (line 42): Test coreset reproducible projection
- `test_empty_fit_and_anomalous_training_rejected` (line 49): Test empty fit and anomalous training rejected
- `test_persistence_and_config_guard` (line 58): Test persistence and config guard
- `test_bank_neighbour_zero_is_self` (line 70): Test bank neighbour zero is self

## [tests/test_data_config.py](../tests/test_data_config.py)

Package marker or automated regression checks.

- `test_config_rejects_typos_and_nonfinite_costs` (line 12): Test config rejects typos and nonfinite costs
- `write_image` (line 27): Write image
- `test_train_rejects_anomalous_folders` (line 32): Test train rejects anomalous folders
- `test_masks_missing_then_correctly_binary_and_aligned` (line 38): Test masks missing then correctly binary and aligned

## [tests/test_evaluation.py](../tests/test_evaluation.py)

Package marker or automated regression checks.

- `test_extreme_costs_and_tied_scores` (line 8): Test extreme costs and tied scores
- `test_optimiser_matches_exhaustive_cost` (line 16): Test optimiser matches exhaustive cost
- `test_invalid_threshold_inputs` (line 41): Test invalid threshold inputs
- `test_split_disjoint_exhaustive_and_reproducible` (line 46): Test split disjoint exhaustive and reproducible
- `test_prevalence_adjustment_and_confusion_order` (line 55): Test prevalence adjustment and confusion order

## [tests/test_integration.py](../tests/test_integration.py)

Opt-in evidence against real local data; CI tests do not download a 5GB dataset.

- `test_real_fixture_ordering` (line 19): Test real fixture ordering
- `test_real_api_end_to_end` (line 34): Test real api end to end

## [tests/test_monitoring.py](../tests/test_monitoring.py)

Package marker or automated regression checks.

- `test_window_and_shift_warning` (line 6): Test window and shift warning
- `test_unchanged_reference_does_not_warn` (line 17): Test unchanged reference does not warn

## Non-Python files

- `configs/default.yaml`: category, preprocessing, model, cost, service and monitoring settings.
- `src/anomaly/frontend.html`: plain drag-and-drop upload and prediction display.
- `pyproject.toml`: package metadata, compatible dependency ranges, lint and test configuration.
- `requirements.lock`: exact environment pins; update deliberately, then revalidate.
- `Makefile`: the task entry points. `reproduce` runs stages sequentially.
- `Dockerfile` / `.dockerignore`: container build, non-root worker and readiness check; build unverified locally.
- `.github/workflows/ci.yml`: lint and offline tests on pushes and pull requests; remote execution unverified.
- `.gitignore`: excludes datasets, banks, generated runtime results and local assistant state.
- `README.md`: short measured summary, failures, reproduction and next steps.
- `docs/START_HERE.html`: offline visual chapters with a calibration-only cost explorer and saved reading position.
- `docs/DESIGN.md`: detailed technical walkthrough and limitations.
- `docs/DECISIONS.md`: inherited/new choices and alternatives; no invented ablation claims.
- `docs/RESUME.md`: quick return note, verification and operational commands.
- `docs/assets/`: compact reference evidence and attributed diagnostic figures; not training data.
- `artifacts/` (ignored): fitted banks, monitoring references and calibrated thresholds.
- `results/` (ignored): metrics, exact split manifests, pixel predictions, diagnostics and benchmarks.
- `.venv/` (ignored): installed local Python environment.
