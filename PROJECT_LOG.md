# Brain MRI AI Decision Support - Project Log

---

## Step 22 - Streamlit Inference Alignment and OASIS Stability Hardening (Completed)

### Actions
- Updated Streamlit checkpoint resolution to prioritize the evaluated Step 2 artifact (`outputs/checkpoints/step2_balanced_2ep`) and fall back safely when unavailable.
- Added checkpoint-aware aggregation guard so `outputs/calibration/aggregation_calibration.json` is only applied when its recorded checkpoint matches the currently loaded checkpoint.
- Added evaluation-aligned aggregation fallback defaults in app runtime to prevent cross-checkpoint calibration drift.
- Aligned app preprocessing with model training/evaluation behavior:
	- filtered non-informative slices using `filter_empty_slices(..., threshold=0.05)`
	- switched 3-channel inference input from repeated single-slice channels to true 2.5D neighbor stacking (`prev, current, next`)
- Updated slice slider default behavior:
	- tumor prediction -> highest tumor-confidence slice
	- healthy prediction -> strongest non-tumor-confidence slice
- Refined healthy-case explainability rendering to keep Grad-CAM panels visually neutral (no highlighted regions) while preserving original MRI context.
- Added in-app checkpoint visibility caption for auditability during debugging.

### Files Added/Modified
- Modified: app.py

### Outcome
- Streamlit patient-level decisions are now consistent with the intended evaluation configuration and no longer use calibration artifacts from unrelated checkpoints.
- OASIS behavior became more stable by removing preprocessing mismatch and aggregation drift between app runtime and report scripts.
- UI explainability behavior now matches expected healthy-case presentation requirements.

---

## Step 1 - Project Structure Initialization (Completed)

### Actions
- Created an independent orphan branch to isolate project history.
- Initialized repository with a modular ML-first layout.
- Structured folders for preprocessing, dataset, models, training, evaluation, and UI.
- Reserved dedicated module space for future patient-level Top-K aggregation logic.
- Standardized naming conventions to keep cross-module imports clean.

### Files Added/Modified
- Added: README.md
- Added: PROJECT_LOG.md
- Added: requirements.txt

### Outcome
- Project foundation became ready for iterative development.
- Initial architecture supported clean expansion from baseline training to deployment.

---

## Step 2 - Python Environment Configuration (Completed)

### Actions
- Verified Python 3.11 runtime compatibility.
- Created isolated virtual environment for dependency control.
- Installed required libraries from requirements manifest.
- Validated import health for core stack (torch, torchvision, numpy, nibabel, scikit-learn, streamlit).
- Confirmed environment can execute preprocessing and training scripts without package conflicts.

### Files Added/Modified
- Modified: requirements.txt

### Outcome
- Reproducible local development environment established.
- Environment became stable for repeated model training and evaluation cycles.

---

## Step 3 - Dataset Structure Validation (Completed)

### Actions
- Validated BraTS 2020 abnormal dataset folder and modality layout.
- Validated OASIS normal dataset folder and NIfTI consistency.
- Confirmed representative volume shapes, dtypes, and intensity ranges.
- Identified OASIS singleton channel issue and confirmed squeeze strategy.
- Verified both datasets can be normalized under one preprocessing contract.

### Files Added/Modified
- None

### Outcome
- Data ingestion risks were reduced before model implementation.
- Dataset compatibility for mixed-source training was confirmed.

---

## Step 4 - MRI Preprocessing Pipeline Implementation (Completed)

## Step 4.1 - Volume-Level Preprocessing (Completed)

### Actions
- Implemented robust NIfTI loader with dimensionality checks.
- Added automatic 4D-to-3D squeeze when trailing singleton channel exists.
- Converted loaded volumes to float32 for consistent numeric behavior.
- Implemented non-zero voxel Z-score normalization.
- Added safeguards for edge cases such as zero standard deviation.

### Files Added/Modified
- Modified: src/preprocessing/volume_utils.py

### Outcome
- BraTS and OASIS volumes are now standardized before slice extraction.
- Downstream preprocessing receives consistent tensor-ready inputs.

---

## Step 4.2 - Slice-Level Processing (Completed)

### Actions
- Implemented axial slice extraction from 3D volumes.
- Implemented informative-slice filtering via non-zero pixel ratio threshold.
- Kept filtering threshold configurable to tune coverage vs noise trade-off.

### Files Added/Modified
- Modified: src/preprocessing/slice_utils.py

### Outcome
- Low-information slices were reduced before training.
- Signal quality of slice-level dataset improved.

---

## Step 4.3 - Dataset Construction Layer (Completed)

### Actions
- Implemented record builder that maps each slice to label and metadata.
- Added per-slice metadata fields: patient_id, slice_index, dataset source.
- Standardized record schema for later patient grouping and split safety.

### Files Added/Modified
- Modified: src/dataset/dataset_builder.py

### Outcome
- Unified slice-level dataset representation created for train/eval workflows.
- Patient-level aggregation support became straightforward.

---

## Step 4.4 - Dataset Record Serialization (Completed)

### Actions
- Added compressed serialization for dataset records using gzip.
- Implemented reusable build script for regenerating records after data updates.
- Shifted expensive preprocessing from training runtime to artifact build step.

### Files Added/Modified
- Added: src/utils/build_dataset_records.py
- Added: data/dataset_records.pkl.gz

### Outcome
- Training startup time decreased significantly.
- Dataset preparation became repeatable and versionable.

---

## Step 5 - Patient-Level Safe Dataset Splitting (Completed)

### Actions
- Implemented patient-level train/validation splitting.
- Added deterministic seed-driven shuffling.
- Added overlap checks to prevent patient leakage.

### Files Added/Modified
- Modified: src/dataset/split_utils.py

### Outcome
- Validation integrity improved by eliminating patient overlap.
- Reported metrics became more reliable for clinical interpretation.

---

## Step 6 - Input Transformation Layer (Completed)

### Actions
- Implemented patient-indexed neighbor lookup for 2.5D context construction.
- Added boundary-safe 2.5D stacking for first/last slices.
- Added resizing path for model-compatible spatial dimensions.
- Separated train-time and eval-time transform behavior.

### Files Added/Modified
- Modified: src/dataset/input_transforms.py

### Outcome
- Context-aware 2.5D model inputs are now generated consistently.
- Input pipeline supports deterministic evaluation and augmented training.

---

## Step 7 - Model Definition (Completed)

### Actions
- Implemented CNN classifier for binary MRI slice prediction.
- Built progressive feature extractor blocks (3->32->64->128->256 channels).
- Added dropout-regularized classifier head.
- Added Kaiming initialization for stable optimization.
- Added feature/gradient capture hooks for Grad-CAM explainability.
- Implemented model factory registry for architecture extensibility.
- Added tests for model construction, forward pass, output shape, and gradient hooks.

### Files Added/Modified
- Added/Modified: src/models/cnn_model.py
- Added/Modified: src/models/model_factory.py
- Added/Modified: tests/test_model_definition.py

### Outcome
- Core model is modular, test-validated, and explainability-ready.
- Architecture can be instantiated consistently across training/evaluation/app code.

---

## Step 8 - Training Pipeline Implementation (Completed)

## Step 8.1 - Dataset Loader Integration (Completed)

### Actions
- Implemented MRISliceDataset for record-based loading.
- Integrated transform pipeline into dataset retrieval.
- Verified channel-first tensor layout and label type correctness.
- Added DataLoader helper utilities for train/validation flows.

### Files Added/Modified
- Modified: src/dataset/mri_dataset.py
- Modified: tests/test_mri_dataset.py

### Outcome
- High-throughput batch loading became available for model training.
- Train/eval data paths became consistent and reusable.

---

## Step 8.2 - Training Loop Implementation (Completed)

### Actions
- Implemented train/validate epoch loops.
- Integrated criterion, optimizer, scheduler, and device management.
- Added metric history tracking (loss/accuracy/lr) across epochs.
- Added checkpoint save/load support in trainer abstraction.

### Files Added/Modified
- Modified: src/training/trainer.py

### Outcome
- End-to-end learning loop became reproducible and monitorable.
- Trainer abstraction reduced duplication in execution scripts.

---

## Step 8.3 - Training Execution Script (Completed)

### Actions
- Implemented CLI for hyperparameters and runtime settings.
- Integrated dataset loading, patient split, model creation, optimizer, scheduler, and trainer.
- Added deterministic seed configuration path.
- Added checkpoint naming/output directory flow.

### Files Added/Modified
- Modified: src/training/train_model.py

### Outcome
- Production-style training entry point established.
- Hyperparameter sweeps became possible without code edits.

---

## Step 9 - Model Evaluation Pipeline (Completed)

## Step 9.1 - Prediction Collection (Completed)

### Actions
- Implemented checkpoint-based predictor loading and inference execution.
- Collected true labels, predicted labels, and class probabilities.
- Enforced eval mode and no-grad inference behavior.
- Persisted outputs for downstream diagnostics and calibration.

### Files Added/Modified
- Modified: src/evaluation/predictor.py
- Modified: src/models/cnn_model.py
- Added: outputs/preds.npy
- Added: outputs/labels.npy
- Added: outputs/probs.npy

### Outcome
- Evaluation artifacts became available for metrics, error analysis, and calibration.
- Inference behavior became consistent with deployment expectations.

---

## Step 9.2 - Evaluation Metrics (Completed)

### Actions
- Implemented core binary metrics: accuracy, precision, recall, F1.
- Implemented confusion matrix computation.
- Structured metric outputs for report module consumption.

### Files Added/Modified
- Modified: src/evaluation/metrics.py

### Outcome
- Quantitative performance reporting became standardized.
- Single source of truth for evaluation metrics established.

---

## Step 9.3 - Evaluation Report Generation (Completed)

### Actions
- Added readable console report formatting.
- Added optional JSON report persistence.
- Added confusion-matrix shape validation guards.

### Files Added/Modified
- Modified: src/evaluation/report.py

### Outcome
- Evaluation outputs became both human-readable and machine-archivable.
- Reporting layer now supports debugging and milestone documentation.

---

## Step 10 - Training Execution & System Finalization (Completed)

## Step 10.1 - Full Dataset Generation (Completed)

### Actions
- Scanned BraTS and OASIS sources for eligible volumes.
- Built large slice-level dataset artifact from merged sources.
- Serialized dataset records to compressed artifact for reuse.

### Files Added/Modified
- Modified: src/utils/build_dataset_records.py
- Added: data/dataset_records.pkl.gz

### Outcome
- Full training-ready dataset artifact was generated successfully.
- Preprocessing overhead was moved out of repeated training loops.

---

## Step 10.2 - Model Training Execution (Completed)

### Actions
- Trained CNN using serialized records and patient-safe splits.
- Monitored training and validation behavior across epochs.
- Saved checkpoints for regression comparison and deployment use.
- Logged model artifacts for evaluation and calibration workflows.

### Files Added/Modified
- Modified: src/training/train_model.py
- Added/Modified: outputs/checkpoints/*.pth

### Outcome
- Baseline trained model family became available for downstream hardening.
- Checkpoints enabled controlled iteration over reliability improvements.

---

## Step 11 - Model Evaluation Deep Dive & Validation (Completed)

## Step 11.1 - Initial Evaluation & Error Observation (Completed)

### Actions
- Ran full validation evaluation pipeline.
- Observed model collapse to single-class prediction behavior.
- Confirmed issue through confusion matrix and class output distributions.

### Files Added/Modified
- Modified: src/evaluation/run_evaluation.py
- Modified: src/evaluation/predictor.py
- Added: outputs/preds.npy
- Added: outputs/labels.npy
- Added: outputs/probs.npy

### Outcome
- Critical bias/failure mode was identified early.
- Clear remediation direction established (class imbalance handling).

---

## Step 11.2 - Class Imbalance Handling (Completed)

### Actions
- Computed class frequencies from training split.
- Added weighted CrossEntropyLoss using class weights.
- Retrained and re-evaluated to validate fix.
- Verified prediction balance with confusion-matrix trends.

### Files Added/Modified
- Modified: src/training/train_model.py

### Outcome
- Single-class collapse issue was resolved.
- Balanced class discrimination significantly improved.

---

## Step 11.3 - Failure Case Analysis (Completed)

### Actions
- Implemented misclassification inspection utility.
- Reviewed residual error indices and class-specific patterns.
- Prioritized false-negative behavior for safety-oriented tuning.

### Files Added/Modified
- Modified: src/evaluation/error_analysis.py

### Outcome
- Error profile became explicit and actionable.
- Later calibration work was guided by observed failure modes.

---

## Step 11.4 - Patient-Level Prediction using Top-K Aggregation (Completed)

### Actions
- Grouped slice predictions by patient id.
- Implemented Top-K patient-level decision aggregation.
- Added robust decision components based on suspicious slice count/fraction.
- Computed patient-level evaluation metrics.

### Files Added/Modified
- Modified: src/aggregation/topk_aggregation.py
- Modified: src/evaluation/run_evaluation.py

### Outcome
- Patient-level decision layer became more robust to per-slice noise.
- Clinical-style diagnosis behavior improved relative to slice-only voting.

---

## Step 12 - ROC-AUC Evaluation (Completed)

### Actions
- Added ROC-AUC computation from positive-class probabilities.
- Updated evaluation plumbing to propagate probability arrays.
- Updated report output to include ROC-AUC consistently.

### Files Added/Modified
- Modified: src/evaluation/metrics.py
- Modified: src/evaluation/run_evaluation.py
- Modified: src/evaluation/report.py

### Outcome
- Threshold-independent discrimination metric was added to standard evaluation.
- Model quality can now be compared beyond fixed-threshold accuracy.

---

## Step 13 - Model Explainability using Grad-CAM (Completed)

## Step 13.1 - Initial Grad-CAM Implementation (Completed)

### Actions
- Implemented Grad-CAM activation map generation.
- Added composite visualization panel (MRI, heatmap, overlay).
- Integrated figure saving into evaluation flow.

### Files Added/Modified
- Modified: src/evaluation/gradcam.py
- Modified: src/evaluation/run_evaluation.py

### Outcome
- Baseline explainability workflow became operational.
- Qualitative inspection of model attention became possible.

---

## Step 13.2 - Clinical-Focus Preprocessing & Grad-CAM Refinement (Completed)

### Actions
- Reduced skull-edge shortcut learning risk via preprocessing updates.
- Corrected mask logic for normalized data handling.
- Improved Grad-CAM slice selection strategy and diagnostics.
- Hardened transform tests for deterministic evaluation behavior.

### Files Added/Modified
- Modified: src/dataset/input_transforms.py
- Modified: src/dataset/mri_dataset.py
- Modified: src/preprocessing/volume_utils.py
- Modified: src/training/train_model.py
- Modified: src/evaluation/gradcam.py
- Modified: src/evaluation/run_evaluation.py
- Modified: tests/test_input_transforms.py

### Outcome
- Explainability maps shifted toward more clinically relevant internal regions.
- Grad-CAM reliability improved across runs and checkpoints.

---

## Step 14 - Inference Wrapper for Deployment (Completed)

### Actions
- Built CLI/programmatic inference wrapper.
- Added latest-checkpoint auto-discovery.
- Implemented end-to-end inference from MRI file to patient-level result.
- Returned rich decision payload for app and evaluation consumption.

### Files Added/Modified
- Added/Modified: src/inference.py

### Outcome
- Reusable deployment-oriented inference interface delivered.
- Integration path for Streamlit and scripts became straightforward.

---

## Step 15 - Streamlit Application Development (Completed)

## Step 15.1 - Initial Version of Streamlit App (Completed)

### Actions
- Implemented file upload, slice navigation, and prediction display.
- Integrated Grad-CAM visualization in UI.
- Added early stability fixes for rendering and transform handling.
- Added NIfTI validation and temporary-file lifecycle handling.

### Files Added/Modified
- Modified: app.py
- Modified: requirements.txt

### Outcome
- End-to-end interactive diagnostic demo became available.
- Baseline app stabilized for iterative UX improvements.

---

## Step 15.2 - Streamlit UI Refinement and Export Workflow (Completed)

### Actions
- Improved visual hierarchy and diagnostic flow.
- Added probability graph and export options (PNG/PDF).
- Refined theme/layout behavior for reliability.
- Enhanced reporting bundle behavior for easier sharing.

### Files Added/Modified
- Modified: app.py
- Modified: requirements.txt
- Modified: PROJECT_LOG.md

### Outcome
- App became more presentation-ready and clinician-friendly.
- Reporting and export workflow significantly improved.

---

## Step 16 - Model Reliability Revalidation and Training Hardening (Completed)

## Step 16.1 - OASIS False-Positive Investigation (Completed)

### Actions
- Ran targeted OASIS checks on latest checkpoints.
- Added single-case diagnostics utility.
- Added dataset composition analysis utility.
- Quantified suspicious-slice behavior on healthy controls.

### Files Added/Modified
- Added: test_oasis_sample.py
- Added: analyze_dataset.py

### Outcome
- Healthy-control false-positive behavior became measurable.
- Reliability baseline established for hardening iterations.

---

## Step 16.2 - Training/Evaluation Pipeline Hardening (Completed)

### Actions
- Added balanced-validation option.
- Added sensitivity/specificity metric reporting.
- Updated train/eval flow to expose imbalance effects earlier.

### Files Added/Modified
- Modified: src/dataset/split_utils.py
- Modified: src/training/train_model.py
- Modified: src/evaluation/metrics.py
- Modified: src/evaluation/report.py

### Outcome
- Validation now reflects clinically relevant behavior more clearly.
- Monitoring quality improved beyond raw accuracy.

---

## Step 16.3 - Short-Cycle Retraining and Regression Checks (Completed)

### Actions
- Executed short-cycle retraining runs (5/10/20 epochs).
- Re-ran OASIS checks for each cycle.
- Compared drift across checkpoints before long-run retraining decisions.

### Files Added/Modified
- Modified: src/training/train_model.py
- Modified: test_oasis_sample.py

### Outcome
- Slice-level behavior improved incrementally.
- Aggregation sensitivity remained key residual reliability issue.

---

## Step 16.4 - Aggregation Calibration, Unification, and External OASIS Verification (Completed)

### Actions
- Unified robust patient-level decision logic across evaluation, inference, and app.
- Added aggregation calibration grid-search pipeline.
- Added external OASIS batch verification script.
- Persisted calibrated aggregation config artifact for reuse.

### Files Added/Modified
- Modified: src/aggregation/topk_aggregation.py
- Modified: src/inference.py
- Modified: src/evaluation/run_evaluation.py
- Modified: app.py
- Modified: test_oasis_sample.py
- Added: src/evaluation/calibrate_aggregation.py
- Added: src/evaluation/oasis_batch_check.py
- Added: outputs/calibration/aggregation_calibration.json

### Outcome
- Patient-level behavior became consistent across runtime paths.
- External healthy-control performance became trackable with explicit configuration artifacts.

---

## Step 17 - Slice Threshold Calibration Integration (Completed)

### Actions
- Added slice-threshold calibration utility module.
- Integrated threshold artifact generation into evaluation runtime.
- Wired calibrated threshold usage into app slice-level decision path.
- Added safe fallback to default threshold when artifact is missing.

### Files Added/Modified
- Added: src/evaluation/threshold_calibration.py
- Modified: src/evaluation/run_evaluation.py
- Modified: app.py
- Added: outputs/calibration/slice_threshold_calibration.json

### Outcome
- Slice-level decision threshold is now data-calibrated.
- Threshold behavior remains reusable and consistent after retraining/data updates.

---

## Step 18 - Multimodality Generalization and Data-Root Migration (Completed)

## Step 18.1 - Modality/Scanner/Geometry-Aware Pipeline Upgrade (Completed)

### Actions
- Added automatic modality detection from filename/header context for FLAIR, T1, T1c, T2, and PD.
- Added scanner-aware intensity normalization utilities with robust scaling and optional histogram standardization hooks.
- Added orientation normalization to canonical RAS and voxel-spacing resampling to a common grid.
- Extended dataset record construction to store modality, field strength, and voxel spacing metadata.
- Added dataset adapters to standardize data ingestion contracts across BraTS, OASIS, and IXI.
- Expanded input construction from fixed 2.5D to configurable single-channel, legacy 2.5D, and multimodal channel stacking.
- Added modality dropout support for multimodal training robustness.
- Updated model factory and CNN definition to support variable input channel counts.

### Files Added/Modified
- Added: src/preprocessing/modality_detection.py
- Added: src/preprocessing/scanner_normalization.py
- Added: src/preprocessing/resampling.py
- Added: src/dataset/dataset_adapter.py
- Modified: src/preprocessing/volume_utils.py
- Modified: src/dataset/dataset_builder.py
- Modified: src/dataset/input_transforms.py
- Modified: src/dataset/mri_dataset.py
- Modified: src/models/cnn_model.py
- Modified: src/models/model_factory.py
- Modified: src/training/train_model.py
- Modified: src/inference.py
- Modified: app.py

### Outcome
- Core pipeline became modality-flexible and scanner-aware without removing legacy behavior.
- Training/inference paths now accept variable channel configurations while remaining backward compatible with existing checkpoints and workflows.

---

## Step 18.2 - Dataset Root Migration and Script Hardening (Completed)

### Actions
- Replaced machine-specific absolute dataset paths with project-relative paths under data/raw.
- Updated dataset record build flow to scan data/raw/brats, data/raw/oasis, and optional data/raw/ixi.
- Updated inspection and OASIS diagnostic scripts to discover files dynamically from project data roots.
- Updated legacy utility tests to use robust file discovery instead of hardcoded local paths.

### Files Added/Modified
- Modified: src/utils/build_dataset_records.py
- Modified: inspect_volume.py
- Modified: test_oasis_sample.py
- Modified: src/evaluation/oasis_batch_check.py
- Modified: tests/test_volume_utils.py
- Modified: tests/test_split_utils.py
- Modified: tests/test_slice_utils.py
- Modified: tests/test_dataset_builder.py

### Outcome
- Project portability improved across machines and environments.
- Dataset tooling now works after moving raw datasets into the repository-managed directory layout.

---

## Step 18.3 - Validation, Regression Fixes, and BraTS2020 All-Modality Smoke Test (Completed)

### Actions
- Added new modality pipeline tests and extended existing model/dataset tests for variable-channel behavior.
- Fixed dataset tensor channel-order regression for single-channel and multimodal conversions.
- Updated transform/skull-strip tests to match realistic foreground/background assumptions.
- Ran targeted regression suite and resolved failures until all tests passed.
- Executed BraTS2020 modality coverage + preprocessing + multimodal loader smoke test on data/raw/brats.

### Files Added/Modified
- Added: tests/test_modality_pipeline.py
- Modified: tests/test_model_definition.py
- Modified: tests/test_mri_dataset.py
- Modified: tests/test_input_transforms.py
- Modified: src/dataset/mri_dataset.py

### Outcome
- Targeted regression suite passed with full success (19/19).
- BraTS2020 smoke validation confirmed modality availability and multimodal channel construction:
- FLAIR/T1/T1c/T2 were each detected for all discovered patients.
- Multimodal sample tensors were generated correctly with 4-channel input shape.

---

## Step 18.4 - Open-Set Modality Acceptance and Automatic Detection Expansion (Completed)

### Actions
- Expanded modality detection from fixed-label mapping to open-set inference so non-canonical MRI modalities are automatically tagged instead of defaulting to unknown.
- Added custom modality token support for common advanced sequences (for example ADC, SWI, DWI/DTI, ASL, TOF) while retaining canonical mapping for FLAIR/T1/T1c/T2/PD.
- Updated detection priority so true modality tokens are selected ahead of generic filename tokens.
- Removed modality-drop filtering in dataset adapter flow so non-canonical modalities are still ingested and tracked.
- Updated dataset record construction to always preserve detected modality token in metadata for downstream training and auditability.
- Added focused tests and runtime checks to confirm custom modality inference and record preservation behavior.

### Files Added/Modified
- Modified: src/preprocessing/modality_detection.py
- Modified: src/dataset/dataset_adapter.py
- Modified: src/dataset/dataset_builder.py
- Modified: tests/test_modality_pipeline.py

### Outcome
- Pipeline now accepts arbitrary MRI modality tokens and automatically labels them during ingestion.
- Unknown/non-standard sequences no longer get silently dropped from dataset scanning.
- Canonical modalities remain backward compatible while open-set modalities are preserved for flexible mixed-dataset experimentation.

---

## Step 19 - Modality-Aware Slice Threshold Calibration (Completed)

### Actions
- Extended slice-threshold calibration from a single global threshold to modality-aware calibration with global fallback.
- Added calibration utilities to compute per-modality thresholds with minimum sample safeguards and constraint-aware optimization.
- Added artifact save/load flow for modality-threshold configuration.
- Integrated per-modality threshold generation into evaluation so calibration artifacts are produced during evaluation runs.
- Updated inference/runtime threshold selection to use detected modality-specific threshold first, then global fallback.
- Updated Streamlit threshold path to use modality-aware threshold selection for slice-level decisioning.
- Added focused tests for modality-threshold payload creation and threshold lookup precedence.

### Files Added/Modified
- Modified: src/evaluation/threshold_calibration.py
- Modified: src/evaluation/run_evaluation.py
- Modified: src/inference.py
- Modified: app.py
- Added: tests/test_threshold_calibration_modality.py
- Added/Modified: outputs/calibration/modality_threshold_calibration.json

### Outcome
- Thresholding logic is now modality-aware in architecture, with safe fallbacks when modality-specific calibration is unavailable.
- Inference outputs now expose modality and slice threshold used, improving auditability and debugging.
- Calibration framework is ready to produce distinct thresholds per modality once dataset records include modality metadata at scale.

---

## Step 20 - IXI External Validation and Multimodal Runtime Hardening (Completed)

### Actions
- Verified IXI data availability and identified archive-only state in data/raw/IXI.
- Extracted IXI modality archives (T1/T2/PD/MRA) and validated NIfTI discovery counts.
- Ran IXI compatibility smoke checks through adapter scan, dataset record build, dataset tensor generation, and one-file inference.
- Identified multimodal stacking failure caused by per-modality in-plane shape mismatch within a patient.
- Added shape-alignment logic before multimodal channel stacking to robustly support mixed-shape modality inputs.
- Re-ran focused transform/dataset/modality tests and IXI end-to-end smoke checks after the fix.

### Files Added/Modified
- Modified: src/dataset/input_transforms.py
- Modified: tests/test_input_transforms.py
- Modified: tests/test_mri_dataset.py
- Modified: tests/test_modality_pipeline.py
- Added/Modified: data/raw/IXI/*.nii.gz

### Outcome
- IXI files are now usable end-to-end by the current pipeline after archive extraction and stacking fix.
- Multimodal channel construction became resilient to modality-specific slice shape differences.
- External validation path is now stronger for cross-dataset robustness checks.

---

## Step 21 - Sequential Reliability Fix Validation and 4-Dataset Evaluation (Completed)

## Step 21.1 - Sequential Step-Wise Reliability Fixing (Completed)

### Actions
- Applied reliability fixes in strict sequence with evaluation after each step.
- Step 1: Updated intensity normalization to percentile-clipped Z-score on non-zero voxels.
- Rebuilt dataset records with the updated preprocessing path.
- Ran quick probes and evaluations to validate specificity behavior before moving to the next fix.
- Step 2: Enabled balanced training sampling using WeightedRandomSampler for class-balanced slice exposure.

### Files Added/Modified
- Modified: src/preprocessing/volume_utils.py
- Modified: src/training/train_model.py

### Outcome
- Sequential experimentation workflow was preserved without mixing fixes.
- Step 1 improved healthy-domain behavior but did not fully satisfy all patient-level targets alone.
- Step 2 balancing logic was integrated and validated through a new short-cycle training run.

---

## Step 21.2 - Step 2 Two-Epoch Training Run (Completed)

### Actions
- Executed a two-epoch training run using balanced sampling.
- Trained with a fast subset fraction for quick-turn validation.
- Saved a dedicated checkpoint for direct comparison against Step 1 probes.

### Files Added/Modified
- Added/Modified: outputs/checkpoints/step2_balanced_2ep

### Outcome
- Training completed successfully with stable train/validation progression.
- Produced a reproducible checkpoint for external healthy/tumor verification.

---

## Step 21.3 - Full Four-Dataset Evaluation (BraTS2020, BraTS2021, OASIS, IXI) (Completed)

### Actions
- Ran dataset-specific evaluation scripts against the Step 2 checkpoint for:
	- BraTS2020
	- BraTS2021
	- OASIS
	- IXI
- Generated per-dataset JSON reports with slice-level and patient-level metrics.
- Aggregated confusion matrices across all four datasets for overall model-level summary.

### Files Added/Modified
- Added/Modified: scripts/evaluate_ixi_model.py
- Added/Modified: outputs/reports/step2_2ep_brats2020.json
- Added/Modified: outputs/reports/step2_2ep_brats2021.json
- Added/Modified: outputs/reports/step2_2ep_oasis.json
- Added/Modified: outputs/reports/step2_2ep_ixi.json

### Outcome
- Patient-level combined confusion matrix across 4 datasets:
	- [[776, 2], [0, 1620]]
- Patient-level overall metrics:
	- Accuracy: 0.9992
	- Sensitivity: 1.0000
	- Specificity: 0.9974
- Slice-level combined confusion matrix across 4 datasets:
	- [[308491, 11503], [297, 792703]]
- Slice-level overall metrics:
	- Accuracy: 0.9894
	- Sensitivity: 0.9996
	- Specificity: 0.9641

---
