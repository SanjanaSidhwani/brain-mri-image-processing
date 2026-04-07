# Brain MRI AI Decision Support - Project Log

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
