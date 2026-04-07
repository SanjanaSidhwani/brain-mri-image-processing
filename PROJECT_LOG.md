# Brain MRI AI Decision Support – Project Log

---

## Step 1 – Project Structure Initialization (Completed)

### Actions
- Created independent orphan branch
- Initialized repository structure
- Established modular folder layout aligned with pipeline architecture
- Added:
  - README.md
  - PROJECT_LOG.md
  - requirements.txt
- Reserved module space for future Top-K patient-level aggregation

### Outcome
Project foundation initialized and ready for environment setup.

---

## Step 2 – Python Environment Configuration (Completed)

### Actions
- Verified Python 3.11
- Created virtual environment
- Installed required dependencies using requirements.txt
- Validated core library imports
- Committed environment setup

### Outcome
Stable and reproducible development environment configured.

---

## Step 3 – Dataset Structure Validation (Completed)

### Actions
- Downloaded BraTS 2020 dataset (Abnormal class)
- Verified patient-wise folder structure
- Confirmed availability of FLAIR modality and segmentation masks
- Loaded sample BraTS FLAIR volume using NiBabel
- Observed volume shape: (240, 240, 155)
- Verified data type: float64
- Confirmed intensity range: 0 – 625

- Downloaded OASIS dataset (Normal class)
- Verified NIfTI volume structure
- Loaded sample OASIS volume using NiBabel
- Observed volume shape: (176, 208, 176, 1)
- Verified data type: float64
- Confirmed intensity range: 0 – 2870
- Identified extra singleton dimension requiring squeeze to 3D

### Outcome
Both datasets successfully validated and confirmed compatible with the planned 3D MRI preprocessing pipeline.

## Step 4 – MRI Preprocessing Pipeline Implementation (Completed)

---

## Step 4.1 – Volume-Level Preprocessing (Completed)

### Actions
- Implemented NIfTI loading utility using NiBabel
- Added automatic 4D-to-3D squeeze handling
- Converted volume dtype from float64 to float32
- Implemented Z-score normalization on non-zero voxels only
- Added defensive handling for zero standard deviation cases
- Validated functionality on BraTS and OASIS sample volumes

### Outcome
Volume-level preprocessing successfully unified both datasets with consistent dimensionality and standardized intensity distributions (mean ≈ 0, std ≈ 1).

## Step 4.2 – Slice-Level Processing (Completed)

### Actions
- Implemented axial slice extraction from 3D MRI volumes
- Developed non-zero pixel ratio based empty slice filtering
- Maintained clean abstraction layering between volume and slice utilities
- Ensured dataset-agnostic functionality
- Validated filtering on BraTS and OASIS sample volumes

### Outcome
Successfully converted 3D MRI volumes into meaningful 2D axial slices, reducing noise and eliminating empty background slices prior to dataset construction.

## Step 4.3 – Dataset Construction Layer (Completed)

### Actions
- Implemented slice-to-label mapping mechanism
- Designed structured metadata format for each slice
- Added patient ID and dataset source tracking
- Maintained clean separation from preprocessing logic
- Validated dataset construction on BraTS and OASIS samples

### Outcome
Successfully constructed structured slice-level dataset records with complete metadata, enabling safe training, validation splitting, and future patient-level aggregation.

### Enhancement – Dataset Record Serialization

### Actions
- Implemented dataset record serialization for persistent storage
- Created utility script `src/utils/build_dataset_records.py`
- Added support for compressed dataset storage using gzip
- Enabled reuse of preconstructed slice records across multiple training runs
- Eliminated need to rebuild dataset pipeline during every training execution

### Outcome
Dataset slice records can now be serialized and stored on disk as `dataset_records.pkl.gz`.  
This significantly reduces training startup time by allowing the training pipeline to directly load preprocessed dataset records instead of reconstructing them from raw MRI volumes.

## Step 5 – Patient-Level Safe Dataset Splitting (Completed)

### Actions
- Implemented patient-level dataset splitting mechanism
- Ensured all slices from a patient belong to the same split
- Added reproducible shuffling using fixed random seed
- Prevented slice-level data leakage

### Outcome
Dataset is now safely split into training and validation sets at the patient level, ensuring reliable model evaluation and preventing leakage.

## Step 6 – Input Transformation Layer (Completed)

### Actions
- Implemented patient-grouped indexing for neighbor lookup
- Developed 2.5D slice stacking mechanism
- Handled boundary conditions with zero-padding
- Added resizing utility for CNN compatibility
- Validated final input shape consistency

### Outcome
Successfully transformed slice-level dataset into model-ready 2.5D inputs with standardized spatial dimensions.

## Step 7 – Model Definition (Completed)

### Actions
- Implemented CNN architecture for MRI slice classification using PyTorch
- Created modular model definition inside src/models/cnn_model.py
- Designed stacked convolution blocks consisting of Conv2D, BatchNorm, ReLU, and MaxPooling
- Implemented progressive feature extraction layers (3 → 32 → 64 → 128 → 256 channels)
- Added Adaptive Average Pooling for spatial feature aggregation
- Implemented classifier head with fully connected layers and dropout regularization
- Applied Kaiming Normal initialization for convolution and linear layers
- Added feature map storage and gradient hook support for future Grad-CAM explainability
- Implemented centralized model creation system inside src/models/model_factory.py
- Added model registry to support extensible architecture management
- Implemented model configuration interface for architecture parameters
- Created unit tests to validate model creation, forward pass, and output shape
- Verified compatibility with 2.5D stacked MRI inputs (3 × 224 × 224)

### Outcome
Successfully implemented the CNN architecture responsible for MRI slice classification.  
The model definition is modular, Grad-CAM compatible, and validated through unit testing, enabling seamless integration with the upcoming training pipeline.

## Step 8 – Training Pipeline Implementation (Completed)

---

## Step 8.1 – Dataset Loader Integration (Completed)

### Actions
- Implemented PyTorch Dataset class for MRI slice records
- Integrated dataset records with input transformation pipeline
- Ensured image tensors follow channel-first format (3 × 224 × 224)
- Implemented label retrieval for slice-level classification
- Added DataLoader configuration for batch-based training
- Validated dataset loading using sample batches

### Outcome
Successfully implemented the dataset loader responsible for feeding MRI slice data into the training pipeline. The loader integrates dataset records, input transformations, and PyTorch batching utilities, enabling efficient data delivery to the CNN model during training.

## Step 8.2 – Training Loop Implementation (Completed)

### Actions
- Implemented modular training loop inside src/training/trainer.py
- Integrated model, optimizer, and loss function into the training workflow
- Implemented forward pass, loss computation, and backpropagation steps
- Added optimizer updates for model parameter learning
- Implemented validation loop for performance monitoring
- Added device management for CPU/GPU compatibility
- Implemented training and validation loss tracking across epochs

### Outcome
Successfully implemented the core training loop responsible for model learning. The trainer module now orchestrates forward passes, loss computation, gradient updates, and validation evaluation, enabling the CNN model to learn discriminative patterns from MRI slices.

## Step 8.3 – Training Execution Script (Completed)

### Actions
- Created centralized training execution script at src/training/train_model.py
- Implemented command-line argument parsing for flexible training configuration
- Added comprehensive hyperparameter support (learning rate, batch size, epochs, dropout, etc.)
- Integrated model initialization via model factory
- Configured loss function (CrossEntropyLoss) for binary classification
- Implemented optimizer setup with Adam and optional weight decay
- Added optional learning rate scheduler (StepLR) support
- Integrated dataset record loading interface with placeholder implementation
- Implemented patient-level dataset splitting via split_utils
- Created PyTorch dataset instances using MRISliceDataset
- Configured DataLoader creation for batch-based training
- Initialized Trainer class with all required components
- Implemented full training execution workflow
- Added model checkpoint saving functionality with metadata
- Implemented comprehensive logging and progress reporting
- Added dataset statistics and model information display
- Configured reproducible training via random seed setting
- Added automatic device detection (CPU/GPU) with manual override option

### Outcome
Successfully implemented the complete training execution script serving as the central entry point for the training pipeline. The script orchestrates all training components from data loading through model checkpoint saving, providing a production-ready interface for training the MRI classification system with configurable hyperparameters via command-line arguments.

### Step 9 – Model Evaluation Pipeline (Completed)

## Step 9.1 – Prediction Collection (Completed)

### Actions
- Implemented prediction collection module in `src/evaluation/predictor.py`
- Added functionality to load trained model checkpoints
- Implemented inference pipeline using PyTorch evaluation mode
- Disabled gradient computation during prediction using `torch.no_grad()`
- Collected true labels, predicted labels, and prediction probabilities
- Structured prediction outputs for downstream evaluation metrics
- Updated CNN model (`src/models/cnn_model.py`) to conditionally register gradient hooks only when gradients are enabled to ensure compatibility with inference mode

### Outcome
Successfully implemented the prediction collection module for the evaluation pipeline. The system can now run inference on validation datasets and collect prediction outputs required for performance metric computation while maintaining compatibility with Grad-CAM functionality during training.

## Step 9.2 – Evaluation Metrics (Completed)

### Actions
- Implemented classification metrics module in `src/evaluation/metrics.py`
- Computed accuracy, precision, recall, and F1-score for binary classification
- Implemented confusion matrix computation
- Structured metric outputs for evaluation reporting

### Outcome
The system can now compute quantitative performance metrics from prediction outputs, enabling objective evaluation of the MRI classification model.

## Step 9.3 – Evaluation Report Generation (Completed)

### Actions
- Implemented evaluation report module in `src/evaluation/report.py`
- Added formatted console output for evaluation metrics (accuracy, precision, recall, F1-score)
- Implemented readable confusion matrix display for binary classification
- Added support for saving evaluation results to JSON format
- Ensured safe handling of missing metric values using default fallbacks
- Added validation for confusion matrix shape to prevent runtime errors

### Outcome
Successfully implemented a structured evaluation reporting system. The model’s performance can now be clearly presented through formatted outputs and saved reports, making results suitable for analysis, debugging, and project demonstration.

### Step 10 – Training Execution & System Finalization (Completed)

### Step 10.1 – Full Dataset Generation (Completed)

### Actions
- Generated dataset records from BRATS and OASIS datasets
- Implemented dataset scanning for both folder-based and file-based structures
- Built slice-level dataset (~70k slices)
- Serialized dataset using gzip for reuse
- Files Modified / Added
  src/utils/build_dataset_records.py
  data/dataset_records.pkl.gz

### Outcome
Dataset successfully constructed and stored for efficient reuse during training.

### Step 10.2 – Model Training Execution (Completed)

### Actions
- Executed training pipeline using src/training/train_model.py
- Loaded serialized dataset records (~70k slices)
- Performed patient-level train/validation split
- Initialized MRISliceDataset for training and validation
- Created DataLoader instances for batch processing
- Initialized CNN model using model factory
- Moved model to GPU (RTX 4050)
- Initialized optimizer (Adam) and loss function (CrossEntropyLoss)
- Ran full training for multiple epochs
- Monitored training and validation performance
- Saved trained model checkpoint

### Outcome
Training pipeline successfully executed on the full dataset using GPU acceleration. The model achieved high accuracy on both training and validation sets, and a trained model checkpoint was generated for evaluation and future use.

## Step 11 – Model Evaluation Deep Dive & Validation (Completed)

---

## Step 11.1 – Initial Evaluation & Error Observation (Completed)

### Actions
- Implemented evaluation execution script at src/evaluation/run_evaluation.py
- Integrated Predictor for model inference on validation dataset
- Loaded trained model checkpoint from outputs/checkpoints/
- Performed inference on validation dataset using DataLoader
- Collected predicted labels, true labels, and probabilities
- Computed classification metrics using compute_classification_metrics
- Generated evaluation report with accuracy, precision, recall, F1-score, and confusion matrix
- Saved prediction outputs (preds.npy, labels.npy, probs.npy) for further analysis

- Files Modified / Added
  src/evaluation/run_evaluation.py
  src/evaluation/predictor.py
  outputs/preds.npy
  outputs/labels.npy
  outputs/probs.npy

### Outcome
Evaluation pipeline successfully executed end-to-end on the validation dataset.  

However, results revealed abnormal model behavior:
- Accuracy: 0.5833  
- Precision: 0.5833  
- Recall: 1.0000  
- F1 Score: 0.7368  

Confusion matrix showed that the model predicted only a single class (class 1) for all samples, indicating a critical issue in model behavior despite previously observed high training accuracy.  

This suggests potential problems such as class imbalance, data leakage during training, or incorrect evaluation setup, requiring further debugging and validation.

## Step 11.2 – Class Imbalance Handling (Completed)

### Actions
- Identified model bias during evaluation (predicting only one class)
- Analyzed dataset class distribution across training and validation sets
- Implemented class weighting strategy in loss function
- Computed class weights dynamically using training dataset distribution
- Updated CrossEntropyLoss to include class weights
- Retrained model using weighted loss function
- Re-evaluated model performance on validation dataset

- Files Modified / Added
  src/training/train_model.py

### Outcome
Class imbalance issue successfully resolved.  

Model no longer predicts a single class and correctly distinguishes between both classes.  

Improved evaluation results:
- Accuracy: 0.9996  
- Precision: 1.0000  
- Recall: 0.9993  
- F1 Score: 0.9996  

Confusion matrix confirms balanced predictions with minimal misclassification, indicating reliable model performance.

## Step 11.3 – Failure Case Analysis (Completed)

### Actions
- Implemented error inspection in src/evaluation/error_analysis.py
- Identified misclassified samples using prediction vs ground truth comparison
- Extracted indices of incorrect predictions
- Printed sample misclassifications including predicted and actual labels
- Analyzed error patterns in model predictions

- Files Modified / Added
  src/evaluation/error_analysis.py

### Outcome
Model errors were minimal, with only a small number of misclassified samples observed.  

Analysis revealed that all errors were false negatives, where the model predicted the normal class (0) for abnormal cases (1).  

This indicates that while overall model accuracy is very high, there is a slight tendency to miss certain tumor cases.  

The insight highlights an important limitation of the model and provides direction for further improvements in sensitivity and reliability.

## Step 11.4 – Patient-Level Prediction using Top-K Aggregation (Completed)

### Actions
- Implemented Top-K aggregation strategy in src/aggregation/topk_aggregation.py
- Grouped slice-level predictions by patient ID
- Selected top-K highest confidence slices per patient
- Computed patient-level prediction based on aggregated slice probabilities
- Implemented patient-level ground truth extraction
- Integrated aggregation into evaluation pipeline
- Computed evaluation metrics at patient level

- Files Modified / Added
  src/aggregation/topk_aggregation.py
  src/evaluation/run_evaluation.py

### Outcome
Successfully extended the system from slice-level classification to patient-level diagnosis.  

Patient-level evaluation achieved perfect performance:
- Accuracy: 1.0000  
- Precision: 1.0000  
- Recall: 1.0000  
- F1 Score: 1.0000  

Top-K aggregation improved robustness by compensating for occasional slice-level errors, ensuring reliable patient-level predictions.

## Step 12 – ROC-AUC Evaluation

---
## Step 12.1 – ROC-AUC Metric Integration (Completed)

### Actions
- Extended evaluation metrics to include ROC-AUC score
- Imported roc_auc_score from sklearn.metrics
- Extracted class-wise probabilities from model outputs
- Computed ROC-AUC using probability of positive class
- Integrated ROC-AUC computation into evaluation pipeline
- Updated evaluation execution to pass probability outputs

- Files Modified / Added
  src/evaluation/metrics.py
  src/evaluation/run_evaluation.py
  src/evaluation/report.py

### Outcome
Successfully integrated ROC-AUC metric into evaluation pipeline.  

Model achieved near-perfect ROC-AUC, indicating strong separability between normal and abnormal classes across all classification thresholds.  

This confirms that the model is not only accurate but also highly confident and robust in distinguishing between classes.

## Step 13 – Model Explainability using Grad-CAM (Completed)

### Step 13.1 – Initial Grad-CAM Implementation (Completed)

### Actions
- Implemented Grad-CAM module in src/evaluation/gradcam.py
- Utilized stored convolutional feature maps and gradients from CNN model
- Generated class-specific activation maps using weighted feature maps
- Integrated Grad-CAM into evaluation pipeline (run_evaluation.py)
- Visualized results using center MRI slice from 2.5D input
- Created combined panel output showing MRI slice, heatmap, and overlay
- Saved visualization outputs with auto-increment naming for tracking

- Files Modified / Added
  src/evaluation/gradcam.py
  src/evaluation/run_evaluation.py

### Outcome
Successfully generated Grad-CAM visualizations for model predictions.

The heatmaps highlight regions of high activation, with noticeable emphasis on structural boundaries (e.g., brain edges) along with internal regions. This indicates that the model is utilizing both global structural features and localized patterns for classification.

The current visualization reflects the model's learned feature representations but requires further refinement to improve localization of clinically relevant regions (e.g., tumor areas).

### Step 13.2 – Clinical-Focus Preprocessing & Grad-CAM Refinement (Completed)

### Actions
- Refactored preprocessing pipeline to reduce shortcut learning from skull boundaries:
  - Center crop + constrained augmentation for training
  - Deterministic center crop + resize for validation
  - Added transform parameter validation
- Implemented skull-strip utility suppressing outer skull-edge cues in 2.5D inputs
  - Corrected mask logic for z-score normalized slices (changed from > 0 to != 0)
- Improved Grad-CAM reliability:
  - Fixed class handling and removed duplicated gradient normalization
  - Auto-load latest checkpoint from outputs/checkpoints
  - Selected Grad-CAM slice using highest tumor probability instead of max pixels
  - Added selection diagnostics in evaluation logs
- Hardened transform tests with deterministic, dataset-independent checks

- Files Modified / Added
  src/dataset/input_transforms.py
  src/dataset/mri_dataset.py
  src/preprocessing/volume_utils.py
  src/training/train_model.py
  src/evaluation/gradcam.py
  src/evaluation/run_evaluation.py
  tests/test_input_transforms.py

### Outcome
Grad-CAM quality improved from edge-dominant activations toward more internal, clinically relevant regions. The explainability workflow is now stable and driven by clinical probability signals for ongoing tuning and model validation.

## Step 14 – Inference Wrapper for Deployment (Completed)

### Actions
- Created src/inference.py module for model prediction interface
  - Implemented predict_on_mri() function for end-to-end inference
  - Integrated NIfTI loading, preprocessing (normalization, skull stripping, cropping)
  - Automatic latest checkpoint detection from outputs/checkpoints
  - Multi-slice voting mechanism (>50% tumor slices = positive prediction)
  - Returns prediction, confidence, tumor probability, and slice statistics
  - Auto device selection (GPU if available, CPU fallback)
  - Configurable preprocessing parameters (crop size, transforms, skull strip flag)
- Implemented get_latest_checkpoint() utility for checkpoint auto-loading
- Added command-line interface for standalone inference testing
  - Usage: python src/inference.py path/to/patient.nii --crop_size 180
  - Provides formatted results output with prediction and confidence metrics
- Designed Streamlit-compatible function signature
  - Takes file path → Returns simple dict with results
  - No direct dependency on Streamlit (clean separation)

- Files Modified / Added
  src/inference.py

### Outcome
Successfully implemented production-ready inference wrapper decoupling analysis logic from UI implementation. The module provides a clean, reusable interface for Streamlit integration without tight coupling. Supports both CLI and programmatic usage. Ready for Streamlit application development.

## Step 15 – Streamlit Application Development (Completed)

---

## Step 15.1 – Initial Version of Streamlit App (Completed)

### Actions
- Built initial Streamlit app in `app.py` for MRI upload and interactive slice navigation
- Added NIfTI upload support (`.nii`, `.nii.gz`) with file validation and patient metadata display
- Integrated cached model loading, slice-level prediction, and confidence display
- Added optional Grad-CAM with 3-panel output: Original, Heatmap, Overlay
- Applied stability fixes for volume-to-slice skull stripping, transform tensor input, overlay size alignment, and Streamlit image rendering
- Added Streamlit dependency to requirements

- Files Modified / Added
  app.py
  requirements.txt

### Outcome
Delivered a stable Streamlit baseline with upload, prediction, and Grad-CAM explainability. Ready for creative UI enhancements in the next iteration.

## Step 15.2 – Streamlit UI Refinement and Export Workflow (Completed)

### Actions
- Rebranded UI to **SYNAPSE X** and updated header hierarchy for a cleaner product identity.
- Removed emoji-based visual elements and standardized professional text-first presentation.
- Applied typography upgrade: **Playfair Display** (headings) + **Source Sans 3** (body).
- Restructured page flow for clearer clinical reading order:
  1. Patient Information
  2. Patient Diagnosis
  3. Slice Selection
  4. Slice-Level Analysis
  5. Explainability / Probability sections
- Improved Patient Information readability:
  - Patient ID placed on a dedicated full-width line.
  - Volume Shape and Total Slices moved to the next row.
- Implemented dual-level diagnosis UX:
  - Patient-level diagnosis card/metrics as primary output.
  - Slice-level diagnosis as secondary output.
- Added whole-volume probability inference utilities:
  - Slice-wise probability scan across the uploaded volume.
  - Highest tumor-probability slice identification.
- Added probability distribution visualization across slices and updated to blue-theme styling.
- Fixed Grad-CAM panel consistency:
  - Uniform image sizing across Original / Heatmap / Overlay panels.
  - Improved layout alignment and section spacing.
- Removed deprecated Streamlit image sizing usage and migrated to current supported arguments.
- Implemented export workflow improvements:
  - Sidebar **Exports** section for all download actions.
  - Grad-CAM panel PNG export.
  - Probability graph PNG export.
  - Full PDF report export.
- Enhanced PDF report content so it acts like a printable bundle:
  - Includes patient/slice summaries.
  - Embeds Grad-CAM panel and probability graph in report body.
- Performed theme reliability pass:
  - Removed unstable in-app theme switch.
  - Restored native Streamlit Light/Dark control from top-right menu.
  - Reduced custom CSS scope to avoid breaking widget behavior and image interactions.

### Files Modified / Added
- app.py
- requirements.txt
- PROJECT_LOG.md

### Outcome
Step 15.2 delivered a production-ready Streamlit UX pass over the baseline app: improved visual hierarchy, clearer diagnosis communication, robust explainability presentation, and complete export/report functionality while preserving compatibility with native Streamlit theming behavior.

## Step 16 – Model Reliability Revalidation and Training Hardening (Completed)

---

## Step 16.1 – OASIS False-Positive Investigation (Completed)

### Actions
- Ran targeted OASIS sanity checks using latest checkpoints to validate real-world behavior.
- Added `test_oasis_sample.py` for single-case end-to-end diagnostics:
  - Slice-level class counts
  - Probability statistics
  - Top suspicious slices
  - Patient-level decision summary
- Added `analyze_dataset.py` to audit dataset composition and class balance.

### Outcome
Confirmed that high headline validation scores did not fully reflect healthy-control behavior; OASIS checks exposed persistent patient-level false-positive risk requiring training/evaluation hardening.

## Step 16.2 – Training/Evaluation Pipeline Hardening (Completed)

### Actions
- Added balanced validation split option:
  - Implemented `split_dataset_by_patient_balanced_val(...)` in `src/dataset/split_utils.py`
  - Added `--balanced_validation` flag in `src/training/train_model.py`
- Expanded clinical evaluation metrics:
  - Added Sensitivity and Specificity in `src/evaluation/metrics.py`
  - Updated reporting output in `src/evaluation/report.py`

### Outcome
Training and evaluation now support fairer monitoring for imbalanced classes and provide clinically meaningful performance signals beyond overall accuracy.

## Step 16.3 – Short-Cycle Retraining and Regression Checks (Completed)

### Actions
- Executed iterative retraining runs (5/10/20 epochs) with balanced validation enabled.
- Re-ran OASIS diagnostics after each run to measure behavioral change.

### Findings
- Slice-level behavior improved across retraining cycles (higher normal-slice fraction, lower median tumor probability on OASIS).
- Remaining issue localized to patient-level aggregation sensitivity: a small number of high-probability slices can still flip final decision.

### Files Modified / Added
- Added:
  - `test_oasis_sample.py`
  - `analyze_dataset.py`
- Modified:
  - `src/dataset/split_utils.py`
  - `src/training/train_model.py`
  - `src/evaluation/metrics.py`
  - `src/evaluation/report.py`

### Outcome
Model reliability work is now correctly tracked under a dedicated training/evaluation parent step, separate from Streamlit UI work. Next hardening item is aggregation-rule unification and thresholding at patient level.

## Step 16.4 – Aggregation Calibration, Unification, and External OASIS Verification (Completed)

### Actions
- Implemented robust patient-level aggregation strategy and made it configurable:
  - Parameters include threshold, top-k, aggregation method, suspicious-slice count, suspicious probability cutoff, and suspicious fraction.
  - Updated `src/aggregation/topk_aggregation.py` to support parameterized patient prediction.

- Unified patient decision logic across all runtime paths:
  - `app.py` (Streamlit diagnosis)
  - `src/inference.py` (CLI/programmatic inference)
  - `src/evaluation/run_evaluation.py` (evaluation pipeline)
  - `test_oasis_sample.py` (sanity script)

- Added aggregation calibration pipeline:
  - Created `src/evaluation/calibrate_aggregation.py`.
  - Performed grid search over aggregation parameters using validation patients.
  - Saved best configuration to:
    - `outputs/calibration/aggregation_calibration.json`

- Added external normal-control batch verification:
  - Created `src/evaluation/oasis_batch_check.py`.
  - Evaluated full available OASIS normal cohort with latest checkpoint + calibrated aggregation.

### Best Aggregation Parameters (Current)
- threshold: `0.90`
- top_k: `30`
- method: `median`
- min_suspicious_slices: `15`
- suspicious_prob_threshold: `0.85`
- min_suspicious_fraction: `0.50`

### Results
- Internal validation remained near-perfect with unified calibrated aggregation.
- External OASIS batch check summary:
  - Evaluated: `196`
  - False Positives: `2`
  - True Negatives: `194`
  - False Positive Rate: `0.0102`
  - Specificity: `0.9898`

### Files Modified / Added
- Modified:
  - `src/aggregation/topk_aggregation.py`
  - `src/inference.py`
  - `src/evaluation/run_evaluation.py`
  - `app.py`
  - `test_oasis_sample.py`
- Added:
  - `src/evaluation/calibrate_aggregation.py`
  - `src/evaluation/oasis_batch_check.py`
  - `outputs/calibration/aggregation_calibration.json`

### Outcome
Patient-level decision behavior is now consistent across app, inference, and evaluation. OASIS false-positive behavior has been reduced from the earlier failure mode to a low residual rate (about 1%), while preserving strong tumor detection on validation and known tumor sanity checks.

## Step 17 – Slice Threshold Calibration Integration (Completed)

### Actions
- Added threshold calibration utility in `src/evaluation/threshold_calibration.py`.
- Integrated calibration + artifact save in `src/evaluation/run_evaluation.py`.
- Wired calibrated slice threshold usage into `app.py`.

### Slice Threshold Artifact (Current)
- file: `outputs/calibration/slice_threshold_calibration.json`
- objective: balanced accuracy (with minimum-specificity guard)

### Results
- Slice-level decisions now use a calibrated threshold instead of a fixed 0.50.
- Evaluation reports include default and calibrated-threshold results.
- Streamlit uses calibrated threshold when available, with safe fallback.

### Files Modified / Added
- Added:
  - `src/evaluation/threshold_calibration.py`
- Modified:
  - `src/evaluation/run_evaluation.py`
  - `app.py`

### Outcome
Threshold calibration is now part of the standard pipeline and reusable after retraining or dataset updates.
