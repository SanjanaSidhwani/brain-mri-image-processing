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

## Step 8 – Training Pipeline Implementation

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