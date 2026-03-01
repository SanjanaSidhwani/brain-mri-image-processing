# Brain MRI AI Decision Support – Project Log

---

## Step 1 – Project Structure Initialization

**Branch:** sanjana-sidhwani  
**Status:** Completed  

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

## Step 2 – Python Environment Configuration

**Branch:** sanjana-sidhwani  
**Status:** Completed  

### Actions
- Verified Python 3.11
- Created virtual environment
- Installed required dependencies using requirements.txt
- Validated core library imports
- Committed environment setup

### Outcome
Stable and reproducible development environment configured.

## Step 3 – Dataset Structure Validation

**Branch:** sanjana-sidhwani  
**Status:** Completed  

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