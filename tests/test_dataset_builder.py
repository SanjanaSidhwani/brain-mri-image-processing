import numpy as np
from pathlib import Path

from src.dataset.dataset_builder import build_dataset_from_volumes


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"


def _find_first(patterns):
    for pattern in patterns:
        hits = sorted(DATA_RAW.rglob(pattern))
        if hits:
            return str(hits[0])
    raise FileNotFoundError(f"No file found in {DATA_RAW} for patterns {patterns}")


BRATS_PATH = _find_first(["*flair*.nii", "*flair*.nii.gz", "*.nii", "*.nii.gz"])

OASIS_PATH = _find_first(["*oasis*.nii", "*oasis*.nii.gz", "*.nii", "*.nii.gz"])


def test_dataset_builder():
    print("\n===== Testing Dataset Builder =====")

    sample_list = [
        (BRATS_PATH, 1, "BraTS20_Training_001", "BraTS"),
        (OASIS_PATH, 0, "OAS1_0028_MR1", "OASIS"),
    ]

    dataset = build_dataset_from_volumes(sample_list)

    print("Total slice records:", len(dataset))

    if len(dataset) == 0:
        print("ERROR: Dataset is empty.")
        return

    sample = dataset[0]

    print("\nSample Record Keys:", sample.keys())
    print("Slice shape:", sample["slice"].shape)
    print("Label:", sample["label"])
    print("Patient ID:", sample["patient_id"])
    print("Slice Index:", sample["slice_index"])
    print("Dataset Name:", sample["dataset"])

    # Additional integrity checks
    print("\nData Type of Slice:", sample["slice"].dtype)
    print("Min value:", np.min(sample["slice"]))
    print("Max value:", np.max(sample["slice"]))

    print("\nDataset Construction Test Completed Successfully.")


if __name__ == "__main__":
    test_dataset_builder()