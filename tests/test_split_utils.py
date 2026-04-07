import numpy as np
from pathlib import Path

from src.dataset.dataset_builder import build_dataset_from_volumes
from src.dataset.split_utils import split_dataset_by_patient


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


def test_patient_split():
    print("\n===== Testing Patient-Level Split =====")

    # Build dataset with multiple patients
    sample_list = [
        (BRATS_PATH, 1, "BraTS20_Training_001", "BraTS"),
        (OASIS_PATH, 0, "OAS1_0028_MR1", "OASIS"),
    ]

    dataset = build_dataset_from_volumes(sample_list)

    print("Total slice records:", len(dataset))

    train_dataset, val_dataset = split_dataset_by_patient(
        dataset,
        train_ratio=0.5,
        seed=42
    )

    print("Train slices:", len(train_dataset))
    print("Validation slices:", len(val_dataset))

    train_ids = {r["patient_id"] for r in train_dataset}
    val_ids = {r["patient_id"] for r in val_dataset}

    print("Train patients:", train_ids)
    print("Validation patients:", val_ids)

    overlap = train_ids & val_ids
    print("Overlap:", overlap)

    if overlap:
        print("ERROR: Data leakage detected!")
    else:
        print("No leakage detected. Split is safe.")

    # Extra integrity check
    total_after_split = len(train_dataset) + len(val_dataset)
    print("Total after split:", total_after_split)

    assert total_after_split == len(dataset), "Mismatch in slice counts!"

    print("\nPatient-Level Split Test Completed Successfully.")


if __name__ == "__main__":
    test_patient_split()