import numpy as np

from src.dataset.dataset_builder import build_dataset_from_volumes
from src.dataset.split_utils import split_dataset_by_patient


BRATS_PATH = r"C:\datasets\brats\brats20-dataset-training-validation\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_flair.nii"

OASIS_PATH = r"C:\datasets\oasis\OASIS_Clean_Data\OASIS_Clean_Data\OAS1_0028_MR1_mpr_n4_anon_111_t88_masked_gfc.nii"


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