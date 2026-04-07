import numpy as np
from pathlib import Path

from src.preprocessing.volume_utils import load_nifti, zscore_normalize
from src.preprocessing.slice_utils import (
    extract_axial_slices,
    filter_empty_slices
)

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


def test_volume(path, dataset_name):
    print(f"\n===== Testing {dataset_name} =====")

    volume = load_nifti(path)

    print("After Loading:")
    print("Shape:", volume.shape)
    print("Dtype:", volume.dtype)
    print("Min:", np.min(volume))
    print("Max:", np.max(volume))

    normalized = zscore_normalize(volume)

    mask = volume != 0

    if np.sum(mask) > 0:
        print("\nAfter Normalization (non-zero region):")
        print("Mean:", np.mean(normalized[mask]))
        print("Std:", np.std(normalized[mask]))
    else:
        print("Warning: No non-zero voxels found.")
    slices = extract_axial_slices(normalized)
    filtered_slices = filter_empty_slices(slices)

    print("\nSlice Processing:")
    print("Total slices:", len(slices))
    print("After filtering:", len(filtered_slices))

    if len(slices) > 0:
        retention = (len(filtered_slices) / len(slices)) * 100
        print("Retention %:", round(retention, 2))

    if filtered_slices:
        print("Sample slice shape:", filtered_slices[0].shape)
    else:
        print("No slices retained after filtering.")


if __name__ == "__main__":
    test_volume(BRATS_PATH, "BraTS")
    test_volume(OASIS_PATH, "OASIS")