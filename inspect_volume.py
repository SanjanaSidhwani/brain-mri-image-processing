import nibabel as nib
import numpy as np
from pathlib import Path


def inspect_volume(name, path):
    print(f"\n===== {name} =====")
    volume = nib.load(path)
    data = volume.get_fdata()
    print("Shape:", data.shape)
    print("Dtype:", data.dtype)
    print("Min:", np.min(data))
    print("Max:", np.max(data))


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"


def find_first(patterns):
    for pattern in patterns:
        matches = sorted(DATA_RAW.rglob(pattern))
        if matches:
            return str(matches[0])
    raise FileNotFoundError(f"No dataset file found in {DATA_RAW} for patterns: {patterns}")


brats_path = find_first(["*flair*.nii", "*flair*.nii.gz", "*t2*.nii", "*t2*.nii.gz"])
oasis_path = find_first(["*oasis*.nii", "*oasis*.nii.gz", "*.nii", "*.nii.gz"])

inspect_volume("BraTS FLAIR", brats_path)
inspect_volume("OASIS", oasis_path)