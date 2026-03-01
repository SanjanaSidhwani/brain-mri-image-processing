import nibabel as nib
import numpy as np


def inspect_volume(name, path):
    print(f"\n===== {name} =====")
    volume = nib.load(path)
    data = volume.get_fdata()
    print("Shape:", data.shape)
    print("Dtype:", data.dtype)
    print("Min:", np.min(data))
    print("Max:", np.max(data))


# BraTS FLAIR sample
brats_path = r"C:\datasets\brats\brats20-dataset-training-validation\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_flair.nii"

# OASIS sample (replace with one real OASIS file path)
oasis_path = r"C:\datasets\oasis\OASIS_Clean_Data\OASIS_Clean_Data\OAS1_0028_MR1_mpr_n4_anon_111_t88_masked_gfc.nii"

inspect_volume("BraTS FLAIR", brats_path)
inspect_volume("OASIS", oasis_path)