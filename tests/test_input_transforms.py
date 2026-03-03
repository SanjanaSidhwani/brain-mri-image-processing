from src.dataset.dataset_builder import build_dataset_from_volumes
from src.dataset.split_utils import split_dataset_by_patient
from src.dataset.input_transforms import build_patient_index, transform_record


BRATS_PATH = r"C:\datasets\brats\brats20-dataset-training-validation\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_flair.nii"

OASIS_PATH = r"C:\datasets\oasis\OASIS_Clean_Data\OASIS_Clean_Data\OAS1_0028_MR1_mpr_n4_anon_111_t88_masked_gfc.nii"


def test_input_transform():
    print("\n===== Testing Input Transform (2.5D + Resize) =====")

    sample_list = [
        (BRATS_PATH, 1, "BraTS20_Training_001", "BraTS"),
        (OASIS_PATH, 0, "OAS1_0028_MR1", "OASIS"),
    ]

    dataset = build_dataset_from_volumes(sample_list)
    train_dataset, _ = split_dataset_by_patient(dataset, train_ratio=0.5)

    patient_index = build_patient_index(train_dataset)

    sample_record = train_dataset[0]

    transformed = transform_record(
        sample_record,
        patient_index,
        target_size=224
    )

    print("Transformed shape:", transformed.shape)

    assert transformed.shape == (224, 224, 3)

    print("Input Transform Test Completed Successfully.")


if __name__ == "__main__":
    test_input_transform()