import numpy as np

from src.preprocessing.modality_detection import detect_modality
from src.preprocessing.scanner_normalization import robust_intensity_scale
from src.dataset.dataset_builder import create_slice_record


def test_detect_modality_from_filename():
    assert detect_modality("BraTS20_001_flair.nii.gz") == "flair"
    assert detect_modality("IXI123-T1.nii.gz") == "t1"
    assert detect_modality("patient_t1ce.nii.gz") == "t1c"
    assert detect_modality("scan_pd.nii.gz") == "pd"


def test_detect_custom_modality_from_filename_token():
    assert detect_modality("subject_001_adc.nii.gz") == "adc"
    assert detect_modality("patient_swi_session.nii.gz") == "swi"


def test_create_slice_record_preserves_any_modality_token():
    rec = create_slice_record(
        slice_2d=np.zeros((8, 8), dtype=np.float32),
        label=0,
        patient_id="p1",
        slice_index=0,
        dataset_name="dummy",
        modality="adc",
    )
    assert rec["modality"] == "adc"
    assert rec["modalities"] == ["adc"]


def test_robust_intensity_scale_output_range():
    volume = np.zeros((8, 8, 8), dtype=np.float32)
    volume[2:6, 2:6, 2:6] = np.linspace(10, 500, num=64, dtype=np.float32).reshape(4, 4, 4)

    scaled = robust_intensity_scale(volume)

    assert scaled.shape == volume.shape
    assert np.min(scaled[volume != 0]) >= 0.0
    assert np.max(scaled[volume != 0]) <= 1.0
