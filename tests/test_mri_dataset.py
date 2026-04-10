import numpy as np
import torch

from src.dataset.mri_dataset import MRISliceDataset, create_dataloader


def generate_dummy_records(num_samples=10):
    records = []

    for i in range(num_samples):
        record = {
            "slice": np.random.rand(224, 224),
            "label": i % 2,
            "patient_id": f"patient_{i//5}",
            "slice_index": i,
            "dataset": "dummy"
        }

        records.append(record)

    return records


def test_dataset_creation():
    records = generate_dummy_records()

    dataset = MRISliceDataset(records)

    assert len(dataset) == len(records)


def test_dataset_sample_format():
    records = generate_dummy_records()

    dataset = MRISliceDataset(records)

    image, label = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert isinstance(label, torch.Tensor)

    assert image.shape == (3, 224, 224)
    assert label.dtype == torch.long


def test_dataloader_batch():
    records = generate_dummy_records(20)

    dataset = MRISliceDataset(records)

    dataloader = create_dataloader(dataset, batch_size=4)

    images, labels = next(iter(dataloader))

    assert images.shape == (4, 3, 224, 224)
    assert labels.shape == (4,)


def test_dataset_auto_mode_single_channel_with_modality_records():
    records = generate_dummy_records(6)
    for r in records:
        r["modality"] = "flair"

    dataset = MRISliceDataset(records, channel_mode="auto")
    image, _ = dataset[0]

    assert dataset.resolved_channel_mode == "single"
    assert dataset.input_channels == 1
    assert image.shape == (1, 224, 224)


def test_dataset_lazy_loading_for_lightweight_records(monkeypatch):
    records = [
        {
            "label": 1,
            "patient_id": "patient_1",
            "slice_index": 1,
            "dataset": "dummy",
            "volume_path": "dummy_path.nii.gz",
            "modality": "t1",
            "to_ras": False,
            "target_spacing": None,
            "apply_scanner_normalization": False,
            "use_histogram_standardization": False,
        }
    ]

    dataset = MRISliceDataset(records, channel_mode="single", target_size=224)

    fake_volume = np.zeros((32, 32, 3), dtype=np.float32)
    fake_volume[:, :, 1] = 2.0

    monkeypatch.setattr(dataset, "_get_preprocessed_volume", lambda rec: fake_volume)

    image, label = dataset[0]

    assert image.shape == (1, 224, 224)
    assert torch.all(image > 0)
    assert int(label.item()) == 1


if __name__ == "__main__":
    test_dataset_creation()
    test_dataset_sample_format()
    test_dataloader_batch()

    print("===== Testing MRI Dataset Loader =====")
    print("All Step 8.1 tests passed successfully.")