import numpy as np
import torch
import torchvision.transforms as T

from src.dataset.input_transforms import (
    build_eval_transform,
    build_patient_index,
    build_train_transform,
    stack_2_5d,
    transform_record,
)
from src.preprocessing.volume_utils import strip_skull


def _make_patient_records(height: int = 256, width: int = 256):
    records = []
    for slice_index in range(3):
        records.append(
            {
                "slice": np.full((height, width), fill_value=float(slice_index + 1), dtype=np.float32),
                "label": 1,
                "patient_id": "patient_001",
                "slice_index": slice_index,
                "dataset": "dummy",
            }
        )
    return records


def test_skull_strip_removes_outer_boundary():
    height, width = 256, 256

    outer_val = 50.0
    inner_val = 100.0

    slice_2d = np.zeros((height, width), dtype=np.float32)
    slice_2d[20:230, 20:230] = outer_val
    slice_2d[50:200, 50:200] = inner_val
    
    stripped = strip_skull(slice_2d, margin=10)
    
    assert stripped.shape == (height, width)
    assert np.any(stripped == inner_val)
    assert stripped[0, 0] < 1.0


def test_transform_record_resize_and_no_resize_modes():
    records = _make_patient_records(height=256, width=192)
    patient_index = build_patient_index(records)

    resized = transform_record(
        record=records[1],
        patient_index=patient_index,
        target_size=224,
        pre_resize=True,
        apply_skull_strip=True,
    )
    native = transform_record(
        record=records[1],
        patient_index=patient_index,
        target_size=224,
        pre_resize=False,
        apply_skull_strip=True,
    )

    assert resized.shape == (224, 224, 3)
    assert native.shape == (256, 192, 3)


def test_stack_2_5d_with_skull_stripping():
    records = _make_patient_records(height=256, width=256)
    patient_index = build_patient_index(records)
    
    stacked_with_strip = stack_2_5d(records[1], records, apply_skull_strip=True)
    stacked_no_strip = stack_2_5d(records[1], records, apply_skull_strip=False)
    
    assert stacked_with_strip.shape == (256, 256, 3)
    assert stacked_no_strip.shape == (256, 256, 3)


def test_transform_record_resize_and_no_resize_modes_old():
    records = _make_patient_records(height=256, width=192)
    patient_index = build_patient_index(records)

    resized = transform_record(
        record=records[1],
        patient_index=patient_index,
        target_size=224,
        pre_resize=True,
    )
    native = transform_record(
        record=records[1],
        patient_index=patient_index,
        target_size=224,
        pre_resize=False,
    )

    assert resized.shape == (224, 224, 3)
    assert native.shape == (256, 192, 3)


def test_train_transform_policy_and_output_shape():
    transform = build_train_transform(target_size=224, center_crop_size=180)
    assert isinstance(transform, T.Compose)

    transform_names = [t.__class__.__name__ for t in transform.transforms]
    assert transform_names == [
        "CenterCrop",
        "RandomResizedCrop",
        "RandomHorizontalFlip",
        "RandomRotation",
        "Normalize",
    ]

    x = torch.randn(3, 224, 224)
    y = transform(x)
    assert y.shape == (3, 224, 224)


def test_eval_transform_is_deterministic_and_center_crop_only_plus_resize():
    transform = build_eval_transform(target_size=224, center_crop_size=180)

    transform_names = [t.__class__.__name__ for t in transform.transforms]
    assert transform_names == ["CenterCrop", "Resize", "Normalize"]

    x = torch.randn(3, 224, 224)
    y1 = transform(x)
    y2 = transform(x)

    assert y1.shape == (3, 224, 224)
    assert torch.allclose(y1, y2)


def test_train_transform_preserves_channel_alignment_for_identical_channels():
    transform = build_train_transform(target_size=224, center_crop_size=180)

    channel = torch.linspace(0.0, 1.0, steps=224 * 224, dtype=torch.float32).view(224, 224)
    x = torch.stack([channel, channel.clone(), channel.clone()], dim=0)

    torch.manual_seed(0)
    y = transform(x)

    assert y.shape == (3, 224, 224)
    assert torch.allclose(y[0], y[1], atol=1e-6)
    assert torch.allclose(y[1], y[2], atol=1e-6)


if __name__ == "__main__":
    test_skull_strip_removes_outer_boundary()
    test_stack_2_5d_with_skull_stripping()
    test_transform_record_resize_and_no_resize_modes()
    test_train_transform_policy_and_output_shape()
    test_eval_transform_is_deterministic_and_center_crop_only_plus_resize()
    test_train_transform_preserves_channel_alignment_for_identical_channels()
    print("All tests passed!")