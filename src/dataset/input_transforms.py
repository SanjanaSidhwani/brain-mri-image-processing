import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple

import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from ..preprocessing.volume_utils import strip_skull


MRI_CHANNEL_MEAN: Tuple[float, float, float] = (0.0, 0.0, 0.0)
MRI_CHANNEL_STD: Tuple[float, float, float] = (1.0, 1.0, 1.0)

DEFAULT_CENTER_CROP_RATIO: float = 0.8


def build_patient_index(dataset: List[Dict]) -> Dict[str, List[Dict]]:
    patient_index = {}

    for record in dataset:
        pid = record["patient_id"]
        modality = record.get("modality", "_legacy")

        if pid not in patient_index:
            patient_index[pid] = {
                "all": [],
                "by_modality": {},
            }

        if modality not in patient_index[pid]["by_modality"]:
            patient_index[pid]["by_modality"][modality] = []

        patient_index[pid]["all"].append(record)
        patient_index[pid]["by_modality"][modality].append(record)

    for pid in patient_index:
        patient_index[pid]["all"].sort(key=lambda x: x["slice_index"])
        for modality in patient_index[pid]["by_modality"]:
            patient_index[pid]["by_modality"][modality].sort(key=lambda x: x["slice_index"])

    return patient_index


def stack_2_5d(record: Dict, patient_slices: List[Dict], apply_skull_strip: bool = True) -> np.ndarray:
 
    current_index = record["slice_index"]
    slice_map = {r["slice_index"]: r["slice"] for r in patient_slices}

    h, w = record["slice"].shape
    zero_slice = np.zeros((h, w), dtype=np.float32)

    prev_slice = slice_map.get(current_index - 1, zero_slice)
    curr_slice = slice_map.get(current_index)
    next_slice = slice_map.get(current_index + 1, zero_slice)

    if apply_skull_strip:
        prev_slice = strip_skull(prev_slice, margin=20)
        curr_slice = strip_skull(curr_slice, margin=20)
        next_slice = strip_skull(next_slice, margin=20)

    stacked = np.stack([prev_slice, curr_slice, next_slice], axis=-1)

    return stacked


def stack_single_channel(record: Dict, apply_skull_strip: bool = True) -> np.ndarray:
    curr_slice = record["slice"]
    if apply_skull_strip:
        curr_slice = strip_skull(curr_slice, margin=20)
    return curr_slice


def stack_multimodal(
    record: Dict,
    by_modality: Dict[str, List[Dict]],
    modality_order: Optional[List[str]] = None,
    apply_skull_strip: bool = True,
    modality_dropout_p: float = 0.0,
) -> np.ndarray:
    current_index = record["slice_index"]
    modalities = modality_order or sorted(by_modality.keys())

    channels = []
    channel_sources = []
    h, w = record["slice"].shape
    zero_slice = np.zeros((h, w), dtype=np.float32)

    for modality in modalities:
        recs = by_modality.get(modality, [])
        mapping = {r["slice_index"]: r["slice"] for r in recs}
        sl = mapping.get(current_index, zero_slice)
        if apply_skull_strip:
            sl = strip_skull(sl, margin=20)
        channels.append(sl)
        channel_sources.append(np.count_nonzero(sl) > 0)

    stacked = np.stack(channels, axis=-1)

    if modality_dropout_p > 0.0 and stacked.shape[-1] > 1:
        keep_mask = np.random.rand(stacked.shape[-1]) > modality_dropout_p

        # Keep at least one informative modality channel.
        if not keep_mask.any():
            idx = int(np.argmax(np.array(channel_sources, dtype=np.int32)))
            keep_mask[idx] = True

        for idx, keep in enumerate(keep_mask):
            if not keep:
                stacked[..., idx] = 0.0

    return stacked


def resize_image(image: np.ndarray, target_size: int) -> np.ndarray:

    resized = cv2.resize(
        image,
        (target_size, target_size),
        interpolation=cv2.INTER_LINEAR
    )

    return resized


def transform_record(
    record: Dict,
    patient_index: Dict[str, List[Dict]],
    target_size: int,
    pre_resize: bool = True,
    apply_skull_strip: bool = True,
    channel_mode: str = "2.5d",
    modality_order: Optional[List[str]] = None,
    modality_dropout_p: float = 0.0,
) -> np.ndarray:

    pid = record["patient_id"]
    patient_info = patient_index[pid]
    patient_slices = patient_info["all"]
    by_modality = patient_info["by_modality"]

    mode = channel_mode.lower()
    if mode == "single":
        stacked = stack_single_channel(record, apply_skull_strip=apply_skull_strip)
    elif mode == "multimodal":
        stacked = stack_multimodal(
            record,
            by_modality=by_modality,
            modality_order=modality_order,
            apply_skull_strip=apply_skull_strip,
            modality_dropout_p=modality_dropout_p,
        )
    else:
        # Legacy default behavior.
        stacked = stack_2_5d(record, patient_slices, apply_skull_strip=apply_skull_strip)

    if pre_resize:
        return resize_image(stacked, target_size)

    return stacked


def resolve_center_crop_size(
    target_size: int,
    center_crop_size: Optional[int],
) -> int:

    if target_size <= 0:
        raise ValueError(f"target_size must be > 0, got {target_size}")

    if center_crop_size is None:
        center_crop_size = int(round(target_size * DEFAULT_CENTER_CROP_RATIO))

    if center_crop_size <= 0:
        raise ValueError(f"center_crop_size must be > 0, got {center_crop_size}")

    if center_crop_size > target_size:
        raise ValueError(
            f"center_crop_size ({center_crop_size}) cannot exceed target_size ({target_size})"
        )

    return center_crop_size


def validate_normalization(
    mean: Tuple[float, ...],
    std: Tuple[float, ...],
    num_channels: int,
) -> None:

    if len(mean) != num_channels or len(std) != num_channels:
        raise ValueError(
            f"mean/std lengths must match num_channels={num_channels}, got len(mean)={len(mean)}, len(std)={len(std)}"
        )

    if any(s <= 0.0 for s in std):
        raise ValueError(f"All std values must be > 0, got {std}")


def build_train_transform(
    target_size: int = 224,
    center_crop_size: Optional[int] = None,
    mean: Optional[Tuple[float, ...]] = None,
    std: Optional[Tuple[float, ...]] = None,
    num_channels: int = 3,
):

    crop_size = resolve_center_crop_size(target_size, center_crop_size)

    if mean is None:
        mean = tuple([0.0] * num_channels)
    if std is None:
        std = tuple([1.0] * num_channels)

    validate_normalization(mean, std, num_channels=num_channels)

    return T.Compose([
        T.CenterCrop(crop_size),
        T.RandomResizedCrop(
            size=target_size,
            scale=(0.8, 1.0),
            ratio=(0.95, 1.05),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(
            degrees=10,
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        ),
        T.Normalize(mean=mean, std=std),
    ])


def build_eval_transform(
    target_size: int = 224,
    center_crop_size: Optional[int] = None,
    mean: Optional[Tuple[float, ...]] = None,
    std: Optional[Tuple[float, ...]] = None,
    num_channels: int = 3,
):

    crop_size = resolve_center_crop_size(target_size, center_crop_size)

    if mean is None:
        mean = tuple([0.0] * num_channels)
    if std is None:
        std = tuple([1.0] * num_channels)

    validate_normalization(mean, std, num_channels=num_channels)

    return T.Compose([
        T.CenterCrop(crop_size),
        T.Resize(
            size=(target_size, target_size),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),
        T.Normalize(mean=mean, std=std),
    ])