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
                "slice_cache": {},
                "modality_slice_cache": {},
            }

        if modality not in patient_index[pid]["by_modality"]:
            patient_index[pid]["by_modality"][modality] = []
        if modality not in patient_index[pid]["modality_slice_cache"]:
            patient_index[pid]["modality_slice_cache"][modality] = {}

        patient_index[pid]["all"].append(record)
        patient_index[pid]["by_modality"][modality].append(record)
        if record.get("slice") is not None:
            patient_index[pid]["slice_cache"][record["slice_index"]] = record["slice"]
            patient_index[pid]["modality_slice_cache"][modality][record["slice_index"]] = record["slice"]

    for pid in patient_index:
        patient_index[pid]["all"].sort(key=lambda x: x["slice_index"])
        for modality in patient_index[pid]["by_modality"]:
            patient_index[pid]["by_modality"][modality].sort(key=lambda x: x["slice_index"])

    return patient_index


def _get_slice(record: Dict, fetch_slice_fn=None) -> np.ndarray:
    sl = record.get("slice")
    if sl is not None:
        return sl

    if fetch_slice_fn is None:
        raise KeyError("Record has no 'slice' data and no fetch_slice_fn was provided")

    return fetch_slice_fn(record)


def _resolve_reference_slice(records: List[Dict], fetch_slice_fn=None) -> np.ndarray:
    for candidate in records:
        sl = _get_slice(candidate, fetch_slice_fn=fetch_slice_fn)
        if sl is not None:
            return np.asarray(sl, dtype=np.float32)

    raise ValueError("Unable to resolve a reference slice for stacking")


def stack_2_5d(
    record: Dict,
    patient_slices: List[Dict],
    patient_info: Optional[Dict] = None,
    apply_skull_strip: bool = True,
    fetch_slice_fn=None,
) -> np.ndarray:
 
    current_index = record["slice_index"]
    if patient_info is not None and "slice_cache" in patient_info:
        slice_map = patient_info["slice_cache"]
    else:
        slice_map = {r["slice_index"]: _get_slice(r, fetch_slice_fn=fetch_slice_fn) for r in patient_slices}

    reference_slice = _get_slice(record, fetch_slice_fn=fetch_slice_fn)
    if reference_slice is None:
        reference_slice = _resolve_reference_slice(patient_slices, fetch_slice_fn=fetch_slice_fn)

    h, w = np.asarray(reference_slice).shape
    zero_slice = np.zeros((h, w), dtype=np.float32)

    prev_slice = slice_map.get(current_index - 1, zero_slice)
    curr_slice = slice_map.get(current_index)
    next_slice = slice_map.get(current_index + 1, zero_slice)

    if curr_slice is None:
        curr_slice = reference_slice if reference_slice.shape == (h, w) else zero_slice

    if apply_skull_strip:
        prev_slice = strip_skull(prev_slice, margin=20)
        curr_slice = strip_skull(curr_slice, margin=20)
        next_slice = strip_skull(next_slice, margin=20)

    stacked = np.stack([prev_slice, curr_slice, next_slice], axis=-1)

    return stacked


def stack_single_channel(record: Dict, apply_skull_strip: bool = True, fetch_slice_fn=None) -> np.ndarray:
    curr_slice = _get_slice(record, fetch_slice_fn=fetch_slice_fn)
    if apply_skull_strip:
        curr_slice = strip_skull(curr_slice, margin=20)
    return curr_slice


def stack_multimodal(
    record: Dict,
    by_modality: Dict[str, List[Dict]],
    patient_info: Optional[Dict] = None,
    modality_order: Optional[List[str]] = None,
    apply_skull_strip: bool = True,
    modality_dropout_p: float = 0.0,
    fetch_slice_fn=None,
) -> np.ndarray:
    current_index = record["slice_index"]
    modalities = modality_order or sorted(by_modality.keys())

    channels = []
    channel_sources = []
    reference_slice = _get_slice(record, fetch_slice_fn=fetch_slice_fn)
    if reference_slice is None:
        reference_slice = _resolve_reference_slice(by_modality.get(modalities[0], []) if modalities else [], fetch_slice_fn=fetch_slice_fn)

    h, w = np.asarray(reference_slice).shape
    zero_slice = np.zeros((h, w), dtype=np.float32)

    for modality in modalities:
        recs = by_modality.get(modality, [])
        if patient_info is not None and "modality_slice_cache" in patient_info:
            mapping = patient_info["modality_slice_cache"].get(modality, {})
        else:
            mapping = {r["slice_index"]: _get_slice(r, fetch_slice_fn=fetch_slice_fn) for r in recs}
        sl = mapping.get(current_index, zero_slice)

        if sl is None:
            sl = zero_slice

        if sl.shape != (h, w):
            sl = cv2.resize(sl.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

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
    fetch_slice_fn=None,
) -> np.ndarray:

    pid = record["patient_id"]
    patient_info = patient_index[pid]
    patient_slices = patient_info["all"]
    by_modality = patient_info["by_modality"]

    mode = channel_mode.lower()
    if mode == "single":
        stacked = stack_single_channel(
            record,
            apply_skull_strip=apply_skull_strip,
            fetch_slice_fn=fetch_slice_fn,
        )
    elif mode == "multimodal":
        stacked = stack_multimodal(
            record,
            by_modality=by_modality,
            patient_info=patient_info,
            modality_order=modality_order,
            apply_skull_strip=apply_skull_strip,
            modality_dropout_p=modality_dropout_p,
            fetch_slice_fn=fetch_slice_fn,
        )
    else:
        # Legacy default behavior.
        stacked = stack_2_5d(
            record,
            patient_slices,
            patient_info=patient_info,
            apply_skull_strip=apply_skull_strip,
            fetch_slice_fn=fetch_slice_fn,
        )

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