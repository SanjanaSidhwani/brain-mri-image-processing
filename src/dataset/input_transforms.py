import numpy as np
import cv2
from typing import Dict, List


def build_patient_index(dataset: List[Dict]) -> Dict[str, List[Dict]]:
  
    patient_index = {}

    for record in dataset:
        pid = record["patient_id"]

        if pid not in patient_index:
            patient_index[pid] = []

        patient_index[pid].append(record)

    for pid in patient_index:
        patient_index[pid].sort(key=lambda x: x["slice_index"])

    return patient_index


def stack_2_5d(record: Dict, patient_slices: List[Dict]) -> np.ndarray:
 
    current_index = record["slice_index"]
    slice_map = {r["slice_index"]: r["slice"] for r in patient_slices}

    h, w = record["slice"].shape
    zero_slice = np.zeros((h, w), dtype=np.float32)

    prev_slice = slice_map.get(current_index - 1, zero_slice)
    curr_slice = slice_map.get(current_index)
    next_slice = slice_map.get(current_index + 1, zero_slice)

    stacked = np.stack([prev_slice, curr_slice, next_slice], axis=-1)

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
    target_size: int
) -> np.ndarray:

    pid = record["patient_id"]
    patient_slices = patient_index[pid]

    stacked = stack_2_5d(record, patient_slices)
    resized = resize_image(stacked, target_size)

    return resized