import numpy as np

from ..preprocessing.volume_utils import load_nifti, zscore_normalize
from ..preprocessing.slice_utils import (
    extract_axial_slices,
    is_informative_slice
)
from ..preprocessing.modality_detection import detect_modality
from ..preprocessing.scanner_normalization import (
    apply_optional_histogram_standardization,
    normalize_by_scanner_strength,
)


def create_slice_record(
    slice_2d,
    label,
    patient_id,
    slice_index,
    dataset_name,
    volume_path=None,
    modality="unknown",
    field_strength_t=None,
    voxel_spacing=None,
    to_ras=True,
    target_spacing=(1.0, 1.0, 1.0),
    apply_scanner_normalization=True,
    use_histogram_standardization=False,
    include_slice_data=True,
):
    record = {
        "label": label,
        "patient_id": patient_id,
        "slice_index": slice_index,  
        "dataset": dataset_name,
        "volume_path": volume_path,
        "modality": modality,
        "modalities": [modality],
        "field_strength_t": field_strength_t,
        "voxel_spacing": voxel_spacing,
        "to_ras": bool(to_ras),
        "target_spacing": target_spacing,
        "apply_scanner_normalization": bool(apply_scanner_normalization),
        "use_histogram_standardization": bool(use_histogram_standardization),
    }

    if include_slice_data:
        record["slice"] = slice_2d

    return record


def _normalize_volume_entry(volume_item):
    if hasattr(volume_item, "volume_path"):
        return {
            "volume_path": volume_item.volume_path,
            "label": volume_item.label,
            "patient_id": volume_item.patient_id,
            "dataset_name": volume_item.dataset_name,
            "modality": getattr(volume_item, "modality", "unknown"),
            "field_strength_t": getattr(volume_item, "field_strength_t", None),
        }

    if isinstance(volume_item, dict):
        required = ("volume_path", "label", "patient_id", "dataset_name")
        if any(k not in volume_item for k in required):
            raise ValueError(f"Volume dict is missing required keys: {required}")
        out = dict(volume_item)
        out.setdefault("modality", "unknown")
        out.setdefault("field_strength_t", None)
        return out

    if isinstance(volume_item, tuple) or isinstance(volume_item, list):
        if len(volume_item) < 4:
            raise ValueError("Expected tuple/list volume entry with at least 4 items")
        out = {
            "volume_path": volume_item[0],
            "label": volume_item[1],
            "patient_id": volume_item[2],
            "dataset_name": volume_item[3],
            "modality": volume_item[4] if len(volume_item) > 4 else "unknown",
            "field_strength_t": volume_item[5] if len(volume_item) > 5 else None,
        }
        return out

    raise TypeError(f"Unsupported volume entry type: {type(volume_item)}")


def build_volume_dataset(
    volume,
    label,
    patient_id,
    dataset_name,
    volume_path,
    threshold=0.05,
    modality="unknown",
    field_strength_t=None,
    voxel_spacing=None,
    to_ras=True,
    target_spacing=(1.0, 1.0, 1.0),
    apply_scanner_normalization=True,
    use_histogram_standardization=False,
    include_slice_data=True,
):
    
    if volume.ndim != 3:
        raise ValueError(
            f"Expected 3D volume before dataset construction, got shape {volume.shape}"
        )
    
    slices = extract_axial_slices(volume)
    
    slice_records = []
    
    for original_index, slice_2d in enumerate(slices):
        
        if is_informative_slice(slice_2d, threshold=threshold):
            
            record = create_slice_record(
                slice_2d=slice_2d,
                label=label,
                patient_id=patient_id,
                slice_index=original_index, 
                dataset_name=dataset_name,
                volume_path=volume_path,
                modality=modality,
                field_strength_t=field_strength_t,
                voxel_spacing=voxel_spacing,
                to_ras=to_ras,
                target_spacing=target_spacing,
                apply_scanner_normalization=apply_scanner_normalization,
                use_histogram_standardization=use_histogram_standardization,
                include_slice_data=include_slice_data,
            )
            
            slice_records.append(record)
    
    return slice_records


def build_dataset_from_volumes(
    list_of_volumes,
    threshold=0.05,
    target_spacing=(1.0, 1.0, 1.0),
    to_ras=True,
    apply_scanner_normalization=True,
    use_histogram_standardization=False,
    histogram_landmarks=None,
    record_mode="full",
):
    if record_mode not in {"full", "lightweight"}:
        raise ValueError("record_mode must be one of {'full', 'lightweight'}")

    include_slice_data = record_mode == "full"

    master_dataset = []
    
    for volume_item in list_of_volumes:
        entry = _normalize_volume_entry(volume_item)

        volume, meta = load_nifti(
            entry["volume_path"],
            to_ras=to_ras,
            target_spacing=target_spacing,
            return_metadata=True,
        )

        modality = entry.get("modality") or meta.get("modality") or detect_modality(entry["volume_path"])
        field_strength_t = entry.get("field_strength_t")
        if field_strength_t is None:
            field_strength_t = meta.get("field_strength_t")

        if apply_scanner_normalization:
            volume = normalize_by_scanner_strength(volume, field_strength_t)
        if use_histogram_standardization:
            volume = apply_optional_histogram_standardization(
                volume=volume,
                modality=modality,
                landmark_map=histogram_landmarks,
            )

        normalized_volume = zscore_normalize(volume)
        
        slice_records = build_volume_dataset(
            volume=normalized_volume,
            label=entry["label"],
            patient_id=entry["patient_id"],
            dataset_name=entry["dataset_name"],
            volume_path=entry["volume_path"],
            threshold=threshold,
            modality=modality,
            field_strength_t=field_strength_t,
            voxel_spacing=meta.get("voxel_spacing"),
            to_ras=to_ras,
            target_spacing=target_spacing,
            apply_scanner_normalization=apply_scanner_normalization,
            use_histogram_standardization=use_histogram_standardization,
            include_slice_data=include_slice_data,
        )
        
        master_dataset.extend(slice_records)
    
    return master_dataset