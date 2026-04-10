import numpy as np
import nibabel as nib

from .modality_detection import detect_field_strength_t, detect_modality
from .resampling import get_voxel_spacing, reorient_to_ras_image, resample_nifti_image


def load_nifti(
    file_path,
    to_ras=True,
    target_spacing=(1.0, 1.0, 1.0),
    return_metadata=False,
):
    nifti_img = nib.load(file_path)

    original_spacing = get_voxel_spacing(nifti_img)

    if to_ras:
        nifti_img = reorient_to_ras_image(nifti_img)

    if target_spacing is not None:
        nifti_img = resample_nifti_image(nifti_img, target_spacing=target_spacing)

    spacing = get_voxel_spacing(nifti_img)

    
    data = nifti_img.get_fdata(dtype=np.float32)
    
    if data.ndim == 4:
        if data.shape[-1] == 1:
            data = np.squeeze(data, axis=-1)
        else:
            raise ValueError(
                f"4D volume with last dimension {data.shape[-1]} cannot be squeezed to 3D. "
                f"Expected last dimension to be 1."
            )
    
    if data.ndim != 3:
        raise ValueError(
            f"Expected 3D volume after processing, but got shape {data.shape}"
        )
    
    # Already loaded as float32 above.

    if not return_metadata:
        return data

    metadata = {
        "modality": detect_modality(file_path, nifti_img),
        "field_strength_t": detect_field_strength_t(file_path, nifti_img),
        "voxel_spacing": spacing,
        "original_spacing": original_spacing,
    }

    return data, metadata


def zscore_normalize(volume, percentile_low=1.0, percentile_high=99.0):
    """
    Z-score normalize a volume with intensity clipping.
    
    Args:
        volume: 3D MRI volume
        percentile_low: lower percentile for clipping (default 1.0)
        percentile_high: upper percentile for clipping (default 99.0)
    
    Returns:
        Z-score normalized volume with clipped intensities
    """
    if volume.ndim != 3:
        raise ValueError(
            f"Expected 3D volume, but got shape {volume.shape}"
        )
    
    normalized = volume.copy()
    mask = volume != 0
    
    if not np.any(mask):
        raise ValueError(
            "Volume contains only zero voxels. Cannot normalize."
        )
    
    # Step 1: Clip to percentile range (only applied to non-zero voxels)
    non_zero_vals = volume[mask]
    p_low = np.percentile(non_zero_vals, percentile_low)
    p_high = np.percentile(non_zero_vals, percentile_high)
    clipped = np.clip(non_zero_vals, p_low, p_high)
    
    # Step 2: Z-score normalize on clipped values
    mean = np.mean(clipped)
    std = np.std(clipped)
    
    if std == 0:
        normalized[mask] = 0.0
    else:
        normalized[mask] = (clipped - mean) / (std + 1e-8)
    
    return normalized


def strip_skull(slice_2d: np.ndarray, margin: int = 20) -> np.ndarray:
    """
    Remove outer skull boundary by cropping to brain region and padding back.
    
    This isolates the internal brain structures (including tumors) from the
    bright skull boundary that models often use as a shortcut for classification.
    
    Args:
        slice_2d: 2D MRI slice (H, W)
        margin: padding around detected brain region in pixels
    
    Returns:
        Skull-stripped slice with same shape as input, zeros outside brain
    """
    if slice_2d.ndim != 2:
        raise ValueError(f"Expected 2D slice, got shape {slice_2d.shape}")
    
    # After z-score normalization, valid brain voxels can be both positive and negative.
    # Keep all non-background voxels instead of only positive intensities.
    mask = slice_2d != 0
    
    if not mask.any():
        return slice_2d
    
    ys, xs = np.where(mask)
    y1 = max(0, ys.min() - margin)
    y2 = min(slice_2d.shape[0], ys.max() + margin + 1)
    x1 = max(0, xs.min() - margin)
    x2 = min(slice_2d.shape[1], xs.max() + margin + 1)
    
    cropped = slice_2d[y1:y2, x1:x2]
    
    padded = np.pad(
        cropped,
        ((y1, slice_2d.shape[0] - y2), (x1, slice_2d.shape[1] - x2)),
        mode='constant',
        constant_values=0.0
    )
    
    return padded
