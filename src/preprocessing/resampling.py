from __future__ import annotations

from typing import Tuple

import nibabel as nib
import numpy as np
from nibabel.processing import resample_to_output


def reorient_to_ras_image(img: nib.spatialimages.SpatialImage) -> nib.spatialimages.SpatialImage:
    return nib.as_closest_canonical(img)


def get_voxel_spacing(img: nib.spatialimages.SpatialImage) -> Tuple[float, float, float]:
    zooms = img.header.get_zooms()
    return float(zooms[0]), float(zooms[1]), float(zooms[2])


def resample_nifti_image(
    img: nib.spatialimages.SpatialImage,
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    order: int = 1,
) -> nib.spatialimages.SpatialImage:
    current = get_voxel_spacing(img)
    if np.allclose(np.array(current), np.array(target_spacing), atol=1e-3):
        return img
    return resample_to_output(img, voxel_sizes=target_spacing, order=order)
