from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np


def robust_intensity_scale(volume: np.ndarray, low_q: float = 1.0, high_q: float = 99.0) -> np.ndarray:
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {volume.shape}")

    scaled = volume.astype(np.float32, copy=True)
    mask = scaled != 0
    if not np.any(mask):
        return scaled

    values = scaled[mask]

    # Guard against memory pressure on large volumes by percentile estimation from a sample.
    max_percentile_samples = 200_000
    if values.size > max_percentile_samples:
        step = int(np.ceil(values.size / max_percentile_samples))
        values = values[::step]

    lo = np.percentile(values, low_q)
    hi = np.percentile(values, high_q)
    if hi <= lo:
        return scaled

    denom = float(hi - lo)
    if denom <= 0:
        return scaled

    # In-place arithmetic to reduce memory spikes on large volumes.
    np.subtract(scaled, lo, out=scaled)
    np.divide(scaled, denom, out=scaled, where=mask)
    np.clip(scaled, 0.0, 1.0, out=scaled)
    scaled[~mask] = 0.0
    return scaled


@dataclass
class HistogramLandmarks:
    modality: str
    percentiles: np.ndarray
    values: np.ndarray


def compute_histogram_landmarks(
    volumes: Iterable[np.ndarray],
    modality: str,
    percentiles: Optional[Iterable[float]] = None,
) -> HistogramLandmarks:
    pct = np.array(list(percentiles) if percentiles is not None else [1, 10, 25, 50, 75, 90, 99], dtype=np.float32)

    collected = []
    for vol in volumes:
        mask = vol != 0
        if not np.any(mask):
            continue
        collected.append(np.percentile(vol[mask], pct))

    if not collected:
        raise ValueError("Cannot compute histogram landmarks from empty/non-informative volumes")

    values = np.median(np.stack(collected, axis=0), axis=0)
    return HistogramLandmarks(modality=modality, percentiles=pct, values=values.astype(np.float32))


def histogram_standardize(volume: np.ndarray, landmarks: HistogramLandmarks) -> np.ndarray:
    standardized = volume.astype(np.float32, copy=True)
    mask = standardized != 0
    if not np.any(mask):
        return standardized

    src_vals = np.percentile(standardized[mask], landmarks.percentiles)
    if np.any(np.diff(src_vals) <= 0) or np.any(np.diff(landmarks.values) <= 0):
        return standardized

    standardized[mask] = np.interp(standardized[mask], src_vals, landmarks.values).astype(np.float32)
    return standardized


def normalize_by_scanner_strength(volume: np.ndarray, field_strength_t: Optional[float]) -> np.ndarray:
    scaled = robust_intensity_scale(volume)

    # Keep scanner-specific dynamic ranges roughly comparable without overfitting assumptions.
    if field_strength_t is None:
        return scaled

    if field_strength_t >= 2.5:
        return np.clip(scaled * 0.95, 0.0, 1.0)

    return np.clip(scaled * 1.05, 0.0, 1.0)


def apply_optional_histogram_standardization(
    volume: np.ndarray,
    modality: str,
    landmark_map: Optional[Dict[str, HistogramLandmarks]] = None,
) -> np.ndarray:
    if not landmark_map:
        return volume

    landmarks = landmark_map.get(modality)
    if landmarks is None:
        return volume

    return histogram_standardize(volume, landmarks)
