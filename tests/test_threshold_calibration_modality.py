import json
from pathlib import Path

from src.evaluation.threshold_calibration import (
    calibrate_thresholds_by_modality,
    load_threshold_for_modality,
    save_modality_threshold_calibration,
)


def test_calibrate_thresholds_by_modality_builds_payload():
    labels = [0, 0, 1, 1, 0, 1, 0, 1]
    probs = [0.1, 0.2, 0.9, 0.8, 0.3, 0.7, 0.4, 0.6]
    modalities = ["t1", "t1", "t1", "t1", "t2", "t2", "t2", "t2"]

    payload = calibrate_thresholds_by_modality(
        true_labels=labels,
        positive_class_probs=probs,
        modalities=modalities,
        min_samples_per_modality=2,
    )

    assert "global" in payload
    assert "per_modality" in payload
    assert "t1" in payload["per_modality"]
    assert "t2" in payload["per_modality"]


def test_load_threshold_for_modality_prefers_modality_then_global(tmp_path):
    config_path = tmp_path / "modality_threshold_calibration.json"
    payload = {
        "global": {"threshold": 0.55},
        "per_modality": {
            "mra": {"threshold": 0.88},
        },
    }
    save_modality_threshold_calibration(payload, str(config_path))

    thr_mra = load_threshold_for_modality("mra", modality_config_path=str(config_path))
    thr_t1 = load_threshold_for_modality("t1", modality_config_path=str(config_path))

    assert abs(thr_mra - 0.88) < 1e-8
    assert abs(thr_t1 - 0.55) < 1e-8
