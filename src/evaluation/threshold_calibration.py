from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional
import json

import numpy as np


@dataclass
class ThresholdCalibrationResult:
    threshold: float
    objective: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sensitivity: float
    specificity: float
    balanced_accuracy: float
    tp: int
    tn: int
    fp: int
    fn: int
    min_specificity_constraint: Optional[float] = None
    min_sensitivity_constraint: Optional[float] = None


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _metrics_from_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(np.int64)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    precision = _safe_div(tp, tp + fp)
    recall = sensitivity
    accuracy = _safe_div(tp + tn, len(y_true))
    f1_score = _safe_div(2 * precision * recall, precision + recall)
    balanced_accuracy = 0.5 * (sensitivity + specificity)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def calibrate_binary_threshold(
    true_labels: Iterable[int],
    positive_class_probs: Iterable[float],
    objective: str = "balanced_accuracy",
    min_specificity: Optional[float] = None,
    min_sensitivity: Optional[float] = None,
) -> ThresholdCalibrationResult:
    if objective not in {"balanced_accuracy", "f1_score", "accuracy"}:
        raise ValueError(
            f"Unsupported objective '{objective}'. Choose from: balanced_accuracy, f1_score, accuracy"
        )

    y_true = np.asarray(list(true_labels), dtype=np.int64)
    y_prob = np.asarray(list(positive_class_probs), dtype=np.float32)

    if y_true.ndim != 1 or y_prob.ndim != 1:
        raise ValueError("true_labels and positive_class_probs must be 1D arrays")
    if len(y_true) != len(y_prob):
        raise ValueError("true_labels and positive_class_probs must have the same length")
    if len(y_true) == 0:
        raise ValueError("Cannot calibrate threshold on empty inputs")

    # Evaluate meaningful candidate points: unique probabilities plus edges.
    candidates = np.unique(y_prob)
    candidates = np.concatenate(([0.0], candidates, [1.0]))

    scored = []
    for threshold in candidates:
        metrics = _metrics_from_threshold(y_true, y_prob, float(threshold))

        if min_specificity is not None and metrics["specificity"] < float(min_specificity):
            continue
        if min_sensitivity is not None and metrics["sensitivity"] < float(min_sensitivity):
            continue

        scored.append(metrics)

    if not scored:
        # Fall back to unconstrained optimization when constraints are too strict.
        for threshold in candidates:
            scored.append(_metrics_from_threshold(y_true, y_prob, float(threshold)))

    best = sorted(
        scored,
        key=lambda m: (m[objective], m["specificity"], m["sensitivity"], -abs(m["threshold"] - 0.5)),
        reverse=True,
    )[0]

    return ThresholdCalibrationResult(
        threshold=float(best["threshold"]),
        objective=objective,
        accuracy=float(best["accuracy"]),
        precision=float(best["precision"]),
        recall=float(best["recall"]),
        f1_score=float(best["f1_score"]),
        sensitivity=float(best["sensitivity"]),
        specificity=float(best["specificity"]),
        balanced_accuracy=float(best["balanced_accuracy"]),
        tp=int(best["tp"]),
        tn=int(best["tn"]),
        fp=int(best["fp"]),
        fn=int(best["fn"]),
        min_specificity_constraint=min_specificity,
        min_sensitivity_constraint=min_sensitivity,
    )


def save_threshold_calibration(result: ThresholdCalibrationResult, filepath: str) -> None:
    payload = asdict(result)
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def calibrate_thresholds_by_modality(
    true_labels: Iterable[int],
    positive_class_probs: Iterable[float],
    modalities: Iterable[str],
    objective: str = "balanced_accuracy",
    min_specificity: Optional[float] = None,
    min_sensitivity: Optional[float] = None,
    min_samples_per_modality: int = 20,
) -> Dict[str, Dict]:
    y_true = np.asarray(list(true_labels), dtype=np.int64)
    y_prob = np.asarray(list(positive_class_probs), dtype=np.float32)
    y_mod = np.asarray([str(m).lower() if m is not None else "unknown" for m in modalities], dtype=object)

    if len(y_true) != len(y_prob) or len(y_true) != len(y_mod):
        raise ValueError("true_labels, positive_class_probs, and modalities must have equal length")

    global_result = calibrate_binary_threshold(
        true_labels=y_true,
        positive_class_probs=y_prob,
        objective=objective,
        min_specificity=min_specificity,
        min_sensitivity=min_sensitivity,
    )

    per_modality: Dict[str, ThresholdCalibrationResult] = {}
    sample_counts: Dict[str, int] = {}

    for modality in sorted(np.unique(y_mod)):
        mask = y_mod == modality
        count = int(np.sum(mask))
        sample_counts[modality] = count

        if count < int(min_samples_per_modality):
            continue

        # Skip pathological single-class modalities for threshold search.
        if len(np.unique(y_true[mask])) < 2:
            continue

        per_modality[modality] = calibrate_binary_threshold(
            true_labels=y_true[mask],
            positive_class_probs=y_prob[mask],
            objective=objective,
            min_specificity=min_specificity,
            min_sensitivity=min_sensitivity,
        )

    return {
        "global": asdict(global_result),
        "per_modality": {k: asdict(v) for k, v in per_modality.items()},
        "objective": objective,
        "min_specificity_constraint": min_specificity,
        "min_sensitivity_constraint": min_sensitivity,
        "min_samples_per_modality": int(min_samples_per_modality),
        "sample_counts": sample_counts,
    }


def save_modality_threshold_calibration(payload: Dict, filepath: str) -> None:
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_threshold_for_modality(
    modality: Optional[str],
    modality_config_path: str = "outputs/calibration/modality_threshold_calibration.json",
    global_config_path: str = "outputs/calibration/slice_threshold_calibration.json",
    default_threshold: float = 0.5,
) -> float:
    modality_key = (modality or "unknown").lower()

    modality_path = Path(modality_config_path)
    if modality_path.exists():
        try:
            with open(modality_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            per_modality = payload.get("per_modality", {})
            if modality_key in per_modality:
                thr = float(per_modality[modality_key].get("threshold", default_threshold))
                if 0.0 <= thr <= 1.0:
                    return thr

            global_payload = payload.get("global", {})
            if isinstance(global_payload, dict):
                thr = float(global_payload.get("threshold", default_threshold))
                if 0.0 <= thr <= 1.0:
                    return thr
        except Exception:
            pass

    global_path = Path(global_config_path)
    if global_path.exists():
        try:
            with open(global_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            thr = float(payload.get("threshold", default_threshold))
            if 0.0 <= thr <= 1.0:
                return thr
        except Exception:
            pass

    return default_threshold
