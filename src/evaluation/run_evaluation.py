import torch
from torch.utils.data import DataLoader
from src.aggregation.topk_aggregation import topk_patient_prediction, get_patient_labels
from src.evaluation.predictor import Predictor
from src.evaluation.metrics import compute_classification_metrics
from src.evaluation.report import generate_report
from src.evaluation.gradcam import GradCAM, save_gradcam_panel
from src.evaluation.threshold_calibration import (
    calibrate_binary_threshold,
    calibrate_thresholds_by_modality,
    save_modality_threshold_calibration,
    save_threshold_calibration,
)

from src.dataset.mri_dataset import MRISliceDataset
from src.dataset.split_utils import split_dataset_by_patient
from src.dataset.input_transforms import build_eval_transform

import pickle
import gzip
import json
from collections import Counter
import numpy as np
from pathlib import Path


def load_dataset(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def get_latest_checkpoint(checkpoint_dir="outputs/checkpoints"):
    checkpoint_dir = Path(checkpoint_dir)
    candidates = list(checkpoint_dir.glob("*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest)


def load_aggregation_params(config_path="outputs/calibration/aggregation_calibration.json"):
    defaults = {
        "threshold": 0.70,
        "top_k": 20,
        "method": "median",
        "min_suspicious_slices": 8,
        "suspicious_prob_threshold": 0.90,
        "min_suspicious_fraction": 0.30,
    }
    path = Path(config_path)
    if not path.exists():
        return defaults
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        params = payload.get("best", {}).get("params", {})
        merged = defaults.copy()
        merged.update({k: params[k] for k in defaults.keys() if k in params})
        return merged
    except Exception:
        return defaults


def select_representative_slice_index(records, preferred_label=None):
    best_idx = 0
    best_score = -1

    for idx, record in enumerate(records):
        if preferred_label is not None and record["label"] != preferred_label:
            continue

        brain_pixels = int(np.count_nonzero(record["slice"]))
        if brain_pixels > best_score:
            best_score = brain_pixels
            best_idx = idx

    if best_score < 0:
        for idx, record in enumerate(records):
            brain_pixels = int(np.count_nonzero(record["slice"]))
            if brain_pixels > best_score:
                best_score = brain_pixels
                best_idx = idx

    return best_idx


def select_highest_tumor_probability_slice_index(records, probabilities, preferred_label=1):
    if len(records) != len(probabilities):
        raise ValueError(
            f"records ({len(records)}) and probabilities ({len(probabilities)}) must have equal length"
        )

    best_idx = None
    best_prob = -1.0

    for idx, (record, prob_vec) in enumerate(zip(records, probabilities)):
        if preferred_label is not None and record["label"] != preferred_label:
            continue

        tumor_prob = float(prob_vec[1])
        if tumor_prob > best_prob:
            best_prob = tumor_prob
            best_idx = idx

    if best_idx is None:
        best_idx = int(np.argmax([float(p[1]) for p in probabilities]))

    return best_idx


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_records = load_dataset("data/dataset_records.pkl.gz")

    train_records, val_records = split_dataset_by_patient(dataset_records)

    print("\n===== DATA DISTRIBUTION =====")
    print("Train:", Counter([r["label"] for r in train_records]))
    print("Val:", Counter([r["label"] for r in val_records]))
    print("================================\n")

    val_transform = build_eval_transform(target_size=224, center_crop_size=180)
    val_dataset = MRISliceDataset(val_records, target_size=224, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    checkpoint_path = get_latest_checkpoint("outputs/checkpoints")
    print(f"Using checkpoint: {checkpoint_path}")

    predictor = Predictor.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )
    predictor.model.eval()
    outputs = predictor.collect_predictions(val_loader)

    aggregation_params = load_aggregation_params()
    patient_preds = topk_patient_prediction(
        records=val_records,
        probs=outputs["probabilities"],
        k=aggregation_params["top_k"],
        threshold=aggregation_params["threshold"],
        method=aggregation_params["method"],
        min_suspicious_slices=aggregation_params["min_suspicious_slices"],
        suspicious_prob_threshold=aggregation_params["suspicious_prob_threshold"],
        min_suspicious_fraction=aggregation_params["min_suspicious_fraction"],
    )

    patient_labels = get_patient_labels(val_records)

    y_true = []
    y_pred = []

    for pid in patient_labels:
        y_true.append(patient_labels[pid])
        y_pred.append(patient_preds[pid])
    print("\n===== Patient Predictions =====")

    for pid in patient_labels:
        print(f"Patient: {pid} | True: {patient_labels[pid]} | Pred: {patient_preds[pid]}")

    print("\n===== Patient-Level Evaluation =====")

    patient_probs = [[1 - p, p] for p in y_pred]

    patient_metrics = compute_classification_metrics(
        y_true,
        y_pred,
        patient_probs
    )

    generate_report(patient_metrics)


    print("\n===== Slice-Level Evaluation =====")

    metrics = compute_classification_metrics(
        outputs["true_labels"],
        outputs["predicted_labels"],
        outputs["probabilities"]
    )

    generate_report(metrics)

    # Calibrate a slice-level threshold using validation outputs.
    class1_probs = [float(p[1]) for p in outputs["probabilities"]]
    calibration = calibrate_binary_threshold(
        true_labels=outputs["true_labels"],
        positive_class_probs=class1_probs,
        objective="balanced_accuracy",
        min_specificity=0.98,
    )
    save_threshold_calibration(
        calibration,
        "outputs/calibration/slice_threshold_calibration.json",
    )

    calibrated_preds = [1 if prob >= calibration.threshold else 0 for prob in class1_probs]
    calibrated_metrics = compute_classification_metrics(
        outputs["true_labels"],
        calibrated_preds,
        outputs["probabilities"],
    )

    print("\n===== Slice-Level Evaluation (Calibrated Threshold) =====")
    print(f"Calibrated threshold: {calibration.threshold:.4f}")
    generate_report(calibrated_metrics)

    modalities = [str(r.get("modality", "unknown")).lower() for r in val_records]
    modality_calibration = calibrate_thresholds_by_modality(
        true_labels=outputs["true_labels"],
        positive_class_probs=class1_probs,
        modalities=modalities,
        objective="balanced_accuracy",
        min_specificity=0.98,
        min_samples_per_modality=20,
    )
    save_modality_threshold_calibration(
        modality_calibration,
        "outputs/calibration/modality_threshold_calibration.json",
    )

    per_modality_thresholds = {
        k: float(v["threshold"]) for k, v in modality_calibration.get("per_modality", {}).items()
    }
    global_threshold = float(modality_calibration.get("global", {}).get("threshold", calibration.threshold))

    modality_calibrated_preds = [
        1 if prob >= per_modality_thresholds.get(mod, global_threshold) else 0
        for prob, mod in zip(class1_probs, modalities)
    ]
    modality_calibrated_metrics = compute_classification_metrics(
        outputs["true_labels"],
        modality_calibrated_preds,
        outputs["probabilities"],
    )

    print("\n===== Slice-Level Evaluation (Per-Modality Calibrated Thresholds) =====")
    print(f"Global fallback threshold: {global_threshold:.4f}")
    if per_modality_thresholds:
        print("Per-modality thresholds:")
        for mod, thr in sorted(per_modality_thresholds.items()):
            print(f"  {mod}: {thr:.4f}")
    else:
        print("Per-modality thresholds: none met calibration constraints/sample requirements")
    generate_report(modality_calibrated_metrics)

    print("\n===== Grad-CAM Visualization =====")

    gradcam = GradCAM(predictor.model)

    sample_idx = select_highest_tumor_probability_slice_index(
        val_records,
        outputs["probabilities"],
        preferred_label=1,
    )
    print(
        "Selected Grad-CAM slice | "
        f"idx={sample_idx} | "
        f"label={val_records[sample_idx]['label']} | "
        f"tumor_prob={outputs['probabilities'][sample_idx][1]:.4f}"
    )
    sample_image, _ = val_dataset[sample_idx]
    image_np = sample_image[1].detach().cpu().numpy()
    image_np = (image_np - image_np.min()) / (image_np.max() + 1e-8)

    input_tensor = sample_image.unsqueeze(0).to(device)
    input_tensor.requires_grad = True

    cam = gradcam.generate(input_tensor, class_idx=1)

    save_gradcam_panel(
    image=image_np,
    cam=cam,
    save_path="outputs/figures/gradcam_panel.png"
    )


if __name__ == "__main__":
    main()