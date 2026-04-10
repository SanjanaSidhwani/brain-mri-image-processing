import torch
from torch.utils.data import DataLoader
import argparse
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


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run model evaluation")
    parser.add_argument("--dataset_path", type=str, default="data/dataset_records.pkl.gz")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--eval_scope", type=str, default="val", choices=["val", "all"])
    parser.add_argument("--target_size", type=int, default=224)
    parser.add_argument("--center_crop_size", type=int, default=180)
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


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
        "healthy_override_topk_max": 0.20,
        "healthy_override_max_suspicious_slices": 3,
        "healthy_override_max_suspicious_fraction": 0.05,
        "hard_tumor_topk_min": 0.95,
        "hard_tumor_min_suspicious_slices": 3,
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

    def _brain_pixels(rec):
        sl = rec.get("slice")
        if sl is None:
            return 0
        return int(np.count_nonzero(sl))

    for idx, record in enumerate(records):
        if preferred_label is not None and record["label"] != preferred_label:
            continue

        brain_pixels = _brain_pixels(record)
        if brain_pixels > best_score:
            best_score = brain_pixels
            best_idx = idx

    if best_score < 0:
        for idx, record in enumerate(records):
            brain_pixels = _brain_pixels(record)
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


def print_dataset_breakdown(records, title):
    dataset_counts = Counter([str(r.get("dataset", "unknown")).lower() for r in records])
    print(f"\n===== {title} Dataset Breakdown =====")
    for name, count in sorted(dataset_counts.items()):
        print(f"{name}: {count}")
    print("================================")


def maybe_report_dataset_metrics(records, probs, preds, title_prefix):
    labels = [int(r["label"]) for r in records]
    if len(set(labels)) < 2:
        print(f"\n===== {title_prefix} =====")
        print("Skipped full metric report: only one class present in ground truth.")
        print(f"Ground truth distribution: {Counter(labels)}")
        print(f"Prediction distribution: {Counter(preds)}")
        return

    metrics = compute_classification_metrics(labels, preds, probs)
    print(f"\n===== {title_prefix} =====")
    generate_report(metrics)


def main():

    args = parse_arguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_records = load_dataset(args.dataset_path)

    train_records, val_records = split_dataset_by_patient(dataset_records)
    eval_records = dataset_records if args.eval_scope == "all" else val_records

    print("\n===== DATA DISTRIBUTION =====")
    print("Train:", Counter([r["label"] for r in train_records]))
    print("Val:", Counter([r["label"] for r in val_records]))
    if args.eval_scope == "all":
        print("Eval scope: all records")
        print("Eval:", Counter([r["label"] for r in eval_records]))
    else:
        print("Eval scope: validation split")
    print("================================\n")

    print_dataset_breakdown(train_records, "Train")
    print_dataset_breakdown(val_records, "Val")
    print_dataset_breakdown(eval_records, "Eval")

    val_transform = build_eval_transform(
        target_size=args.target_size,
        center_crop_size=args.center_crop_size,
    )
    val_dataset = MRISliceDataset(
        eval_records,
        target_size=args.target_size,
        transform=val_transform,
        channel_mode="2.5d",
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    checkpoint_path = args.checkpoint_path or get_latest_checkpoint("outputs/checkpoints")
    print(f"Using checkpoint: {checkpoint_path}")

    predictor = Predictor.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )
    predictor.model.eval()
    outputs = predictor.collect_predictions(val_loader)

    aggregation_params = load_aggregation_params()
    patient_preds = topk_patient_prediction(
        records=eval_records,
        probs=outputs["probabilities"],
        k=aggregation_params["top_k"],
        threshold=aggregation_params["threshold"],
        method=aggregation_params["method"],
        min_suspicious_slices=aggregation_params["min_suspicious_slices"],
        suspicious_prob_threshold=aggregation_params["suspicious_prob_threshold"],
        min_suspicious_fraction=aggregation_params["min_suspicious_fraction"],
    )

    patient_labels = get_patient_labels(eval_records)

    y_true = []
    y_pred = []

    for pid in patient_labels:
        y_true.append(patient_labels[pid])
        y_pred.append(patient_preds[pid])
    print("\n===== Patient Predictions =====")
    preview_ids = list(patient_labels.keys())[:25]
    for pid in preview_ids:
        print(f"Patient: {pid} | True: {patient_labels[pid]} | Pred: {patient_preds[pid]}")
    if len(patient_labels) > len(preview_ids):
        print(f"... {len(patient_labels) - len(preview_ids)} more patients not shown")

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

    modalities = [str(r.get("modality", "unknown")).lower() for r in eval_records]
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

    print("\n===== Per-Dataset Slice-Level Evaluation =====")
    by_dataset_indices = {}
    for idx, record in enumerate(eval_records):
        dname = str(record.get("dataset", "unknown")).lower()
        by_dataset_indices.setdefault(dname, []).append(idx)

    for dname, indices in sorted(by_dataset_indices.items()):
        d_records = [eval_records[i] for i in indices]
        d_probs = [outputs["probabilities"][i] for i in indices]
        d_preds = [outputs["predicted_labels"][i] for i in indices]
        maybe_report_dataset_metrics(
            d_records,
            d_probs,
            d_preds,
            title_prefix=f"{dname.upper()} Slice-Level Evaluation",
        )

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