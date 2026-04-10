import argparse
import gzip
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.aggregation.topk_aggregation import topk_patient_prediction
from src.dataset.input_transforms import build_eval_transform
from src.dataset.mri_dataset import MRISliceDataset
from src.evaluation.predictor import Predictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate current model on IXI records only")
    parser.add_argument("--dataset_path", type=str, default="data/dataset_records.pkl.gz")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ixi",
        help="Value of record['dataset'] to evaluate (e.g., ixi, brats, oasis)",
    )
    parser.add_argument(
        "--path_contains",
        type=str,
        default=None,
        help="Optional case-insensitive substring that must appear in volume_path (e.g., brats2021)",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--target_size", type=int, default=224)
    parser.add_argument("--center_crop_size", type=int, default=180)
    parser.add_argument("--output_json", type=str, default="outputs/reports/ixi_evaluation_report.json")
    return parser.parse_args()


def get_latest_checkpoint(checkpoint_dir: Path) -> Path:
    checkpoints = sorted(checkpoint_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    return checkpoints[-1]


def load_records(path: Path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def safe_metrics(y_true, y_pred, probabilities):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])

    total = max(1, tn + fp + fn + tp)
    accuracy = (tn + tp) / total
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = (2 * precision * recall) / max(1e-12, (precision + recall))

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "sensitivity": float(recall),
        "specificity": float(specificity),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }

    # ROC AUC is undefined when only one class exists in y_true.
    if len(set(y_true)) == 2 and probabilities is not None and len(probabilities) > 0:
        from sklearn.metrics import roc_auc_score

        class1_probs = [float(p[1]) for p in probabilities]
        metrics["roc_auc"] = float(roc_auc_score(y_true, class1_probs))
    else:
        metrics["roc_auc"] = None

    return metrics


def print_metric_block(title: str, metrics: dict):
    cm = metrics["confusion_matrix"]
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"F1 Score:    {metrics['f1_score']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    if metrics["roc_auc"] is None:
        print("ROC AUC:     N/A (single-class labels)")
    else:
        print(f"ROC AUC:     {metrics['roc_auc']:.4f}")
    print("Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(cm)


def main():
    args = parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else get_latest_checkpoint(
        PROJECT_ROOT / "outputs" / "checkpoints"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    all_records = load_records(dataset_path)
    dataset_key = str(args.dataset_name).lower().strip()
    selected_records = [r for r in all_records if str(r.get("dataset", "")).lower() == dataset_key]

    if args.path_contains:
        needle = str(args.path_contains).lower()
        selected_records = [
            r for r in selected_records if needle in str(r.get("volume_path", "")).lower()
        ]

    if not selected_records:
        raise ValueError(
            f"No records found for dataset={dataset_key}"
            + (f" and path_contains={args.path_contains}" if args.path_contains else "")
        )

    label_counts = Counter([int(r["label"]) for r in selected_records])
    modality_counts = Counter([str(r.get("modality", "unknown")).lower() for r in selected_records])
    patient_ids = sorted({str(r["patient_id"]) for r in selected_records})

    print(f"Dataset key: {dataset_key}")
    if args.path_contains:
        print(f"Path filter: {args.path_contains}")
    print("Slice records:", len(selected_records))
    print("Patients:", len(patient_ids))
    print("Label counts:", dict(label_counts))
    print("Modality counts:", dict(modality_counts))
    print("Using checkpoint:", checkpoint_path)

    val_transform = build_eval_transform(
        target_size=args.target_size,
        center_crop_size=args.center_crop_size,
    )

    dataset = MRISliceDataset(
        selected_records,
        target_size=args.target_size,
        transform=val_transform,
        channel_mode="2.5d",
    )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    predictor = Predictor.load_from_checkpoint(str(checkpoint_path), device)
    outputs = predictor.collect_predictions(loader)

    slice_metrics = safe_metrics(
        y_true=outputs["true_labels"],
        y_pred=outputs["predicted_labels"],
        probabilities=outputs["probabilities"],
    )

    # Use same aggregation defaults as the rest of the project.
    patient_preds = topk_patient_prediction(
        records=selected_records,
        probs=outputs["probabilities"],
        k=20,
        threshold=0.70,
        method="median",
        min_suspicious_slices=8,
        suspicious_prob_threshold=0.90,
        min_suspicious_fraction=0.30,
    )

    patient_true = {}
    for r in selected_records:
        patient_true[str(r["patient_id"])] = int(r["label"])

    patient_y_true = [patient_true[pid] for pid in sorted(patient_true.keys())]
    patient_y_pred = [int(patient_preds.get(pid, 0)) for pid in sorted(patient_true.keys())]

    patient_prob_proxy = [[1 - p, p] for p in patient_y_pred]
    patient_metrics = safe_metrics(
        y_true=patient_y_true,
        y_pred=patient_y_pred,
        probabilities=patient_prob_proxy,
    )

    per_modality_positive_rate = defaultdict(dict)
    probs = outputs["probabilities"]
    for rec, prob, pred in zip(selected_records, probs, outputs["predicted_labels"]):
        mod = str(rec.get("modality", "unknown")).lower()
        if "count" not in per_modality_positive_rate[mod]:
            per_modality_positive_rate[mod]["count"] = 0
            per_modality_positive_rate[mod]["predicted_tumor"] = 0
            per_modality_positive_rate[mod]["mean_tumor_prob"] = 0.0

        per_modality_positive_rate[mod]["count"] += 1
        per_modality_positive_rate[mod]["predicted_tumor"] += int(pred == 1)
        per_modality_positive_rate[mod]["mean_tumor_prob"] += float(prob[1])

    for mod, block in per_modality_positive_rate.items():
        c = max(1, block["count"])
        block["predicted_tumor_rate"] = float(block["predicted_tumor"] / c)
        block["mean_tumor_prob"] = float(block["mean_tumor_prob"] / c)

    report = {
        "dataset": dataset_key,
        "path_filter": args.path_contains,
        "checkpoint": str(checkpoint_path),
        "num_slice_records": len(selected_records),
        "num_patients": len(patient_ids),
        "label_counts": dict(label_counts),
        "modality_counts": dict(modality_counts),
        "slice_metrics": slice_metrics,
        "patient_metrics": patient_metrics,
        "patient_prediction_counts": dict(Counter(patient_y_pred)),
        "per_modality_summary": dict(per_modality_positive_rate),
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    title_prefix = f"{dataset_key.upper()}"
    if args.path_contains:
        title_prefix = f"{title_prefix} ({args.path_contains})"

    print_metric_block(f"{title_prefix} Slice-Level Report", slice_metrics)
    print_metric_block(f"{title_prefix} Patient-Level Report", patient_metrics)
    print("\nPatient prediction counts:", report["patient_prediction_counts"])
    print("Report saved to:", output_path)


if __name__ == "__main__":
    main()
