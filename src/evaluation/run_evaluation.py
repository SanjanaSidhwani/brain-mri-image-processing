import torch
from torch.utils.data import DataLoader
from src.aggregation.topk_aggregation import topk_patient_prediction, get_patient_labels
from src.evaluation.predictor import Predictor
from src.evaluation.metrics import compute_classification_metrics
from src.evaluation.report import generate_report

from src.dataset.mri_dataset import MRISliceDataset
from src.dataset.split_utils import split_dataset_by_patient

import pickle
import gzip
from collections import Counter


def load_dataset(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_records = load_dataset("data/dataset_records.pkl.gz")

    train_records, val_records = split_dataset_by_patient(dataset_records)

    print("\n===== DATA DISTRIBUTION =====")
    print("Train:", Counter([r["label"] for r in train_records]))
    print("Val:", Counter([r["label"] for r in val_records]))
    print("================================\n")

    val_dataset = MRISliceDataset(val_records)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    predictor = Predictor.load_from_checkpoint(
        checkpoint_path="outputs/checkpoints/mri_classifier_20260322_011228.pth",
        device=device,
    )

    
    outputs = predictor.collect_predictions(val_loader)

    
    patient_preds = topk_patient_prediction(
        records=val_records,
        probs=outputs["probabilities"],
        k=5
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


if __name__ == "__main__":
    main()