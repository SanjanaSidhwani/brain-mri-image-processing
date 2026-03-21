import torch
from torch.utils.data import DataLoader

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

    metrics = compute_classification_metrics(
        outputs["true_labels"],
        outputs["predicted_labels"]
    )

    generate_report(metrics)


if __name__ == "__main__":
    main()