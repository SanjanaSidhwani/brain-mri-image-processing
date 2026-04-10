import argparse
import gzip
import pickle
import random
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.input_transforms import build_eval_transform
from src.dataset.mri_dataset import MRISliceDataset
from src.evaluation.metrics import compute_classification_metrics
from src.evaluation.predictor import Predictor
from src.evaluation.report import generate_report


def parse_args():
    parser = argparse.ArgumentParser(description="Run small balanced evaluation subset")
    parser.add_argument("--dataset_path", type=str, default="data/dataset_records.pkl.gz")
    parser.add_argument("--checkpoint_path", type=str, default="outputs/checkpoints/mri_classifier_20260331_015015.pth")
    parser.add_argument("--per_class", type=int, default=100, help="Number of records to sample per class")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with gzip.open(dataset_path, "rb") as f:
        records = pickle.load(f)

    class0 = [r for r in records if r.get("label") == 0]
    class1 = [r for r in records if r.get("label") == 1]

    if len(class0) < args.per_class or len(class1) < args.per_class:
        raise ValueError(
            f"Not enough samples for requested per_class={args.per_class}. "
            f"Have class0={len(class0)}, class1={len(class1)}"
        )

    subset = random.sample(class0, args.per_class) + random.sample(class1, args.per_class)
    random.shuffle(subset)

    print(f"Subset size: {len(subset)}")
    print(f"Subset distribution: {Counter([r['label'] for r in subset])}")

    val_transform = build_eval_transform(target_size=224, center_crop_size=180)
    dataset = MRISliceDataset(
        subset,
        target_size=224,
        transform=val_transform,
        channel_mode="2.5d",
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    predictor = Predictor.load_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        device=device,
    )
    outputs = predictor.collect_predictions(loader)

    metrics = compute_classification_metrics(
        outputs["true_labels"],
        outputs["predicted_labels"],
        outputs["probabilities"],
    )

    print("\n===== Slice-Level Metrics (balanced subset) =====")
    generate_report(metrics)


if __name__ == "__main__":
    main()
