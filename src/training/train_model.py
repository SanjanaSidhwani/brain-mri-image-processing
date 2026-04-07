import os
import sys
from pathlib import Path
import argparse
from datetime import datetime
import random
import pickle
import gzip

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model_factory import create_model
from src.dataset.mri_dataset import MRISliceDataset, create_train_val_dataloaders
from src.dataset.split_utils import split_dataset_by_patient, split_dataset_by_patient_balanced_val
from src.dataset.input_transforms import build_train_transform, build_eval_transform
from src.training.trainer import Trainer


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train MRI Classification Model"
    )

    parser.add_argument("--architecture", type=str, default="cnn")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--dropout_rate", type=float, default=0.5)

    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--use_scheduler", action="store_true")
    parser.add_argument("--scheduler_step", type=int, default=15)
    parser.add_argument("--scheduler_gamma", type=float, default=0.1)

    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--balanced_validation", action="store_true", 
                        help="Use balanced validation set (50%% tumor, 50%% normal)")
    parser.add_argument("--target_size", type=int, default=224)
    parser.add_argument("--center_crop_size", type=int, default=180)
    parser.add_argument(
        "--channel_mode",
        type=str,
        default="auto",
        choices=["auto", "single", "2.5d", "multimodal"],
        help="Input channel construction mode"
    )
    parser.add_argument(
        "--modality_dropout",
        type=float,
        default=0.0,
        help="Probability of dropping each modality channel during training in multimodal mode"
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--random_seed", type=int, default=42)

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/dataset_records.pkl.gz",
        help="Path to serialized dataset records"
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="outputs/checkpoints"
    )

    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default=None
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"]
    )

    return parser.parse_args()


def setup_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)

    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    return device


def load_dataset_records(dataset_path: str):

    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path}\n"
            "You must generate dataset_records.pkl using the dataset pipeline."
        )

    if str(dataset_path).endswith(".gz"):
        with gzip.open(dataset_path, "rb") as f:
            dataset_records = pickle.load(f)
    else:
        with open(dataset_path, "rb") as f:
            dataset_records = pickle.load(f)

    if not isinstance(dataset_records, list):
        raise ValueError("Dataset file must contain a list of dataset records")

    return dataset_records


def create_checkpoint_path(checkpoint_dir: str, checkpoint_name: str = None):

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if checkpoint_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"mri_classifier_{timestamp}.pth"

    return str(checkpoint_dir / checkpoint_name)


def print_dataset_info(train_dataset, val_dataset):

    print("\n" + "=" * 60)
    print("Dataset Information")
    print("=" * 60)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    train_patients = set(
        train_dataset.dataset_records[i]["patient_id"]
        for i in range(len(train_dataset))
    )

    val_patients = set(
        val_dataset.dataset_records[i]["patient_id"]
        for i in range(len(val_dataset))
    )

    print(f"Training patients: {len(train_patients)}")
    print(f"Validation patients: {len(val_patients)}")

    print("=" * 60 + "\n")


def print_model_info(model, device):

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "=" * 60)
    print("Model Information")
    print("=" * 60)

    print(f"Architecture: {model.__class__.__name__}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Device: {device}")

    print("=" * 60 + "\n")


def main():

    args = parse_arguments()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    device = setup_device(args.device)

    print("\n" + "=" * 60)
    print("MRI Slice Classification - Training Pipeline")
    print("=" * 60)

    print("\n[Step 1/7] Loading dataset records...")

    dataset_records = load_dataset_records(args.dataset_path)

    print(f"✓ Loaded {len(dataset_records)} slice records")

    print("\n[Step 2/7] Splitting dataset...")

    if args.balanced_validation:
        train_records, val_records = split_dataset_by_patient_balanced_val(
            dataset=dataset_records,
            train_ratio=args.train_ratio,
            val_balance_ratio=0.5,
            seed=args.random_seed
        )
        print("Using balanced validation set (50%% tumor, 50%% normal)")
    else:
        train_records, val_records = split_dataset_by_patient(
            dataset=dataset_records,
            train_ratio=args.train_ratio,
            seed=args.random_seed
        )

    print("\n[Step 3/7] Creating datasets...")

    train_dataset = MRISliceDataset(
        dataset_records=train_records,
        target_size=args.target_size,
        transform=None,
        channel_mode=args.channel_mode,
        modality_dropout=args.modality_dropout,
    )

    val_dataset = MRISliceDataset(
        dataset_records=val_records,
        target_size=args.target_size,
        transform=None,
        channel_mode=args.channel_mode,
        modality_dropout=0.0,
    )

    input_channels = train_dataset.input_channels

    train_transform = build_train_transform(
        target_size=args.target_size,
        center_crop_size=args.center_crop_size,
        num_channels=input_channels,
    )

    val_transform = build_eval_transform(
        target_size=args.target_size,
        center_crop_size=args.center_crop_size,
        num_channels=input_channels,
    )

    train_dataset.transform = train_transform
    val_dataset.transform = val_transform

    print_dataset_info(train_dataset, val_dataset)

    print("[Step 4/7] Creating dataloaders...")

    train_loader, val_loader = create_train_val_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    print("[Step 5/7] Initializing model...")

    model = create_model(
        architecture=args.architecture,
        num_classes=args.num_classes,
        dropout_rate=args.dropout_rate,
        input_channels=input_channels,
    )

    print_model_info(model, device)

    from collections import Counter

    labels = [r["label"] for r in train_records]
    counts = Counter(labels)

    total = sum(counts.values())

    weight_0 = total / counts[0]
    weight_1 = total / counts[1]

    class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    scheduler = None

    if args.use_scheduler:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.scheduler_step,
            gamma=args.scheduler_gamma
        )

    print("[Step 6/7] Initializing trainer...")

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler
    )

    print("[Step 7/7] Starting training...\n")

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs
    )

    print("\nTraining complete.")

    checkpoint_path = create_checkpoint_path(
        args.checkpoint_dir,
        args.checkpoint_name
    )

    trainer.save_checkpoint(
        filepath=checkpoint_path,
        epoch=args.num_epochs,
        args=vars(args)
    )

    print(f"\nCheckpoint saved: {checkpoint_path}")


if __name__ == "__main__":
    main()