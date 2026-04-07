import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional

from .input_transforms import build_patient_index, transform_record


class MRISliceDataset(Dataset):

    def __init__(
        self,
        dataset_records: List[Dict],
        target_size: int = 224,
        transform: Optional[callable] = None,
        channel_mode: str = "auto",
        modality_order: Optional[List[str]] = None,
        modality_dropout: float = 0.0,
    ):
        self.dataset_records = dataset_records
        self.target_size = target_size
        self.transform = transform
        self.channel_mode = channel_mode
        self.modality_order = modality_order
        self.modality_dropout = modality_dropout

        self.patient_index = build_patient_index(dataset_records)
        self.modalities = sorted({r.get("modality") for r in dataset_records if r.get("modality")})
        self.resolved_channel_mode = self._resolve_channel_mode(channel_mode)
        self.input_channels = self._resolve_input_channels()

    def _resolve_channel_mode(self, channel_mode: str) -> str:
        mode = channel_mode.lower()
        if mode != "auto":
            return mode

        if not self.modalities:
            return "2.5d"

        if len(self.modalities) == 1:
            return "single"

        return "multimodal"

    def _resolve_input_channels(self) -> int:
        if self.resolved_channel_mode == "single":
            return 1

        if self.resolved_channel_mode == "multimodal":
            if self.modality_order:
                return len(self.modality_order)
            return max(1, len(self.modalities))

        return 3

    def __len__(self) -> int:
        return len(self.dataset_records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        record = self.dataset_records[idx]

        image = transform_record(
            record=record,
            patient_index=self.patient_index,
            target_size=self.target_size,
            pre_resize=(self.transform is None),
            apply_skull_strip=True,
            channel_mode=self.resolved_channel_mode,
            modality_order=self.modality_order,
            modality_dropout_p=self.modality_dropout,
        )

        image = self._to_tensor(image)

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(record["label"], dtype=torch.long)

        return image, label

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)

        # Convert HWC -> CHW only when array clearly looks channel-last.
        if image.ndim == 3 and image.shape[0] > 16 and image.shape[-1] <= 16:
            image = np.transpose(image, (2, 0, 1))

        tensor = torch.from_numpy(image).float()

        return tensor

    def get_patient_id(self, idx: int) -> str:
        return self.dataset_records[idx]["patient_id"]

    def get_slice_index(self, idx: int) -> int:
        return self.dataset_records[idx]["slice_index"]


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False
) -> DataLoader:

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )

    return dataloader


def create_train_val_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int = 16,
    num_workers: int = 0,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:

    train_loader = create_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = create_dataloader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    return train_loader, val_loader


def get_class_distribution(dataset: Dataset) -> Dict[int, int]:

    class_counts = {}

    for idx in range(len(dataset)):
        _, label = dataset[idx]

        if isinstance(label, torch.Tensor):
            label = label.item()

        class_counts[label] = class_counts.get(label, 0) + 1

    return class_counts


def compute_class_weights(dataset: Dataset) -> torch.Tensor:

    class_dist = get_class_distribution(dataset)

    num_classes = len(class_dist)
    total_samples = sum(class_dist.values())

    weights = torch.zeros(num_classes)

    for class_id, count in class_dist.items():
        weights[class_id] = total_samples / (num_classes * count)

    return weights


if __name__ == "__main__":
    from dataset_builder import build_dataset_from_volumes
    from split_utils import split_dataset_by_patient

    sample_volumes = [
        ("path/to/volume1.nii.gz", 0, "patient_001", "dataset1"),
        ("path/to/volume2.nii.gz", 1, "patient_002", "dataset1"),
    ]

    full_dataset = build_dataset_from_volumes(sample_volumes)

    train_records, val_records = split_dataset_by_patient(
        full_dataset,
        train_ratio=0.8,
        seed=42
    )

    train_dataset = MRISliceDataset(train_records, target_size=224)
    val_dataset = MRISliceDataset(val_records, target_size=224)

    train_loader, val_loader = create_train_val_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=16,
        num_workers=0
    )

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

    images, labels = next(iter(train_loader))
    print(f"\nSample batch:")
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Image dtype: {images.dtype}")
    print(f"Labels dtype: {labels.dtype}")

    class_dist = get_class_distribution(train_dataset)
    print(f"\nTraining class distribution: {class_dist}")

    class_weights = compute_class_weights(train_dataset)
    print(f"Class weights: {class_weights}")