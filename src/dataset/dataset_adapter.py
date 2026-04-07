from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from src.preprocessing.modality_detection import detect_modality


@dataclass
class VolumeSample:
    volume_path: str
    modality: str
    patient_id: str
    label: int
    dataset_name: str
    field_strength_t: Optional[float] = None

    def to_legacy_tuple(self):
        return (self.volume_path, self.label, self.patient_id, self.dataset_name)


class DatasetAdapter(ABC):
    def __init__(self, root: str, dataset_name: str):
        self.root = Path(root)
        self.dataset_name = dataset_name

    @abstractmethod
    def scan(self) -> List[VolumeSample]:
        raise NotImplementedError


class BratsAdapter(DatasetAdapter):
    def __init__(self, root: str):
        super().__init__(root=root, dataset_name="brats")

    def scan(self) -> List[VolumeSample]:
        samples: List[VolumeSample] = []
        for patient in self.root.iterdir():
            if not patient.is_dir():
                continue
            for file in patient.glob("*.nii*"):
                modality = detect_modality(str(file))
                samples.append(
                    VolumeSample(
                        volume_path=str(file),
                        modality=modality,
                        patient_id=patient.name,
                        label=1,
                        dataset_name=self.dataset_name,
                    )
                )
        return samples


class OasisAdapter(DatasetAdapter):
    def __init__(self, root: str):
        super().__init__(root=root, dataset_name="oasis")

    def scan(self) -> List[VolumeSample]:
        samples: List[VolumeSample] = []
        for file in self.root.rglob("*.nii*"):
            name = file.name.lower()
            if "seg" in name:
                continue
            samples.append(
                VolumeSample(
                    volume_path=str(file),
                    modality=detect_modality(str(file)),
                    patient_id=file.stem,
                    label=0,
                    dataset_name=self.dataset_name,
                )
            )
        return samples


class IXIAdapter(DatasetAdapter):
    def __init__(self, root: str):
        super().__init__(root=root, dataset_name="ixi")

    def scan(self) -> List[VolumeSample]:
        samples: List[VolumeSample] = []
        for file in self.root.rglob("*.nii*"):
            modality = detect_modality(str(file))
            patient_id = file.stem.split("-")[0]
            samples.append(
                VolumeSample(
                    volume_path=str(file),
                    modality=modality,
                    patient_id=patient_id,
                    label=0,
                    dataset_name=self.dataset_name,
                )
            )
        return samples
