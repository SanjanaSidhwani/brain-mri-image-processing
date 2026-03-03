import numpy as np
from typing import List, Tuple, Dict, Any


def get_unique_patients(dataset: List[Dict[str, Any]]) -> List[str]:
  
    if not dataset:
        raise ValueError("Dataset is empty. Cannot extract patients.")

    unique_patients = {record["patient_id"] for record in dataset}
    return list(unique_patients)


def split_patients(
    unique_patients: List[str],
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List[str], List[str]]:

    if len(unique_patients) < 2:
        raise ValueError("At least 2 patients are required for splitting.")

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    patients = unique_patients.copy()

    rng = np.random.default_rng(seed)
    rng.shuffle(patients)

    split_index = int(len(patients) * train_ratio)

    train_patients = patients[:split_index]
    val_patients = patients[split_index:]

    return train_patients, val_patients


def split_dataset_by_patient(
    dataset: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    

    if not dataset:
        raise ValueError("Dataset is empty. Cannot split.")

    unique_patients = get_unique_patients(dataset)

    train_patients, val_patients = split_patients(
        unique_patients,
        train_ratio=train_ratio,
        seed=seed
    )

    train_patients_set = set(train_patients)
    val_patients_set = set(val_patients)

    train_dataset = []
    val_dataset = []

    for record in dataset:
        patient_id = record["patient_id"]

        if patient_id in train_patients_set:
            train_dataset.append(record)
        elif patient_id in val_patients_set:
            val_dataset.append(record)
        else:
            raise RuntimeError(
                f"Patient {patient_id} not found in train or validation sets."
            )

    train_ids = {r["patient_id"] for r in train_dataset}
    val_ids = {r["patient_id"] for r in val_dataset}

    if train_ids & val_ids:
        raise RuntimeError("Data leakage detected! Overlapping patients found.")

    return train_dataset, val_dataset