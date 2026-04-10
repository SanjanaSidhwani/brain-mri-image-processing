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


def split_dataset_by_patient_balanced_val(
    dataset: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    val_balance_ratio: float = 0.5,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split dataset by patient while ensuring balanced validation set.
    
    Args:
        dataset: Full dataset records
        train_ratio: Ratio of patients to use for training (e.g., 0.8)
        val_balance_ratio: Target ratio for positive class in validation (e.g., 0.5 for 50/50)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_dataset, balanced_val_dataset)
    """
    if not dataset:
        raise ValueError("Dataset is empty. Cannot split.")

    # First, split patients by train/val ratio
    unique_patients = get_unique_patients(dataset)
    train_patients, val_patients = split_patients(
        unique_patients,
        train_ratio=train_ratio,
        seed=seed
    )

    train_patients_set = set(train_patients)
    val_patients_set = set(val_patients)

    # Separate train and val records
    train_dataset = []
    val_dataset = []

    for record in dataset:
        patient_id = record["patient_id"]
        if patient_id in train_patients_set:
            train_dataset.append(record)
        elif patient_id in val_patients_set:
            val_dataset.append(record)

    # Balance validation set by class while maintaining patient separation
    val_positive = [r for r in val_dataset if r["label"] == 1]
    val_negative = [r for r in val_dataset if r["label"] == 0]

    # Shuffle within class buckets so truncation does not depend on record order.
    rng = np.random.default_rng(seed)
    rng.shuffle(val_positive)
    rng.shuffle(val_negative)

    # Calculate how many of each class we need for the target ratio
    target_positive_count = int(len(val_dataset) * val_balance_ratio)
    target_negative_count = len(val_dataset) - target_positive_count

    # Limit to available samples
    val_positive = val_positive[:target_positive_count]
    val_negative = val_negative[:target_negative_count]

    balanced_val_dataset = val_positive + val_negative

    # Verify no data leakage
    train_ids = {r["patient_id"] for r in train_dataset}
    val_ids = {r["patient_id"] for r in balanced_val_dataset}

    if train_ids & val_ids:
        raise RuntimeError("Data leakage detected! Overlapping patients found.")

    return train_dataset, balanced_val_dataset